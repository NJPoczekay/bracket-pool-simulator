"""Deterministic offline report generation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bracket_sim.domain.bracket_graph import BracketGraph, build_bracket_graph
from bracket_sim.domain.constraints import validate_constraints
from bracket_sim.domain.models import (
    ChampionOddsRow,
    ChampionSensitivityRow,
    EntryReportRow,
    GameOutcomeSensitivityRow,
    PivotalGameRow,
    PoolEntry,
    ReportBundleResult,
    ReportConfig,
    ReportSummary,
    Team,
    TeamAdvancementOddsRow,
)
from bracket_sim.domain.scoring import (
    aggregate_win_share_totals,
    build_predicted_wins_matrix,
    build_team_seeds_array,
    score_entries,
    validate_entries,
)
from bracket_sim.domain.scoring_systems import resolve_scoring_spec
from bracket_sim.domain.simulator import simulate_tournament
from bracket_sim.infrastructure.storage.normalized_loader import load_normalized_input
from bracket_sim.infrastructure.storage.report_bundle import (
    build_report_artifact_paths,
    build_report_manifest,
    ensure_fresh_report_output_dir,
    ensure_report_output_dir,
    generate_report_id,
    write_champion_sensitivity_csv,
    write_entry_summary_csv,
    write_game_outcome_sensitivity_csv,
    write_pivotal_games_csv,
    write_report_manifest,
    write_report_summary,
    write_team_advancement_csv,
)

_WIN_THRESHOLDS = np.arange(1, 7, dtype=np.int16)
_TOP_ENTRY_LIMIT = 5
_TOP_CHAMPION_LIMIT = 8


@dataclass
class _ReportAccumulator:
    entry_win_share_totals: npt.NDArray[np.float64]
    entry_score_totals: npt.NDArray[np.int64]
    champion_counts: npt.NDArray[np.int64]
    team_advancement_counts: npt.NDArray[np.int64]
    champion_entry_win_share_totals: npt.NDArray[np.float64]
    champion_entry_score_totals: npt.NDArray[np.int64]
    champion_entry_sim_counts: npt.NDArray[np.int64]
    game_outcome_entry_win_share_totals: npt.NDArray[np.float64]
    game_outcome_entry_score_totals: npt.NDArray[np.int64]
    game_outcome_sim_counts: npt.NDArray[np.int64]


@dataclass(frozen=True)
class _GameOutcomeMetadata:
    outcome_index: int
    game_id: str
    round: int
    round_game_number: int
    game_label: str
    outcome_team_id: str
    outcome_team_name: str


def generate_reports(config: ReportConfig) -> ReportBundleResult:
    """Generate deterministic report artifacts from normalized local inputs."""

    normalized = load_normalized_input(config.input_dir)
    graph = build_bracket_graph(teams=normalized.teams, games=normalized.games)

    constraints_by_game_id = validate_constraints(
        constraints=normalized.constraints,
        graph=graph,
    )

    validate_entries(entries=normalized.entries, graph=graph)
    entry_ids, team_ids, predicted_wins = build_predicted_wins_matrix(
        entries=normalized.entries,
        graph=graph,
    )
    scoring_spec = resolve_scoring_spec(config.scoring_system)
    team_seeds = build_team_seeds_array(team_ids=team_ids, teams=normalized.teams)
    team_index = {team_id: idx for idx, team_id in enumerate(team_ids)}
    game_outcome_metadata, outcome_index_by_key = _build_game_outcome_metadata(
        graph=graph,
        teams=normalized.teams,
        team_index=team_index,
    )

    rating_records_by_team_id = {
        record.team_id: record for record in normalized.ratings.records
    }
    missing_team_ids = sorted(set(team_ids) - set(rating_records_by_team_id))
    if missing_team_ids:
        msg = f"Missing rating for team id(s): {missing_team_ids[:5]}"
        raise ValueError(msg)

    artifact_paths = build_report_artifact_paths(config.output_dir)
    ensure_report_output_dir(config.output_dir)
    ensure_fresh_report_output_dir(artifact_paths)
    report_id = generate_report_id(config=config)

    accumulator = _empty_accumulator(
        n_entries=len(normalized.entries),
        n_teams=len(team_ids),
        n_game_outcomes=len(game_outcome_metadata),
    )
    total_batches = math.ceil(config.n_sims / config.effective_batch_size)

    for batch_index in range(total_batches):
        batch_n_sims = min(
            config.effective_batch_size,
            config.n_sims - (batch_index * config.effective_batch_size),
        )
        batch_seed = _derive_batch_seed(
            seed=config.seed,
            batch_index=batch_index,
            total_batches=total_batches,
        )
        simulation = simulate_tournament(
            graph=graph,
            rating_records_by_team_id=rating_records_by_team_id,
            constraints_by_game_id=constraints_by_game_id,
            n_sims=batch_n_sims,
            seed=batch_seed,
            point_spread_std_dev=config.rating_scale,
            engine=config.engine,
        )
        scores = score_entries(
            predicted_wins=predicted_wins,
            actual_wins=simulation.team_wins,
            round_values=scoring_spec.round_values,
            team_seeds=team_seeds,
            seed_bonus_rounds=scoring_spec.seed_bonus_rounds,
        )
        _accumulate_batch(
            accumulator=accumulator,
            team_wins=simulation.team_wins,
            champions=simulation.champions,
            game_winners=simulation.game_winners,
            scores=scores,
            outcome_index_by_key=outcome_index_by_key,
        )

    team_advancement_rows = _build_team_advancement_rows(
        teams=normalized.teams,
        team_ids=team_ids,
        team_index=team_index,
        team_advancement_counts=accumulator.team_advancement_counts,
        n_sims=config.n_sims,
    )
    entry_summary_rows = _build_entry_summary_rows(
        entries=normalized.entries,
        entry_win_share_totals=accumulator.entry_win_share_totals,
        entry_score_totals=accumulator.entry_score_totals,
        n_sims=config.n_sims,
    )
    champion_odds_rows = _build_champion_odds_rows(
        teams=normalized.teams,
        team_ids=team_ids,
        team_index=team_index,
        champion_counts=accumulator.champion_counts,
        n_sims=config.n_sims,
    )
    champion_sensitivity_rows = _build_champion_sensitivity_rows(
        teams=normalized.teams,
        entries=normalized.entries,
        team_ids=team_ids,
        team_index=team_index,
        entry_rows=entry_summary_rows,
        champion_counts=accumulator.champion_counts,
        champion_entry_win_share_totals=accumulator.champion_entry_win_share_totals,
        champion_entry_score_totals=accumulator.champion_entry_score_totals,
        champion_entry_sim_counts=accumulator.champion_entry_sim_counts,
        n_sims=config.n_sims,
    )
    game_outcome_sensitivity_rows = _build_game_outcome_sensitivity_rows(
        entries=normalized.entries,
        entry_rows=entry_summary_rows,
        game_outcome_metadata=game_outcome_metadata,
        game_outcome_entry_win_share_totals=accumulator.game_outcome_entry_win_share_totals,
        game_outcome_entry_score_totals=accumulator.game_outcome_entry_score_totals,
        game_outcome_sim_counts=accumulator.game_outcome_sim_counts,
        n_sims=config.n_sims,
    )
    pivotal_game_rows = _build_pivotal_game_rows(
        game_outcome_sensitivity_rows=game_outcome_sensitivity_rows,
    )

    summary = ReportSummary(
        report_id=report_id,
        output_dir=config.output_dir,
        n_sims=config.n_sims,
        seed=config.seed,
        engine=config.engine,
        batch_size=config.effective_batch_size,
        entry_count=len(entry_summary_rows),
        team_count=len(team_advancement_rows),
        top_entries=entry_summary_rows[:_TOP_ENTRY_LIMIT],
        top_champions=champion_odds_rows[:_TOP_CHAMPION_LIMIT],
    )

    artifacts = [
        write_report_summary(artifact_paths.summary_path, summary),
        write_team_advancement_csv(artifact_paths.team_advancement_path, team_advancement_rows),
        write_entry_summary_csv(artifact_paths.entry_summary_path, entry_summary_rows),
        write_champion_sensitivity_csv(
            artifact_paths.champion_sensitivity_path,
            champion_sensitivity_rows,
        ),
        write_game_outcome_sensitivity_csv(
            artifact_paths.game_outcome_sensitivity_path,
            game_outcome_sensitivity_rows,
        ),
        write_pivotal_games_csv(
            artifact_paths.pivotal_games_path,
            pivotal_game_rows,
        ),
    ]
    manifest = build_report_manifest(
        config=config,
        report_id=report_id,
        entry_ids=entry_ids,
        team_ids=team_ids,
        artifacts=artifacts,
    )
    write_report_manifest(artifact_paths.manifest_path, manifest)

    return ReportBundleResult(manifest=manifest, summary=summary)


def _empty_accumulator(*, n_entries: int, n_teams: int, n_game_outcomes: int) -> _ReportAccumulator:
    return _ReportAccumulator(
        entry_win_share_totals=np.zeros(n_entries, dtype=np.float64),
        entry_score_totals=np.zeros(n_entries, dtype=np.int64),
        champion_counts=np.zeros(n_teams, dtype=np.int64),
        team_advancement_counts=np.zeros((n_teams, len(_WIN_THRESHOLDS)), dtype=np.int64),
        champion_entry_win_share_totals=np.zeros((n_teams, n_entries), dtype=np.float64),
        champion_entry_score_totals=np.zeros((n_teams, n_entries), dtype=np.int64),
        champion_entry_sim_counts=np.zeros(n_teams, dtype=np.int64),
        game_outcome_entry_win_share_totals=np.zeros(
            (n_game_outcomes, n_entries),
            dtype=np.float64,
        ),
        game_outcome_entry_score_totals=np.zeros((n_game_outcomes, n_entries), dtype=np.int64),
        game_outcome_sim_counts=np.zeros(n_game_outcomes, dtype=np.int64),
    )


def _accumulate_batch(
    *,
    accumulator: _ReportAccumulator,
    team_wins: npt.NDArray[np.int16],
    champions: npt.NDArray[np.int16],
    game_winners: npt.NDArray[np.int16],
    scores: npt.NDArray[np.int32],
    outcome_index_by_key: dict[tuple[int, int], int],
) -> None:
    accumulator.entry_win_share_totals += aggregate_win_share_totals(scores)
    accumulator.entry_score_totals += np.sum(scores, axis=1, dtype=np.int64)
    accumulator.team_advancement_counts += np.sum(
        team_wins[:, :, None] >= _WIN_THRESHOLDS[None, None, :],
        axis=0,
        dtype=np.int64,
    )
    accumulator.champion_counts += np.bincount(
        champions,
        minlength=accumulator.champion_counts.shape[0],
    ).astype(np.int64)

    for champion_idx in np.unique(champions):
        champion_mask = champions == champion_idx
        champion_scores = scores[:, champion_mask]
        champion_count = int(np.sum(champion_mask))
        accumulator.champion_entry_sim_counts[champion_idx] += champion_count
        accumulator.champion_entry_win_share_totals[champion_idx] += aggregate_win_share_totals(
            champion_scores
        )
        accumulator.champion_entry_score_totals[champion_idx] += np.sum(
            champion_scores,
            axis=1,
            dtype=np.int64,
        )

    for game_index in range(game_winners.shape[0]):
        winners = game_winners[game_index]
        for winner_idx in np.unique(winners):
            winner_index = int(winner_idx)
            outcome_index = outcome_index_by_key[(game_index, winner_index)]
            outcome_mask = winners == winner_index
            outcome_count = int(np.sum(outcome_mask))
            accumulator.game_outcome_sim_counts[outcome_index] += outcome_count
            accumulator.game_outcome_entry_win_share_totals[
                outcome_index
            ] += aggregate_win_share_totals(scores[:, outcome_mask])
            accumulator.game_outcome_entry_score_totals[outcome_index] += np.sum(
                scores[:, outcome_mask],
                axis=1,
                dtype=np.int64,
            )


def _build_team_advancement_rows(
    *,
    teams: list[Team],
    team_ids: list[str],
    team_index: dict[str, int],
    team_advancement_counts: npt.NDArray[np.int64],
    n_sims: int,
) -> list[TeamAdvancementOddsRow]:
    valid_team_ids = set(team_ids)
    rows = [
        TeamAdvancementOddsRow(
            team_id=team.team_id,
            team_name=team.name,
            seed=team.seed,
            region=team.region,
            reach_round_of_32=_probability(
                team_advancement_counts[team_index[team.team_id], 0],
                n_sims,
            ),
            reach_sweet_16=_probability(
                team_advancement_counts[team_index[team.team_id], 1],
                n_sims,
            ),
            reach_elite_8=_probability(
                team_advancement_counts[team_index[team.team_id], 2],
                n_sims,
            ),
            reach_final_four=_probability(
                team_advancement_counts[team_index[team.team_id], 3],
                n_sims,
            ),
            reach_title_game=_probability(
                team_advancement_counts[team_index[team.team_id], 4],
                n_sims,
            ),
            win_championship=_probability(
                team_advancement_counts[team_index[team.team_id], 5],
                n_sims,
            ),
        )
        for team in teams
        if team.team_id in valid_team_ids
    ]
    return sorted(rows, key=lambda row: (row.region, row.seed, row.team_name, row.team_id))


def _build_entry_summary_rows(
    *,
    entries: list[PoolEntry],
    entry_win_share_totals: npt.NDArray[np.float64],
    entry_score_totals: npt.NDArray[np.int64],
    n_sims: int,
) -> list[EntryReportRow]:
    rows = [
        EntryReportRow(
            rank=index + 1,
            entry_id=entry.entry_id,
            entry_name=entry.entry_name,
            win_share=float(entry_win_share_totals[index] / n_sims),
            average_score=float(entry_score_totals[index] / n_sims),
        )
        for index, entry in enumerate(entries)
    ]
    sorted_rows = sorted(
        rows,
        key=lambda row: (-row.win_share, -row.average_score, row.entry_name, row.entry_id),
    )
    return [
        row.model_copy(update={"rank": rank})
        for rank, row in enumerate(sorted_rows, start=1)
    ]


def _build_champion_odds_rows(
    *,
    teams: list[Team],
    team_ids: list[str],
    team_index: dict[str, int],
    champion_counts: npt.NDArray[np.int64],
    n_sims: int,
) -> list[ChampionOddsRow]:
    valid_team_ids = set(team_ids)
    rows = [
        ChampionOddsRow(
            rank=index + 1,
            team_id=team.team_id,
            team_name=team.name,
            probability=_probability(champion_counts[team_index[team.team_id]], n_sims),
        )
        for index, team in enumerate(teams)
        if team.team_id in valid_team_ids
    ]
    sorted_rows = sorted(
        rows,
        key=lambda row: (-row.probability, row.team_name, row.team_id),
    )
    return [
        row.model_copy(update={"rank": rank})
        for rank, row in enumerate(sorted_rows, start=1)
    ]


def _build_champion_sensitivity_rows(
    *,
    teams: list[Team],
    entries: list[PoolEntry],
    team_ids: list[str],
    team_index: dict[str, int],
    entry_rows: list[EntryReportRow],
    champion_counts: npt.NDArray[np.int64],
    champion_entry_win_share_totals: npt.NDArray[np.float64],
    champion_entry_score_totals: npt.NDArray[np.int64],
    champion_entry_sim_counts: npt.NDArray[np.int64],
    n_sims: int,
) -> list[ChampionSensitivityRow]:
    team_metadata = {team.team_id: team for team in teams}
    baseline_by_entry_id = {row.entry_id: row for row in entry_rows}
    rows: list[ChampionSensitivityRow] = []

    for team_id in team_ids:
        champion_idx = team_index[team_id]
        champion_simulations = int(champion_entry_sim_counts[champion_idx])
        if champion_simulations == 0:
            continue

        champion_team = team_metadata[team_id]
        champion_probability = _probability(champion_counts[champion_idx], n_sims)
        for entry_index, entry in enumerate(entries):
            entry_row = baseline_by_entry_id[entry.entry_id]
            conditional_win_share = float(
                champion_entry_win_share_totals[champion_idx, entry_index] / champion_simulations
            )
            conditional_average_score = float(
                champion_entry_score_totals[champion_idx, entry_index] / champion_simulations
            )
            rows.append(
                ChampionSensitivityRow(
                    champion_team_id=champion_team.team_id,
                    champion_team_name=champion_team.name,
                    champion_probability=champion_probability,
                    champion_simulations=champion_simulations,
                    entry_rank=entry_row.rank,
                    entry_id=entry_row.entry_id,
                    entry_name=entry_row.entry_name,
                    baseline_win_share=entry_row.win_share,
                    conditional_win_share=conditional_win_share,
                    win_share_delta=conditional_win_share - entry_row.win_share,
                    baseline_average_score=entry_row.average_score,
                    conditional_average_score=conditional_average_score,
                    average_score_delta=conditional_average_score - entry_row.average_score,
                )
            )

    return sorted(
        rows,
        key=lambda row: (
            -row.champion_probability,
            row.champion_team_name,
            row.entry_rank,
            row.entry_name,
            row.entry_id,
        ),
    )


def _build_game_outcome_metadata(
    *,
    graph: BracketGraph,
    teams: list[Team],
    team_index: dict[str, int],
) -> tuple[list[_GameOutcomeMetadata], dict[tuple[int, int], int]]:
    team_metadata = {team.team_id: team for team in teams}
    round_game_numbers: dict[str, int] = {}
    games_by_round: dict[int, list[str]] = defaultdict(list)
    for game_id, game in graph.games_by_id.items():
        games_by_round[game.round].append(game_id)

    for _round_number, game_ids in games_by_round.items():
        for round_game_number, game_id in enumerate(sorted(game_ids), start=1):
            round_game_numbers[game_id] = round_game_number

    metadata: list[_GameOutcomeMetadata] = []
    outcome_index_by_key: dict[tuple[int, int], int] = {}
    for game_index, game_id in enumerate(graph.topological_game_ids):
        game = graph.games_by_id[game_id]
        round_game_number = round_game_numbers[game_id]
        game_label = f"Round {game.round} Game {round_game_number}"
        for outcome_team_id in sorted(graph.possible_teams_by_game_id[game_id]):
            team = team_metadata[outcome_team_id]
            outcome_index = len(metadata)
            metadata.append(
                _GameOutcomeMetadata(
                    outcome_index=outcome_index,
                    game_id=game_id,
                    round=game.round,
                    round_game_number=round_game_number,
                    game_label=game_label,
                    outcome_team_id=outcome_team_id,
                    outcome_team_name=team.name,
                )
            )
            outcome_index_by_key[(game_index, team_index[outcome_team_id])] = outcome_index

    return metadata, outcome_index_by_key


def _build_game_outcome_sensitivity_rows(
    *,
    entries: list[PoolEntry],
    entry_rows: list[EntryReportRow],
    game_outcome_metadata: list[_GameOutcomeMetadata],
    game_outcome_entry_win_share_totals: npt.NDArray[np.float64],
    game_outcome_entry_score_totals: npt.NDArray[np.int64],
    game_outcome_sim_counts: npt.NDArray[np.int64],
    n_sims: int,
) -> list[GameOutcomeSensitivityRow]:
    baseline_by_entry_id = {row.entry_id: row for row in entry_rows}
    rows: list[GameOutcomeSensitivityRow] = []

    for metadata in game_outcome_metadata:
        outcome_simulations = int(game_outcome_sim_counts[metadata.outcome_index])
        if outcome_simulations == 0:
            continue

        outcome_probability = _probability(outcome_simulations, n_sims)
        conditional_win_shares = (
            game_outcome_entry_win_share_totals[metadata.outcome_index] / outcome_simulations
        )
        conditional_average_scores = (
            game_outcome_entry_score_totals[metadata.outcome_index] / outcome_simulations
        )
        outcome_total_swing = float(
            sum(
                abs(
                    float(conditional_win_shares[entry_index])
                    - baseline_by_entry_id[entry.entry_id].win_share
                )
                for entry_index, entry in enumerate(entries)
            )
        )

        for entry_index, entry in enumerate(entries):
            entry_row = baseline_by_entry_id[entry.entry_id]
            conditional_win_share = float(conditional_win_shares[entry_index])
            conditional_average_score = float(conditional_average_scores[entry_index])
            rows.append(
                GameOutcomeSensitivityRow(
                    game_id=metadata.game_id,
                    round=metadata.round,
                    round_game_number=metadata.round_game_number,
                    game_label=metadata.game_label,
                    outcome_team_id=metadata.outcome_team_id,
                    outcome_team_name=metadata.outcome_team_name,
                    outcome_probability=outcome_probability,
                    outcome_simulations=outcome_simulations,
                    entry_rank=entry_row.rank,
                    entry_id=entry_row.entry_id,
                    entry_name=entry_row.entry_name,
                    baseline_win_share=entry_row.win_share,
                    conditional_win_share=conditional_win_share,
                    win_share_delta=conditional_win_share - entry_row.win_share,
                    baseline_average_score=entry_row.average_score,
                    conditional_average_score=conditional_average_score,
                    average_score_delta=conditional_average_score - entry_row.average_score,
                    outcome_total_swing=outcome_total_swing,
                )
            )

    return sorted(
        rows,
        key=lambda row: (
            row.round,
            row.round_game_number,
            row.game_id,
            row.outcome_team_name,
            row.outcome_team_id,
            row.entry_rank,
            row.entry_name,
            row.entry_id,
        ),
    )


def _build_pivotal_game_rows(
    *,
    game_outcome_sensitivity_rows: list[GameOutcomeSensitivityRow],
) -> list[PivotalGameRow]:
    rows_by_outcome: dict[tuple[str, str], list[GameOutcomeSensitivityRow]] = defaultdict(list)
    for row in game_outcome_sensitivity_rows:
        rows_by_outcome[(row.game_id, row.outcome_team_id)].append(row)

    outcome_groups_by_game: dict[str, list[list[GameOutcomeSensitivityRow]]] = defaultdict(list)
    for (game_id, _outcome_team_id), grouped_rows in rows_by_outcome.items():
        outcome_groups_by_game[game_id].append(grouped_rows)

    rows: list[PivotalGameRow] = []
    for outcome_groups in outcome_groups_by_game.values():
        pivotal_group = max(
            outcome_groups,
            key=lambda group: (
                group[0].outcome_total_swing,
                group[0].outcome_probability,
                group[0].outcome_team_name,
                group[0].outcome_team_id,
            ),
        )
        representative = pivotal_group[0]
        top_gainer = max(
            pivotal_group,
            key=lambda row: (row.win_share_delta, -row.entry_rank, row.entry_name, row.entry_id),
        )
        top_loser = min(
            pivotal_group,
            key=lambda row: (row.win_share_delta, row.entry_rank, row.entry_name, row.entry_id),
        )
        rows.append(
            PivotalGameRow(
                rank=len(rows) + 1,
                game_id=representative.game_id,
                round=representative.round,
                round_game_number=representative.round_game_number,
                game_label=representative.game_label,
                pivotal_outcome_team_id=representative.outcome_team_id,
                pivotal_outcome_team_name=representative.outcome_team_name,
                pivotal_outcome_probability=representative.outcome_probability,
                pivotal_outcome_simulations=representative.outcome_simulations,
                pivotal_outcome_total_swing=representative.outcome_total_swing,
                top_gainer_entry_id=top_gainer.entry_id,
                top_gainer_entry_name=top_gainer.entry_name,
                top_gainer_win_share_delta=top_gainer.win_share_delta,
                top_loser_entry_id=top_loser.entry_id,
                top_loser_entry_name=top_loser.entry_name,
                top_loser_win_share_delta=top_loser.win_share_delta,
            )
        )

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -row.pivotal_outcome_total_swing,
            row.round,
            row.round_game_number,
            row.game_id,
        ),
    )
    return [
        row.model_copy(update={"rank": rank})
        for rank, row in enumerate(sorted_rows, start=1)
    ]


def _derive_batch_seed(*, seed: int, batch_index: int, total_batches: int) -> int:
    if total_batches == 1 and batch_index == 0:
        return seed

    sequence = np.random.SeedSequence([seed, batch_index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _probability(count: np.int64 | int, n_sims: int) -> float:
    return float(int(count) / n_sims)
