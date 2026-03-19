"""Shared Bracket Lab runtime and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from bracket_sim.application.bracket_completion import (
    derive_region_champion_game_ids,
    editable_bracket_to_entry,
)
from bracket_sim.domain.bracket_graph import BracketGraph, build_bracket_graph
from bracket_sim.domain.bracket_lab_models import PlayInSlot
from bracket_sim.domain.constraints import validate_constraints
from bracket_sim.domain.models import EntryPick, PoolEntry, RatingRecord
from bracket_sim.domain.product_models import (
    BracketEditPick,
    EditableBracket,
    PickDiagnostic,
    PickDiagnosticTag,
    PoolSettings,
)
from bracket_sim.domain.scoring import (
    build_predicted_wins_matrix,
    score_entries,
    validate_entries,
)
from bracket_sim.domain.scoring_systems import (
    ScoringSpec,
    ScoringSystemKey,
    resolve_scoring_spec,
)
from bracket_sim.domain.simulator import canonical_team_order, simulate_tournament
from bracket_sim.infrastructure.storage.bracket_lab_prepared_loader import (
    BracketLabPreparedInput,
    load_bracket_lab_prepared_input,
)
from bracket_sim.infrastructure.storage.cache_keys import build_cache_key, capture_dataset_hash

ANALYSIS_N_SIMS = 100_000
ANALYSIS_BATCH_SIZE = 1_000
BRACKET_LAB_POINT_SPREAD_STD_DEV = 11.0
DEFAULT_POOL_SETTINGS = PoolSettings(
    pool_size=10,
    scoring_system=ScoringSystemKey.ESPN,
)


@dataclass(frozen=True)
class BracketLabRuntime:
    input_dir: Path
    prepared: BracketLabPreparedInput
    graph: BracketGraph
    constraints_by_game_id: dict[str, str]
    dataset_hash: str
    rating_records_by_team_id: dict[str, RatingRecord]
    team_rank_by_team_id: dict[str, int]
    team_ids: list[str]
    team_index: dict[str, int]
    team_seeds: npt.NDArray[np.int16]
    public_pick_weights_by_game: dict[str, dict[str, float]]
    region_champion_game_ids: dict[str, str]


@dataclass(frozen=True)
class BracketFieldEvaluationContext:
    """Reusable fixed-field evaluation context for one optimization request."""

    runtime: BracketLabRuntime
    pool_settings: PoolSettings
    scoring_spec: ScoringSpec
    actual_team_wins: npt.NDArray[np.int16]
    opponent_max_scores: npt.NDArray[np.int32] | None
    opponent_tie_counts: npt.NDArray[np.int16] | None

    @property
    def n_sims(self) -> int:
        return int(self.actual_team_wins.shape[0])

    def evaluate_brackets(
        self,
        brackets: list[EditableBracket],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        predicted_wins = build_predicted_wins_for_brackets(
            brackets=brackets,
            team_index=self.runtime.team_index,
        )
        scores = score_entries(
            predicted_wins=predicted_wins,
            actual_wins=self.actual_team_wins,
            round_values=self.scoring_spec.round_values,
            team_seeds=self.runtime.team_seeds,
            seed_bonus_rounds=self.scoring_spec.seed_bonus_rounds,
        )
        user_win_shares = user_win_shares_against_field(
            user_scores=scores,
            opponent_max_scores=self.opponent_max_scores,
            opponent_tie_counts=self.opponent_tie_counts,
        )
        win_probabilities = np.mean(user_win_shares, axis=1, dtype=np.float64)
        return win_probabilities, user_win_shares

    def evaluate_bracket(
        self,
        bracket: EditableBracket,
    ) -> tuple[float, npt.NDArray[np.float64]]:
        win_probabilities, user_win_shares = self.evaluate_brackets([bracket])
        return float(win_probabilities[0]), user_win_shares[0]


@dataclass
class PickDiagnosticAccumulator:
    picks: list[BracketEditPick]
    rounds: npt.NDArray[np.int16]
    team_indices: npt.NDArray[np.intp]
    round_by_game_id: dict[str, int]
    team_name_by_id: dict[str, str]
    total_win_share: float
    hit_counts: npt.NDArray[np.int64]
    miss_counts: npt.NDArray[np.int64]
    hit_win_share_totals: npt.NDArray[np.float64]
    miss_win_share_totals: npt.NDArray[np.float64]
    total_sims: int

    @classmethod
    def build(
        cls,
        *,
        bracket: EditableBracket,
        graph: BracketGraph,
        team_index: dict[str, int],
        total_sims: int,
    ) -> PickDiagnosticAccumulator:
        picks = sorted(
            bracket.picks,
            key=lambda pick: (graph.games_by_id[pick.game_id].round, pick.game_id),
        )
        rounds = np.asarray(
            [graph.games_by_id[pick.game_id].round for pick in picks],
            dtype=np.int16,
        )
        team_indices = np.asarray(
            [team_index[pick.winner_team_id or ""] for pick in picks],
            dtype=np.intp,
        )
        return cls(
            picks=picks,
            rounds=rounds,
            team_indices=team_indices,
            round_by_game_id={
                game_id: game.round for game_id, game in graph.games_by_id.items()
            },
            team_name_by_id={
                team_id: team.name for team_id, team in graph.teams_by_id.items()
            },
            total_win_share=0.0,
            hit_counts=np.zeros(len(picks), dtype=np.int64),
            miss_counts=np.zeros(len(picks), dtype=np.int64),
            hit_win_share_totals=np.zeros(len(picks), dtype=np.float64),
            miss_win_share_totals=np.zeros(len(picks), dtype=np.float64),
            total_sims=total_sims,
        )

    @property
    def win_probability(self) -> float:
        return float(self.total_win_share / self.total_sims)

    def accumulate_batch(
        self,
        *,
        team_wins: npt.NDArray[np.int16],
        user_win_shares: npt.NDArray[np.float64],
    ) -> None:
        batch_total_share = float(np.sum(user_win_shares, dtype=np.float64))
        picked_team_wins = team_wins[:, self.team_indices]
        hit_matrix = picked_team_wins >= self.rounds[None, :]
        hit_counts = np.sum(hit_matrix, axis=0, dtype=np.int64)
        hit_share_totals = np.sum(
            hit_matrix * user_win_shares[:, None],
            axis=0,
            dtype=np.float64,
        )

        self.total_win_share += batch_total_share
        self.hit_counts += hit_counts
        self.miss_counts += team_wins.shape[0] - hit_counts
        self.hit_win_share_totals += hit_share_totals
        self.miss_win_share_totals += batch_total_share - hit_share_totals

    def build_pick_diagnostics(self) -> list[PickDiagnostic]:
        overall = self.win_probability
        diagnostics: list[PickDiagnostic] = []
        deltas: list[float] = []
        importance_swings: list[float] = []

        for index, pick in enumerate(self.picks):
            hit_count = int(self.hit_counts[index])
            miss_count = int(self.miss_counts[index])
            survival_probability = float(hit_count / self.total_sims)
            win_if_picked = float(
                self.hit_win_share_totals[index] / hit_count if hit_count > 0 else 0.0
            )
            win_if_missed = float(
                self.miss_win_share_totals[index] / miss_count if miss_count > 0 else 0.0
            )
            delta = win_if_picked - overall
            diagnostics.append(
                PickDiagnostic(
                    game_id=pick.game_id,
                    team_id=pick.winner_team_id or "",
                    team_name=self.team_name_by_id[pick.winner_team_id or ""],
                    round=self.round_by_game_id[pick.game_id],
                    survival_probability=survival_probability,
                    win_probability_if_picked=win_if_picked,
                    delta_win_probability_if_picked=delta,
                    tags=[],
                )
            )
            deltas.append(delta)
            importance_swings.append(abs(win_if_picked - win_if_missed))

        if diagnostics:
            best_index = pick_extreme_index(
                diagnostics=diagnostics,
                values=deltas,
                reverse=True,
            )
            worst_index = pick_extreme_index(
                diagnostics=diagnostics,
                values=deltas,
                reverse=False,
            )
            important_index = pick_extreme_index(
                diagnostics=diagnostics,
                values=importance_swings,
                reverse=True,
            )
            diagnostics[best_index].tags.append(PickDiagnosticTag.BEST_PICK)
            diagnostics[worst_index].tags.append(PickDiagnosticTag.WORST_PICK)
            diagnostics[important_index].tags.append(PickDiagnosticTag.MOST_IMPORTANT)

        return diagnostics


def build_bracket_lab_runtime(input_dir: Path) -> BracketLabRuntime:
    prepared = load_bracket_lab_prepared_input(input_dir)
    graph = build_bracket_graph(teams=prepared.teams, games=prepared.games)
    constraints_by_game_id = validate_constraints(
        constraints=prepared.constraints,
        graph=graph,
    )
    rating_records_by_team_id = build_rating_records_by_team_id(
        prepared=prepared,
        graph=graph,
    )
    team_ids = canonical_team_order(graph)
    team_index = {team_id: idx for idx, team_id in enumerate(team_ids)}
    return BracketLabRuntime(
        input_dir=input_dir,
        prepared=prepared,
        graph=graph,
        constraints_by_game_id=constraints_by_game_id,
        dataset_hash=capture_dataset_hash(input_dir),
        rating_records_by_team_id=rating_records_by_team_id,
        team_rank_by_team_id=build_completion_rank_by_team_id(
            prepared=prepared,
            graph=graph,
            rating_records_by_team_id=rating_records_by_team_id,
        ),
        team_ids=team_ids,
        team_index=team_index,
        team_seeds=np.asarray(
            [graph.teams_by_id[team_id].seed for team_id in team_ids],
            dtype=np.int16,
        ),
        public_pick_weights_by_game=build_public_pick_weights(prepared=prepared),
        region_champion_game_ids=derive_region_champion_game_ids(graph),
    )


def build_field_evaluation_context(
    *,
    runtime: BracketLabRuntime,
    pool_settings: PoolSettings,
    opponent_seed: int,
    simulation_seed: int,
    n_sims: int,
) -> BracketFieldEvaluationContext:
    scoring_spec = resolve_scoring_spec(pool_settings.scoring_system)
    opponent_entries = sample_public_opponents(
        runtime=runtime,
        opponent_count=max(pool_settings.pool_size - 1, 0),
        seed=opponent_seed,
    )
    validate_entries(entries=opponent_entries, graph=runtime.graph)
    simulation = simulate_tournament(
        graph=runtime.graph,
        rating_records_by_team_id=runtime.rating_records_by_team_id,
        constraints_by_game_id=runtime.constraints_by_game_id,
        n_sims=n_sims,
        seed=simulation_seed,
        point_spread_std_dev=BRACKET_LAB_POINT_SPREAD_STD_DEV,
    )

    opponent_max_scores: npt.NDArray[np.int32] | None = None
    opponent_tie_counts: npt.NDArray[np.int16] | None = None
    if opponent_entries:
        _, team_ids, predicted_wins = build_predicted_wins_matrix(
            entries=opponent_entries,
            graph=runtime.graph,
        )
        if team_ids != runtime.team_ids:
            msg = "Bracket field evaluation team ordering drifted from prepared dataset"
            raise RuntimeError(msg)
        opponent_scores = score_entries(
            predicted_wins=predicted_wins,
            actual_wins=simulation.team_wins,
            round_values=scoring_spec.round_values,
            team_seeds=runtime.team_seeds,
            seed_bonus_rounds=scoring_spec.seed_bonus_rounds,
        )
        opponent_max_scores = np.max(opponent_scores, axis=0)
        opponent_tie_counts = np.sum(
            opponent_scores == opponent_max_scores[None, :],
            axis=0,
            dtype=np.int16,
        )

    return BracketFieldEvaluationContext(
        runtime=runtime,
        pool_settings=pool_settings,
        scoring_spec=scoring_spec,
        actual_team_wins=simulation.team_wins,
        opponent_max_scores=opponent_max_scores,
        opponent_tie_counts=opponent_tie_counts,
    )


def shared_evaluation_seed(*, runtime: BracketLabRuntime) -> int:
    return seed_from_cache_key(
        build_cache_key(
            artifact_kind="bracket-lab-evaluation",
            dataset_hash=runtime.dataset_hash,
            settings={"version": 1},
        )
    )


def build_shared_field_evaluation_context(
    *,
    runtime: BracketLabRuntime,
    pool_settings: PoolSettings,
    n_sims: int,
) -> BracketFieldEvaluationContext:
    seed = shared_evaluation_seed(runtime=runtime)
    return build_field_evaluation_context(
        runtime=runtime,
        pool_settings=pool_settings,
        opponent_seed=derive_child_seed(seed=seed, child_index=1),
        simulation_seed=derive_child_seed(seed=seed, child_index=2),
        n_sims=n_sims,
    )


def build_predicted_wins_for_brackets(
    *,
    brackets: list[EditableBracket],
    team_index: dict[str, int],
) -> npt.NDArray[np.int16]:
    predicted_wins = np.zeros((len(brackets), len(team_index)), dtype=np.int16)
    for bracket_index, bracket in enumerate(brackets):
        for pick in bracket.picks:
            winner_team_id = pick.winner_team_id
            if winner_team_id is None:
                msg = f"Bracket is incomplete; winner_team_id is required for game {pick.game_id}"
                raise ValueError(msg)
            predicted_wins[bracket_index, team_index[winner_team_id]] += 1
    return predicted_wins


def build_bracket_diagnostics(
    *,
    context: BracketFieldEvaluationContext,
    bracket: EditableBracket,
) -> tuple[float, list[PickDiagnostic]]:
    win_probability, user_win_shares = context.evaluate_bracket(bracket)
    diagnostics = PickDiagnosticAccumulator.build(
        bracket=bracket,
        graph=context.runtime.graph,
        team_index=context.runtime.team_index,
        total_sims=context.n_sims,
    )
    diagnostics.accumulate_batch(
        team_wins=context.actual_team_wins,
        user_win_shares=user_win_shares,
    )
    return win_probability, diagnostics.build_pick_diagnostics()


def build_rating_records_by_team_id(
    *,
    prepared: BracketLabPreparedInput,
    graph: BracketGraph,
) -> dict[str, RatingRecord]:
    rating_records_by_team_id = {row.team_id: row for row in prepared.ratings}
    play_in_slot_by_placeholder = {
        slot.placeholder_team_id: slot for slot in prepared.play_in_slots
    }

    resolved: dict[str, RatingRecord] = {}
    for team_id in graph.teams_by_id:
        if team_id in rating_records_by_team_id:
            resolved[team_id] = rating_records_by_team_id[team_id]
            continue

        slot = play_in_slot_by_placeholder.get(team_id)
        if slot is None:
            msg = f"Missing rating for team id: {team_id}"
            raise ValueError(msg)
        resolved[team_id] = RatingRecord(
            team_id=team_id,
            rating=weighted_slot_rating(slot),
            tempo=weighted_slot_tempo(slot),
        )

    return resolved


def build_completion_rank_by_team_id(
    *,
    prepared: BracketLabPreparedInput,
    graph: BracketGraph,
    rating_records_by_team_id: dict[str, RatingRecord],
) -> dict[str, int]:
    ranks = {
        row.team_id: row.rank for row in prepared.completion_inputs.kenpom_rankings
    }
    if len(ranks) == len(graph.teams_by_id):
        return ranks

    next_rank = max(ranks.values(), default=0) + 1
    missing_team_ids = sorted(
        (team_id for team_id in graph.teams_by_id if team_id not in ranks),
        key=lambda team_id: (
            -rating_records_by_team_id[team_id].rating,
            graph.teams_by_id[team_id].seed,
            team_id,
        ),
    )
    for team_id in missing_team_ids:
        ranks[team_id] = next_rank
        next_rank += 1
    return ranks


def build_public_pick_weights(
    *,
    prepared: BracketLabPreparedInput,
) -> dict[str, dict[str, float]]:
    weights_by_game: dict[str, dict[str, float]] = {}
    for row in prepared.public_picks:
        weights_by_game.setdefault(row.game_id, {})[row.team_id] = row.pick_percentage
    return weights_by_game


def sample_public_opponents(
    *,
    runtime: BracketLabRuntime,
    opponent_count: int,
    seed: int,
) -> list[PoolEntry]:
    if opponent_count <= 0:
        return []

    rng = np.random.default_rng(seed)
    return [
        PoolEntry(
            entry_id=f"public-{index + 1:04d}",
            entry_name=f"Public Opponent {index + 1}",
            picks=sample_public_picks(runtime=runtime, rng=rng),
        )
        for index in range(opponent_count)
    ]


def sample_public_picks(
    *,
    runtime: BracketLabRuntime,
    rng: np.random.Generator,
) -> list[EntryPick]:
    picks_by_game_id: dict[str, str] = {}

    for game_id in runtime.graph.topological_game_ids:
        game = runtime.graph.games_by_id[game_id]
        if game.round == 1:
            assert game.left_team_id is not None
            assert game.right_team_id is not None
            available_team_ids = [game.left_team_id, game.right_team_id]
        else:
            left_game_id, right_game_id = runtime.graph.children_by_game_id[game_id]
            available_team_ids = [picks_by_game_id[left_game_id], picks_by_game_id[right_game_id]]

        winner_team_id = sample_game_winner(
            available_team_ids=available_team_ids,
            pick_weights=runtime.public_pick_weights_by_game.get(game_id, {}),
            rng=rng,
        )
        picks_by_game_id[game_id] = winner_team_id

    return [
        EntryPick(game_id=game_id, winner_team_id=picks_by_game_id[game_id])
        for game_id in sorted(picks_by_game_id)
    ]


def sample_game_winner(
    *,
    available_team_ids: list[str],
    pick_weights: dict[str, float],
    rng: np.random.Generator,
) -> str:
    probabilities = np.asarray(
        [max(pick_weights.get(team_id, 0.0), 0.0) for team_id in available_team_ids],
        dtype=np.float64,
    )
    total = float(np.sum(probabilities))
    if total <= 0.0:
        probabilities = np.full(len(available_team_ids), 1.0 / len(available_team_ids))
    else:
        probabilities = probabilities / total

    return str(rng.choice(available_team_ids, p=probabilities))


def weighted_slot_rating(slot: PlayInSlot) -> float:
    return float(
        sum(
            candidate.advancement_probability * candidate.rating
            for candidate in slot.candidates
        )
    )


def weighted_slot_tempo(slot: PlayInSlot) -> float:
    return float(
        sum(
            candidate.advancement_probability * candidate.tempo
            for candidate in slot.candidates
        )
    )


def seed_from_cache_key(cache_key: str) -> int:
    return int(cache_key.rsplit("-", maxsplit=1)[-1], 16)


def derive_batch_seed(*, seed: int, batch_index: int, total_batches: int) -> int:
    if total_batches == 1 and batch_index == 0:
        return seed

    sequence = np.random.SeedSequence([seed, batch_index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def derive_child_seed(*, seed: int, child_index: int) -> int:
    sequence = np.random.SeedSequence([seed, child_index, 0xB4A8])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def user_win_shares_against_field(
    *,
    user_scores: npt.NDArray[np.int32],
    opponent_max_scores: npt.NDArray[np.int32] | None,
    opponent_tie_counts: npt.NDArray[np.int16] | None,
) -> npt.NDArray[np.float64]:
    if user_scores.ndim != 2:
        msg = "user_scores must be a 2D array"
        raise ValueError(msg)

    if opponent_max_scores is None or opponent_tie_counts is None:
        return np.ones(user_scores.shape, dtype=np.float64)

    better_mask = user_scores > opponent_max_scores[None, :]
    tie_mask = user_scores == opponent_max_scores[None, :]
    tie_shares = 1.0 / (opponent_tie_counts.astype(np.float64) + 1.0)
    return np.where(
        better_mask,
        1.0,
        np.where(tie_mask, tie_shares[None, :], 0.0),
    )


def user_win_shares_by_sim(scores: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
    max_scores = np.max(scores, axis=0)
    winners = scores == max_scores
    tie_counts = np.sum(winners, axis=0)
    return np.where(winners[0], 1.0 / tie_counts, 0.0).astype(np.float64, copy=False)


def pick_extreme_index(
    *,
    diagnostics: list[PickDiagnostic],
    values: list[float],
    reverse: bool,
) -> int:
    comparator = max if reverse else min
    target = comparator(values)
    matching_indices = [index for index, value in enumerate(values) if value == target]
    ordered = sorted(
        matching_indices,
        key=lambda index: (
            -diagnostics[index].round,
            diagnostics[index].game_id,
        ),
    )
    return ordered[0]


def bracket_to_entry(*, bracket: EditableBracket, graph: BracketGraph) -> PoolEntry:
    return editable_bracket_to_entry(bracket=bracket, graph=graph)
