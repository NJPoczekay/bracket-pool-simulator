"""Bracket Lab analyzer service and prepared-dataset runtime helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from bracket_sim.domain.bracket_graph import BracketGraph, build_bracket_graph
from bracket_sim.domain.bracket_lab_models import PlayInSlot
from bracket_sim.domain.constraints import validate_constraints
from bracket_sim.domain.models import EntryPick, PoolEntry, RatingRecord
from bracket_sim.domain.product_models import (
    AnalyzeBracketRequest,
    BracketAnalysis,
    BracketEditPick,
    BracketLabBootstrap,
    CompletionMode,
    EditableBracket,
    PickDiagnostic,
    PickDiagnosticTag,
    PoolSettings,
    ScoringSystemKey,
)
from bracket_sim.domain.scoring import (
    build_predicted_wins_matrix,
    score_entries,
    validate_entries,
)
from bracket_sim.domain.simulator import canonical_team_order, simulate_tournament
from bracket_sim.infrastructure.storage.bracket_lab_prepared_loader import (
    BracketLabPreparedInput,
    load_bracket_lab_prepared_input,
)
from bracket_sim.infrastructure.storage.cache_keys import build_cache_key, capture_dataset_hash

_ANALYSIS_N_SIMS = 100_000
_ANALYSIS_BATCH_SIZE = 1_000
_ANALYSIS_POINT_SPREAD_STD_DEV = 11.0
_DEFAULT_POOL_SETTINGS = PoolSettings(
    pool_size=10,
    scoring_system=ScoringSystemKey.ESPN,
)


@dataclass(frozen=True)
class _ScoringSpec:
    round_values: tuple[int, int, int, int, int, int]
    seed_bonus: bool = False


@dataclass(frozen=True)
class _BracketLabRuntime:
    input_dir: Path
    prepared: BracketLabPreparedInput
    graph: BracketGraph
    constraints_by_game_id: dict[str, str]
    dataset_hash: str
    rating_records_by_team_id: dict[str, RatingRecord]
    team_ids: list[str]
    team_index: dict[str, int]
    team_seeds: npt.NDArray[np.int16]
    public_pick_weights_by_game: dict[str, dict[str, float]]


class BracketLabService:
    """High-level controller for Bracket Lab bootstrap and analysis workflows."""

    def __init__(self, input_dir: Path) -> None:
        prepared = load_bracket_lab_prepared_input(input_dir)
        graph = build_bracket_graph(teams=prepared.teams, games=prepared.games)
        constraints_by_game_id = validate_constraints(
            constraints=prepared.constraints,
            graph=graph,
        )
        team_ids = canonical_team_order(graph)
        team_index = {team_id: idx for idx, team_id in enumerate(team_ids)}
        self._runtime = _BracketLabRuntime(
            input_dir=input_dir,
            prepared=prepared,
            graph=graph,
            constraints_by_game_id=constraints_by_game_id,
            dataset_hash=capture_dataset_hash(input_dir),
            rating_records_by_team_id=_build_rating_records_by_team_id(
                prepared=prepared,
                graph=graph,
            ),
            team_ids=team_ids,
            team_index=team_index,
            team_seeds=np.asarray(
                [graph.teams_by_id[team_id].seed for team_id in team_ids],
                dtype=np.int16,
            ),
            public_pick_weights_by_game=_build_public_pick_weights(prepared=prepared),
        )

    @property
    def dataset_hash(self) -> str:
        """Return the prepared dataset hash exposed to API clients."""

        return self._runtime.dataset_hash

    def build_bootstrap(self) -> BracketLabBootstrap:
        """Return the typed bootstrap payload for the browser editor."""

        return BracketLabBootstrap(
            dataset_hash=self._runtime.dataset_hash,
            completion_mode=CompletionMode.MANUAL,
            default_pool_settings=_DEFAULT_POOL_SETTINGS,
            teams=sorted(self._runtime.prepared.teams, key=lambda team: team.team_id),
            games=sorted(
                self._runtime.prepared.games,
                key=lambda game: (game.round, game.game_id),
            ),
        )

    def analyze_bracket(self, request: AnalyzeBracketRequest) -> BracketAnalysis:
        """Run a deterministic full-bracket analysis against sampled public opponents."""

        if request.completion_mode != CompletionMode.MANUAL:
            msg = "Phase 2 analysis only supports completion_mode='manual'"
            raise ValueError(msg)

        user_entry = _editable_bracket_to_entry(
            bracket=request.bracket,
            graph=self._runtime.graph,
        )
        cache_key = build_cache_key(
            artifact_kind="analysis",
            dataset_hash=self._runtime.dataset_hash,
            settings={
                "bracket": request.bracket,
                "pool_settings": request.pool_settings,
                "completion_mode": request.completion_mode,
            },
        )
        seed = _seed_from_cache_key(cache_key)

        opponent_entries = _sample_public_opponents(
            runtime=self._runtime,
            opponent_count=max(request.pool_settings.pool_size - 1, 0),
            seed=seed,
        )
        entries = [user_entry, *opponent_entries]
        validate_entries(entries=entries, graph=self._runtime.graph)
        _, team_ids, predicted_wins = build_predicted_wins_matrix(
            entries=entries,
            graph=self._runtime.graph,
        )
        if team_ids != self._runtime.team_ids:
            msg = "Bracket analysis team ordering drifted from prepared dataset"
            raise RuntimeError(msg)

        scoring_spec = _resolve_scoring_spec(request.pool_settings.scoring_system)
        diagnostics = _DiagnosticAccumulator.build(
            bracket=request.bracket,
            graph=self._runtime.graph,
            team_index=self._runtime.team_index,
        )

        total_batches = math.ceil(_ANALYSIS_N_SIMS / _ANALYSIS_BATCH_SIZE)
        for batch_index in range(total_batches):
            batch_n_sims = min(
                _ANALYSIS_BATCH_SIZE,
                _ANALYSIS_N_SIMS - (batch_index * _ANALYSIS_BATCH_SIZE),
            )
            batch_seed = _derive_batch_seed(
                seed=seed,
                batch_index=batch_index,
                total_batches=total_batches,
            )
            simulation = simulate_tournament(
                graph=self._runtime.graph,
                rating_records_by_team_id=self._runtime.rating_records_by_team_id,
                constraints_by_game_id=self._runtime.constraints_by_game_id,
                n_sims=batch_n_sims,
                seed=batch_seed,
                point_spread_std_dev=_ANALYSIS_POINT_SPREAD_STD_DEV,
            )
            scores = score_entries(
                predicted_wins=predicted_wins,
                actual_wins=simulation.team_wins,
                round_values=scoring_spec.round_values,
                team_seeds=self._runtime.team_seeds,
                seed_bonus=scoring_spec.seed_bonus,
            )
            diagnostics.accumulate_batch(team_wins=simulation.team_wins, scores=scores)

        return BracketAnalysis(
            bracket=request.bracket,
            pool_settings=request.pool_settings,
            completion_mode=request.completion_mode,
            dataset_hash=self._runtime.dataset_hash,
            cache_key=cache_key,
            win_probability=diagnostics.win_probability,
            public_percentile=None,
            pick_diagnostics=diagnostics.build_pick_diagnostics(),
        )


@dataclass
class _DiagnosticAccumulator:
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
    total_sims: int = _ANALYSIS_N_SIMS

    @classmethod
    def build(
        cls,
        *,
        bracket: EditableBracket,
        graph: BracketGraph,
        team_index: dict[str, int],
    ) -> _DiagnosticAccumulator:
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
        )

    @property
    def win_probability(self) -> float:
        return float(self.total_win_share / self.total_sims)

    def accumulate_batch(
        self,
        *,
        team_wins: npt.NDArray[np.int16],
        scores: npt.NDArray[np.int32],
    ) -> None:
        user_win_shares = _user_win_shares_by_sim(scores)
        batch_total_share = float(np.sum(user_win_shares))
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
            best_index = _pick_extreme_index(
                diagnostics=diagnostics,
                values=deltas,
                reverse=True,
            )
            worst_index = _pick_extreme_index(
                diagnostics=diagnostics,
                values=deltas,
                reverse=False,
            )
            important_index = _pick_extreme_index(
                diagnostics=diagnostics,
                values=importance_swings,
                reverse=True,
            )
            diagnostics[best_index].tags.append(PickDiagnosticTag.BEST_PICK)
            diagnostics[worst_index].tags.append(PickDiagnosticTag.WORST_PICK)
            diagnostics[important_index].tags.append(PickDiagnosticTag.MOST_IMPORTANT)

        return diagnostics


def _pick_extreme_index(
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


def _editable_bracket_to_entry(*, bracket: EditableBracket, graph: BracketGraph) -> PoolEntry:
    pick_map = {pick.game_id: pick.winner_team_id for pick in bracket.picks}
    expected_game_ids = set(graph.games_by_id)
    if set(pick_map) != expected_game_ids:
        missing = sorted(expected_game_ids - set(pick_map))
        extra = sorted(set(pick_map) - expected_game_ids)
        msg = (
            "Bracket must include exactly one pick for every game. "
            f"Missing={missing[:5]} Extra={extra[:5]}"
        )
        raise ValueError(msg)

    missing_winners = sorted(game_id for game_id, team_id in pick_map.items() if team_id is None)
    if missing_winners:
        msg = f"Bracket is incomplete; winner_team_id is required for {missing_winners[:5]}"
        raise ValueError(msg)

    entry = PoolEntry(
        entry_id="user-bracket",
        entry_name="Your Bracket",
        picks=[
            EntryPick(game_id=game_id, winner_team_id=str(pick_map[game_id]))
            for game_id in sorted(pick_map)
        ],
    )
    validate_entries(entries=[entry], graph=graph)
    return entry


def _build_rating_records_by_team_id(
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
            rating=_weighted_slot_rating(slot),
            tempo=_weighted_slot_tempo(slot),
        )

    return resolved


def _weighted_slot_rating(slot: PlayInSlot) -> float:
    return float(
        sum(
            candidate.advancement_probability * candidate.rating
            for candidate in slot.candidates
        )
    )


def _weighted_slot_tempo(slot: PlayInSlot) -> float:
    return float(
        sum(
            candidate.advancement_probability * candidate.tempo
            for candidate in slot.candidates
        )
    )


def _build_public_pick_weights(
    *,
    prepared: BracketLabPreparedInput,
) -> dict[str, dict[str, float]]:
    weights_by_game: dict[str, dict[str, float]] = {}
    for row in prepared.public_picks:
        weights_by_game.setdefault(row.game_id, {})[row.team_id] = row.pick_percentage
    return weights_by_game


def _sample_public_opponents(
    *,
    runtime: _BracketLabRuntime,
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
            picks=_sample_public_picks(runtime=runtime, rng=rng),
        )
        for index in range(opponent_count)
    ]


def _sample_public_picks(
    *,
    runtime: _BracketLabRuntime,
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

        winner_team_id = _sample_game_winner(
            available_team_ids=available_team_ids,
            pick_weights=runtime.public_pick_weights_by_game.get(game_id, {}),
            rng=rng,
        )
        picks_by_game_id[game_id] = winner_team_id

    return [
        EntryPick(game_id=game_id, winner_team_id=picks_by_game_id[game_id])
        for game_id in sorted(picks_by_game_id)
    ]


def _sample_game_winner(
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


def _resolve_scoring_spec(scoring_system: ScoringSystemKey) -> _ScoringSpec:
    if scoring_system == ScoringSystemKey.ESPN:
        return _ScoringSpec(round_values=(1, 2, 4, 8, 16, 32))
    if scoring_system == ScoringSystemKey.LINEAR:
        return _ScoringSpec(round_values=(1, 2, 3, 4, 5, 6))
    if scoring_system == ScoringSystemKey.FIBONACCI:
        return _ScoringSpec(round_values=(2, 3, 5, 8, 13, 21))
    return _ScoringSpec(round_values=(1, 2, 3, 4, 5, 6), seed_bonus=True)


def _seed_from_cache_key(cache_key: str) -> int:
    return int(cache_key.rsplit("-", maxsplit=1)[-1], 16)


def _derive_batch_seed(*, seed: int, batch_index: int, total_batches: int) -> int:
    if total_batches == 1 and batch_index == 0:
        return seed

    sequence = np.random.SeedSequence([seed, batch_index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _user_win_shares_by_sim(scores: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
    max_scores = np.max(scores, axis=0)
    winners = scores == max_scores
    tie_counts = np.sum(winners, axis=0)
    return np.where(winners[0], 1.0 / tie_counts, 0.0).astype(np.float64, copy=False)
