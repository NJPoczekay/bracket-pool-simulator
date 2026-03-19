"""Bracket Lab analyzer service and prepared-dataset runtime helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from bracket_sim.application.bracket_completion import (
    _team_id_for_region_seed,
    build_initial_bracket,
    canonicalize_bracket,
    derive_region_champion_game_ids,
    editable_bracket_to_entry,
    normalize_completion_mode,
)
from bracket_sim.application.bracket_completion import (
    complete_bracket as complete_editable_bracket,
)
from bracket_sim.domain.bracket_graph import BracketGraph, build_bracket_graph
from bracket_sim.domain.bracket_lab_models import PlayInSlot
from bracket_sim.domain.constraints import validate_constraints
from bracket_sim.domain.models import EntryPick, PoolEntry, RatingRecord
from bracket_sim.domain.product_models import (
    AnalyzeBracketRequest,
    BracketAnalysis,
    BracketCompletionResult,
    BracketEditPick,
    BracketLabBootstrap,
    CompleteBracketRequest,
    CompletionMode,
    EditableBracket,
    OptimizationAlternative,
    OptimizationResult,
    OptimizeBracketRequest,
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

_BRACKET_LAB_EVALUATION_N_SIMS = 100_000
_BRACKET_LAB_EVALUATION_BATCH_SIZE = 1_000
_BRACKET_LAB_POINT_SPREAD_STD_DEV = 11.0
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
    team_rank_by_team_id: dict[str, int]
    team_ids: list[str]
    team_index: dict[str, int]
    team_seeds: npt.NDArray[np.int16]
    public_pick_weights_by_game: dict[str, dict[str, float]]
    region_champion_game_ids: dict[str, str]


@dataclass(frozen=True)
class _BracketEvaluationResult:
    win_probabilities: dict[str, float]
    diagnostics: _DiagnosticAccumulator | None = None


@dataclass(frozen=True)
class _OptimizationCandidate:
    label: str
    bracket: EditableBracket
    summary: str | None = None


class BracketLabService:
    """High-level controller for Bracket Lab bootstrap and analysis workflows."""

    def __init__(self, input_dir: Path) -> None:
        prepared = load_bracket_lab_prepared_input(input_dir)
        graph = build_bracket_graph(teams=prepared.teams, games=prepared.games)
        constraints_by_game_id = validate_constraints(
            constraints=prepared.constraints,
            graph=graph,
        )
        rating_records_by_team_id = _build_rating_records_by_team_id(
            prepared=prepared,
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
            rating_records_by_team_id=rating_records_by_team_id,
            team_rank_by_team_id=_build_completion_rank_by_team_id(
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
            public_pick_weights_by_game=_build_public_pick_weights(prepared=prepared),
            region_champion_game_ids=derive_region_champion_game_ids(graph),
        )

    @property
    def dataset_hash(self) -> str:
        """Return the prepared dataset hash exposed to API clients."""

        return self._runtime.dataset_hash

    @property
    def input_dir(self) -> Path:
        """Return the prepared input directory backing this service."""

        return self._runtime.input_dir

    def build_bootstrap(self) -> BracketLabBootstrap:
        """Return the typed bootstrap payload for the browser editor."""

        return BracketLabBootstrap(
            dataset_hash=self._runtime.dataset_hash,
            completion_mode=CompletionMode.MANUAL,
            default_pool_settings=_DEFAULT_POOL_SETTINGS,
            initial_bracket=build_initial_bracket(
                graph=self._runtime.graph,
                constraints_by_game_id=self._runtime.constraints_by_game_id,
            ),
            completion_inputs=self._runtime.prepared.completion_inputs,
            play_in_slots=self._runtime.prepared.play_in_slots,
            teams=sorted(self._runtime.prepared.teams, key=lambda team: team.team_id),
            games=sorted(
                self._runtime.prepared.games,
                key=lambda game: (game.round, game.game_id),
            ),
        )

    def complete_bracket(self, request: CompleteBracketRequest) -> BracketCompletionResult:
        """Auto-complete a partial bracket while preserving locked picks."""

        return complete_editable_bracket(
            request=request,
            dataset_hash=self._runtime.dataset_hash,
            graph=self._runtime.graph,
            constraints_by_game_id=self._runtime.constraints_by_game_id,
            public_pick_weights_by_game=self._runtime.public_pick_weights_by_game,
            rating_records_by_team_id=self._runtime.rating_records_by_team_id,
            team_rank_by_team_id=self._runtime.team_rank_by_team_id,
            region_champion_game_ids=self._runtime.region_champion_game_ids,
        )

    def analyze_bracket(self, request: AnalyzeBracketRequest) -> BracketAnalysis:
        """Run a deterministic full-bracket analysis against sampled public opponents."""

        if request.completion_mode == CompletionMode.PICK_FOUR:
            msg = "completion_mode='pick_four' is only valid as a completion helper"
            raise ValueError(msg)

        canonical_bracket = canonicalize_bracket(
            bracket=request.bracket,
            graph=self._runtime.graph,
            constraints_by_game_id=self._runtime.constraints_by_game_id,
        )
        normalized_completion_mode = normalize_completion_mode(request.completion_mode)
        user_entry = _editable_bracket_to_entry(
            bracket=canonical_bracket,
            graph=self._runtime.graph,
        )
        cache_key = build_cache_key(
            artifact_kind="analysis",
            dataset_hash=self._runtime.dataset_hash,
            settings={
                "bracket": canonical_bracket,
                "pool_settings": request.pool_settings,
                "completion_mode": normalized_completion_mode,
            },
        )
        validate_entries(entries=[user_entry], graph=self._runtime.graph)
        evaluation = _evaluate_brackets(
            runtime=self._runtime,
            candidate_brackets={"requested": canonical_bracket},
            pool_settings=request.pool_settings,
            diagnostics_target_key="requested",
        )
        diagnostics = evaluation.diagnostics
        if diagnostics is None:
            msg = "Bracket analysis diagnostics were not collected"
            raise RuntimeError(msg)

        return BracketAnalysis(
            bracket=canonical_bracket,
            pool_settings=request.pool_settings,
            completion_mode=request.completion_mode,
            dataset_hash=self._runtime.dataset_hash,
            cache_key=cache_key,
            win_probability=evaluation.win_probabilities["requested"],
            public_percentile=None,
            pick_diagnostics=diagnostics.build_pick_diagnostics(),
        )

    def optimize_bracket(self, request: OptimizeBracketRequest) -> OptimizationResult:
        """Return deterministic bracket recommendations scored on the shared evaluation base."""

        if request.completion_mode == CompletionMode.PICK_FOUR:
            msg = "completion_mode='pick_four' is only valid as a completion helper"
            raise ValueError(msg)

        canonical_bracket = canonicalize_bracket(
            bracket=request.bracket,
            graph=self._runtime.graph,
            constraints_by_game_id=self._runtime.constraints_by_game_id,
        )
        validate_entries(
            entries=[_editable_bracket_to_entry(bracket=canonical_bracket, graph=self._runtime.graph)],
            graph=self._runtime.graph,
        )
        normalized_completion_mode = normalize_completion_mode(request.completion_mode)
        cache_key = build_cache_key(
            artifact_kind="optimization",
            dataset_hash=self._runtime.dataset_hash,
            settings={
                "bracket": canonical_bracket,
                "pool_settings": request.pool_settings,
                "completion_mode": normalized_completion_mode,
            },
        )

        candidates = self._build_optimization_candidates(current_bracket=canonical_bracket)
        evaluation = _evaluate_brackets(
            runtime=self._runtime,
            candidate_brackets={
                candidate_id: candidate.bracket for candidate_id, candidate in candidates.items()
            },
            pool_settings=request.pool_settings,
        )
        ordered_candidate_ids = sorted(
            candidates,
            key=lambda candidate_id: (
                -evaluation.win_probabilities[candidate_id],
                _changed_pick_count(
                    baseline=canonical_bracket,
                    candidate=candidates[candidate_id].bracket,
                ),
                candidate_id,
            ),
        )
        recommended_id = ordered_candidate_ids[0]
        recommended_candidate = candidates[recommended_id]
        alternatives: list[OptimizationAlternative] = []
        for candidate_id in ordered_candidate_ids[1:]:
            candidate = candidates[candidate_id]
            alternatives.append(
                OptimizationAlternative(
                    label=candidate.label,
                    bracket=candidate.bracket,
                    projected_win_probability=evaluation.win_probabilities[candidate_id],
                    changed_pick_count=_changed_pick_count(
                        baseline=canonical_bracket,
                        candidate=candidate.bracket,
                    ),
                    summary=candidate.summary,
                )
            )
            if len(alternatives) == 3:
                break

        return OptimizationResult(
            pool_settings=request.pool_settings,
            completion_mode=request.completion_mode,
            dataset_hash=self._runtime.dataset_hash,
            cache_key=cache_key,
            recommended_bracket=recommended_candidate.bracket,
            projected_win_probability=evaluation.win_probabilities[recommended_id],
            changed_pick_count=_changed_pick_count(
                baseline=canonical_bracket,
                candidate=recommended_candidate.bracket,
            ),
            alternatives=alternatives,
        )

    def _build_optimization_candidates(
        self,
        *,
        current_bracket: EditableBracket,
    ) -> dict[str, _OptimizationCandidate]:
        candidates: dict[str, _OptimizationCandidate] = {}
        seen_brackets: set[tuple[tuple[str, str | None], ...]] = set()

        def add(candidate_id: str, candidate: _OptimizationCandidate) -> None:
            bracket_key = tuple(
                (pick.game_id, pick.winner_team_id)
                for pick in candidate.bracket.picks
            )
            if bracket_key in seen_brackets:
                return
            seen_brackets.add(bracket_key)
            candidates[candidate_id] = candidate

        add(
            "current",
            _OptimizationCandidate(
                label="Current Bracket",
                bracket=current_bracket,
                summary="Your current bracket on the shared 100k evaluation base.",
            ),
        )

        for mode, label in (
            (CompletionMode.TOURNAMENT_SEEDS, "Tournament Seeds"),
            (CompletionMode.POPULAR_PICKS, "Popular Picks"),
            (CompletionMode.KENPOM, "KenPom"),
        ):
            completed = self.complete_bracket(
                CompleteBracketRequest(
                    bracket=EditableBracket(picks=[]),
                    completion_mode=mode,
                )
            )
            add(
                mode.value,
                _OptimizationCandidate(
                    label=label,
                    bracket=completed.completed_bracket,
                    summary=f"{label} baseline evaluated with the same opponent field and tournament draws.",
                ),
            )

        championship_game_id = self._runtime.graph.championship_game_id
        for region in sorted(self._runtime.region_champion_game_ids):
            for seed in (1, 2):
                team_id = _team_id_for_region_seed(
                    graph=self._runtime.graph,
                    region=region,
                    seed=seed,
                )
                completed = self.complete_bracket(
                    CompleteBracketRequest(
                        bracket=EditableBracket(
                            picks=[
                                BracketEditPick(
                                    game_id=championship_game_id,
                                    winner_team_id=team_id,
                                    locked=True,
                                )
                            ]
                        ),
                        completion_mode=CompletionMode.POPULAR_PICKS,
                    )
                )
                add(
                    f"popular-{region}-{seed}",
                    _OptimizationCandidate(
                        label=f"Popular Picks: {region.title()} {seed}-Seed Champion",
                        bracket=completed.completed_bracket,
                        summary=(
                            f"Public-pick bracket with the {region.title()} {seed}-seed forced "
                            "to win the title."
                        ),
                    ),
                )

        return candidates


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
    total_sims: int = _BRACKET_LAB_EVALUATION_N_SIMS

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


def _build_completion_rank_by_team_id(
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


def _editable_bracket_to_entry(*, bracket: EditableBracket, graph: BracketGraph) -> PoolEntry:
    return editable_bracket_to_entry(bracket=bracket, graph=graph)


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


def _shared_evaluation_seed(*, runtime: _BracketLabRuntime) -> int:
    return _seed_from_cache_key(
        build_cache_key(
            artifact_kind="bracket-lab-evaluation",
            dataset_hash=runtime.dataset_hash,
            settings={"version": 1},
        )
    )


def _evaluate_brackets(
    *,
    runtime: _BracketLabRuntime,
    candidate_brackets: dict[str, EditableBracket],
    pool_settings: PoolSettings,
    diagnostics_target_key: str | None = None,
) -> _BracketEvaluationResult:
    if not candidate_brackets:
        msg = "At least one candidate bracket is required"
        raise ValueError(msg)

    scoring_spec = _resolve_scoring_spec(pool_settings.scoring_system)
    evaluation_seed = _shared_evaluation_seed(runtime=runtime)
    opponent_entries = _sample_public_opponents(
        runtime=runtime,
        opponent_count=max(pool_settings.pool_size - 1, 0),
        seed=evaluation_seed,
    )
    validate_entries(entries=opponent_entries, graph=runtime.graph)

    if opponent_entries:
        _, team_ids, opponent_predicted_wins = build_predicted_wins_matrix(
            entries=opponent_entries,
            graph=runtime.graph,
        )
        if team_ids != runtime.team_ids:
            msg = "Bracket evaluation opponent ordering drifted from prepared dataset"
            raise RuntimeError(msg)
    else:
        opponent_predicted_wins = np.zeros((0, len(runtime.team_ids)), dtype=np.int16)

    predicted_wins_by_key: dict[str, npt.NDArray[np.int16]] = {}
    for candidate_key, bracket in candidate_brackets.items():
        entry = _editable_bracket_to_entry(bracket=bracket, graph=runtime.graph)
        _, team_ids, user_predicted_wins = build_predicted_wins_matrix(
            entries=[entry],
            graph=runtime.graph,
        )
        if team_ids != runtime.team_ids:
            msg = "Bracket evaluation team ordering drifted from prepared dataset"
            raise RuntimeError(msg)
        predicted_wins_by_key[candidate_key] = np.vstack(
            [user_predicted_wins, opponent_predicted_wins]
        )

    diagnostics = (
        _DiagnosticAccumulator.build(
            bracket=candidate_brackets[diagnostics_target_key],
            graph=runtime.graph,
            team_index=runtime.team_index,
        )
        if diagnostics_target_key is not None
        else None
    )
    totals = {candidate_key: 0.0 for candidate_key in candidate_brackets}
    total_batches = math.ceil(
        _BRACKET_LAB_EVALUATION_N_SIMS / _BRACKET_LAB_EVALUATION_BATCH_SIZE
    )
    for batch_index in range(total_batches):
        batch_n_sims = min(
            _BRACKET_LAB_EVALUATION_BATCH_SIZE,
            _BRACKET_LAB_EVALUATION_N_SIMS
            - (batch_index * _BRACKET_LAB_EVALUATION_BATCH_SIZE),
        )
        batch_seed = _derive_batch_seed(
            seed=evaluation_seed,
            batch_index=batch_index,
            total_batches=total_batches,
        )
        simulation = simulate_tournament(
            graph=runtime.graph,
            rating_records_by_team_id=runtime.rating_records_by_team_id,
            constraints_by_game_id=runtime.constraints_by_game_id,
            n_sims=batch_n_sims,
            seed=batch_seed,
            point_spread_std_dev=_BRACKET_LAB_POINT_SPREAD_STD_DEV,
        )
        for candidate_key, predicted_wins in predicted_wins_by_key.items():
            scores = score_entries(
                predicted_wins=predicted_wins,
                actual_wins=simulation.team_wins,
                round_values=scoring_spec.round_values,
                team_seeds=runtime.team_seeds,
                seed_bonus=scoring_spec.seed_bonus,
            )
            user_win_shares = _user_win_shares_by_sim(scores)
            totals[candidate_key] += float(np.sum(user_win_shares))
            if diagnostics_target_key == candidate_key and diagnostics is not None:
                diagnostics.accumulate_batch(team_wins=simulation.team_wins, scores=scores)

    return _BracketEvaluationResult(
        win_probabilities={
            candidate_key: total / _BRACKET_LAB_EVALUATION_N_SIMS
            for candidate_key, total in totals.items()
        },
        diagnostics=diagnostics,
    )


def _changed_pick_count(*, baseline: EditableBracket, candidate: EditableBracket) -> int:
    baseline_by_game_id = {
        pick.game_id: pick.winner_team_id
        for pick in baseline.picks
    }
    return sum(
        baseline_by_game_id.get(pick.game_id) != pick.winner_team_id
        for pick in candidate.picks
    )


def _user_win_shares_by_sim(scores: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
    max_scores = np.max(scores, axis=0)
    winners = scores == max_scores
    tie_counts = np.sum(winners, axis=0)
    return np.where(winners[0], 1.0 / tie_counts, 0.0).astype(np.float64, copy=False)
