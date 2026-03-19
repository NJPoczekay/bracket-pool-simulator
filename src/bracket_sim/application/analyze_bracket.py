"""Bracket Lab analyzer service and prepared-dataset runtime helpers."""

from __future__ import annotations

import math
from pathlib import Path

from bracket_sim.application.bracket_completion import (
    build_initial_bracket,
    canonicalize_bracket,
    normalize_completion_mode,
)
from bracket_sim.application.bracket_completion import (
    complete_bracket as complete_editable_bracket,
)
from bracket_sim.application.bracket_lab_eval import (
    ANALYSIS_BATCH_SIZE,
    ANALYSIS_N_SIMS,
    BRACKET_LAB_POINT_SPREAD_STD_DEV,
    DEFAULT_POOL_SETTINGS,
    PickDiagnosticAccumulator,
    bracket_to_entry,
    build_bracket_lab_runtime,
    derive_batch_seed,
    resolve_scoring_spec,
    sample_public_opponents,
    seed_from_cache_key,
    user_win_shares_by_sim,
    weighted_slot_rating,
    weighted_slot_tempo,
)
from bracket_sim.application.optimize_bracket import optimize_bracket as optimize_editable_bracket
from bracket_sim.domain.product_models import (
    AnalyzeBracketRequest,
    BracketAnalysis,
    BracketCompletionResult,
    BracketLabBootstrap,
    CompleteBracketRequest,
    CompletionMode,
    OptimizationResult,
    OptimizeBracketRequest,
)
from bracket_sim.domain.scoring import build_predicted_wins_matrix, score_entries, validate_entries
from bracket_sim.domain.simulator import simulate_tournament
from bracket_sim.infrastructure.storage.cache_keys import build_cache_key


class BracketLabService:
    """High-level controller for Bracket Lab bootstrap and analysis workflows."""

    def __init__(self, input_dir: Path) -> None:
        self._runtime = build_bracket_lab_runtime(input_dir)

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
            default_pool_settings=DEFAULT_POOL_SETTINGS,
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

    def optimize_bracket(self, request: OptimizeBracketRequest) -> OptimizationResult:
        """Optimize one complete bracket without mutating locked picks."""

        return optimize_editable_bracket(
            request=request,
            runtime=self._runtime,
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
        user_entry = bracket_to_entry(
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
        seed = seed_from_cache_key(cache_key)

        opponent_entries = sample_public_opponents(
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

        scoring_spec = resolve_scoring_spec(request.pool_settings.scoring_system)
        diagnostics = PickDiagnosticAccumulator.build(
            bracket=canonical_bracket,
            graph=self._runtime.graph,
            team_index=self._runtime.team_index,
            total_sims=ANALYSIS_N_SIMS,
        )

        total_batches = math.ceil(ANALYSIS_N_SIMS / ANALYSIS_BATCH_SIZE)
        for batch_index in range(total_batches):
            batch_n_sims = min(
                ANALYSIS_BATCH_SIZE,
                ANALYSIS_N_SIMS - (batch_index * ANALYSIS_BATCH_SIZE),
            )
            batch_seed = derive_batch_seed(
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
                point_spread_std_dev=BRACKET_LAB_POINT_SPREAD_STD_DEV,
            )
            scores = score_entries(
                predicted_wins=predicted_wins,
                actual_wins=simulation.team_wins,
                round_values=scoring_spec.round_values,
                team_seeds=self._runtime.team_seeds,
                seed_bonus_rounds=scoring_spec.seed_bonus_rounds,
            )
            diagnostics.accumulate_batch(
                team_wins=simulation.team_wins,
                user_win_shares=user_win_shares_by_sim(scores),
            )

        return BracketAnalysis(
            bracket=canonical_bracket,
            pool_settings=request.pool_settings,
            completion_mode=request.completion_mode,
            dataset_hash=self._runtime.dataset_hash,
            cache_key=cache_key,
            win_probability=diagnostics.win_probability,
            public_percentile=None,
            pick_diagnostics=diagnostics.build_pick_diagnostics(),
        )


_editable_bracket_to_entry = bracket_to_entry
_sample_public_opponents = sample_public_opponents
_weighted_slot_rating = weighted_slot_rating
_weighted_slot_tempo = weighted_slot_tempo
