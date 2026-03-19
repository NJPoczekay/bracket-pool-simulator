"""Bracket Lab analyzer service and prepared-dataset runtime helpers."""

from __future__ import annotations

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
    ANALYSIS_N_SIMS,
    DEFAULT_POOL_SETTINGS,
    build_bracket_diagnostics,
    bracket_to_entry,
    build_bracket_lab_runtime,
    build_shared_field_evaluation_context,
    sample_public_opponents,
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
        cache_key = build_cache_key(
            artifact_kind="analysis",
            dataset_hash=self._runtime.dataset_hash,
            settings={
                "bracket": canonical_bracket,
                "pool_settings": request.pool_settings,
                "completion_mode": normalized_completion_mode,
            },
        )
        context = build_shared_field_evaluation_context(
            runtime=self._runtime,
            pool_settings=request.pool_settings,
            n_sims=ANALYSIS_N_SIMS,
        )
        win_probability, pick_diagnostics = build_bracket_diagnostics(
            context=context,
            bracket=canonical_bracket,
        )

        return BracketAnalysis(
            bracket=canonical_bracket,
            pool_settings=request.pool_settings,
            completion_mode=request.completion_mode,
            dataset_hash=self._runtime.dataset_hash,
            cache_key=cache_key,
            win_probability=win_probability,
            public_percentile=None,
            pick_diagnostics=pick_diagnostics,
        )


_editable_bracket_to_entry = bracket_to_entry
_sample_public_opponents = sample_public_opponents
_weighted_slot_rating = weighted_slot_rating
_weighted_slot_tempo = weighted_slot_tempo
