"""Shared product-facing models for the web/API surface."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bracket_sim.domain.models import Game, Team


class ScoringSystemKey(StrEnum):
    """Supported scoring system identifiers."""

    ESPN = "1-2-4-8-16-32"
    LINEAR = "1-2-3-4-5-6"
    FIBONACCI = "2-3-5-8-13-21"
    ROUND_PLUS_SEED = "round+seed"


class CompletionMode(StrEnum):
    """Bracket completion strategies exposed by the product layer."""

    MANUAL = "manual"
    TOURNAMENT_SEEDS = "tournament_seeds"
    POPULAR_PICKS = "popular_picks"
    INTERNAL_MODEL_RANK = "internal_model_rank"
    KENPOM = "kenpom"
    PICK_FOUR = "pick_four"


class PickDiagnosticTag(StrEnum):
    """Special tags applied to pick-level analysis diagnostics."""

    BEST_PICK = "best_pick"
    WORST_PICK = "worst_pick"
    MOST_IMPORTANT = "most_important"


class CacheArtifactKind(StrEnum):
    """Reusable cache artifacts planned for later product phases."""

    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"


class WorkflowState(StrEnum):
    """Lifecycle state for one top-level product workflow."""

    PLANNED = "planned"
    SETUP_REQUIRED = "setup_required"
    LIVE = "live"


class WorkflowKey(StrEnum):
    """Stable identifiers for the integrated app workflows."""

    BRACKET_LAB = "bracket_lab"
    POOL_TRACKER = "pool_tracker"


class ScoringSystem(BaseModel):
    """Definition for one supported pool scoring system."""

    model_config = ConfigDict(frozen=True)

    key: ScoringSystemKey
    label: str = Field(min_length=1)
    round_values: tuple[int, int, int, int, int, int]
    seed_bonus: bool = False
    implemented: bool = False
    description: str = Field(min_length=1)

    @field_validator("round_values")
    @classmethod
    def validate_round_values(
        cls,
        values: tuple[int, int, int, int, int, int],
    ) -> tuple[int, int, int, int, int, int]:
        """Reject non-positive round scoring values."""

        if any(value <= 0 for value in values):
            msg = "round_values must all be positive"
            raise ValueError(msg)
        return values


class PoolSettings(BaseModel):
    """Shared pool configuration passed into analysis and optimization."""

    model_config = ConfigDict(frozen=True)

    pool_size: int = Field(ge=1)
    scoring_system: ScoringSystemKey = Field(default=ScoringSystemKey.ESPN)


class BracketEditPick(BaseModel):
    """One editable bracket pick that may be missing or locked."""

    model_config = ConfigDict(frozen=True)

    game_id: str = Field(min_length=1)
    winner_team_id: str | None = None
    locked: bool = False

    @field_validator("winner_team_id")
    @classmethod
    def normalize_winner_team_id(cls, value: str | None) -> str | None:
        """Normalize blank team ids to null for partial brackets."""

        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_locked_pick(self) -> BracketEditPick:
        """Reject locked picks without a chosen winner."""

        if self.locked and self.winner_team_id is None:
            msg = "locked picks require winner_team_id"
            raise ValueError(msg)
        return self


class EditableBracket(BaseModel):
    """Partial or complete bracket state for product workflows."""

    model_config = ConfigDict(frozen=True)

    picks: list[BracketEditPick]

    @field_validator("picks")
    @classmethod
    def picks_must_be_unique_by_game(
        cls,
        picks: list[BracketEditPick],
    ) -> list[BracketEditPick]:
        """Reject duplicate editable picks for the same game."""

        seen: set[str] = set()
        for pick in picks:
            if pick.game_id in seen:
                msg = f"Duplicate editable pick for game {pick.game_id}"
                raise ValueError(msg)
            seen.add(pick.game_id)
        return picks


class PickDiagnostic(BaseModel):
    """Per-pick analysis details surfaced to the web/API layer."""

    model_config = ConfigDict(frozen=True)

    game_id: str = Field(min_length=1)
    team_id: str = Field(min_length=1)
    team_name: str = Field(min_length=1)
    round: int = Field(ge=1, le=6)
    survival_probability: float = Field(ge=0.0, le=1.0)
    win_probability_if_picked: float = Field(ge=0.0, le=1.0)
    delta_win_probability_if_picked: float
    tags: list[PickDiagnosticTag] = Field(default_factory=list)


class BracketAnalysis(BaseModel):
    """Stable response contract for full-bracket analysis results."""

    model_config = ConfigDict(frozen=True)

    bracket: EditableBracket
    pool_settings: PoolSettings
    completion_mode: CompletionMode
    dataset_hash: str = Field(min_length=64, max_length=64)
    cache_key: str = Field(min_length=1)
    win_probability: float = Field(ge=0.0, le=1.0)
    public_percentile: float | None = Field(default=None, ge=0.0, le=1.0)
    pick_diagnostics: list[PickDiagnostic]


class OptimizationAlternative(BaseModel):
    """One alternative bracket recommendation from the optimizer."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(min_length=1)
    bracket: EditableBracket
    projected_win_probability: float = Field(ge=0.0, le=1.0)
    changed_pick_count: int = Field(ge=0)
    summary: str | None = None


class OptimizationResult(BaseModel):
    """Stable response contract for optimization workflows."""

    model_config = ConfigDict(frozen=True)

    pool_settings: PoolSettings
    completion_mode: CompletionMode
    dataset_hash: str = Field(min_length=64, max_length=64)
    cache_key: str = Field(min_length=1)
    recommended_bracket: EditableBracket
    projected_win_probability: float = Field(ge=0.0, le=1.0)
    changed_pick_count: int = Field(ge=0)
    alternatives: list[OptimizationAlternative]


class CompletionModeOption(BaseModel):
    """Display metadata for one completion mode in the UI/API."""

    model_config = ConfigDict(frozen=True)

    mode: CompletionMode
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    alias_of: CompletionMode | None = None
    implemented: bool = False


class CachePolicy(BaseModel):
    """Document the shared dataset-hash and cache-key strategy."""

    model_config = ConfigDict(frozen=True)

    manifest_schema_version: int = 1
    dataset_hash_rule: str = Field(min_length=1)
    cache_key_rule: str = Field(min_length=1)
    artifact_kinds: list[CacheArtifactKind]


class ProductWorkflow(BaseModel):
    """One top-level workflow shown in the integrated web app."""

    model_config = ConfigDict(frozen=True)

    key: WorkflowKey
    label: str = Field(min_length=1)
    timing: str = Field(min_length=1)
    description: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    state: WorkflowState


class ProductFoundation(BaseModel):
    """Bootstrap payload for the integrated web shell."""

    model_config = ConfigDict(frozen=True)

    app_name: str = Field(min_length=1)
    roadmap_phase: str = Field(min_length=1)
    workflows: list[ProductWorkflow]
    scoring_systems: list[ScoringSystem]
    completion_modes: list[CompletionModeOption]
    cache_policy: CachePolicy


class CacheKeyPreviewRequest(BaseModel):
    """Request payload for previewing an analysis or optimization cache key."""

    model_config = ConfigDict(frozen=True)

    artifact_kind: CacheArtifactKind
    dataset_hash: str = Field(min_length=64, max_length=64)
    pool_settings: PoolSettings
    completion_mode: CompletionMode = CompletionMode.MANUAL


class CacheKeyPreview(BaseModel):
    """Preview response for the shared cache-key rules."""

    model_config = ConfigDict(frozen=True)

    artifact_kind: CacheArtifactKind
    dataset_hash: str = Field(min_length=64, max_length=64)
    cache_key: str = Field(min_length=1)


class AnalyzeBracketRequest(BaseModel):
    """Request payload for a manual Bracket Lab analysis run."""

    model_config = ConfigDict(frozen=True)

    bracket: EditableBracket
    pool_settings: PoolSettings
    completion_mode: CompletionMode = CompletionMode.MANUAL


class BracketLabBootstrap(BaseModel):
    """Prepared Bracket Lab data required to render the manual editor."""

    model_config = ConfigDict(frozen=True)

    dataset_hash: str = Field(min_length=64, max_length=64)
    completion_mode: CompletionMode = CompletionMode.MANUAL
    default_pool_settings: PoolSettings
    teams: list[Team]
    games: list[Game]


class SaveBracketRequest(BaseModel):
    """Request payload for persisting one manual bracket draft to local disk."""

    model_config = ConfigDict(frozen=True)

    bracket_id: str | None = Field(default=None, min_length=1)
    name: str = Field(min_length=1, max_length=120)
    bracket: EditableBracket
    pool_settings: PoolSettings
    completion_mode: CompletionMode = CompletionMode.MANUAL

    @field_validator("bracket_id")
    @classmethod
    def normalize_bracket_id(cls, value: str | None) -> str | None:
        """Normalize blank bracket ids to null."""

        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Trim saved bracket names and reject blanks."""

        normalized = value.strip()
        if not normalized:
            msg = "Saved bracket name is required"
            raise ValueError(msg)
        return normalized


class SavedBracketSummary(BaseModel):
    """One saved bracket row shown in dropdown selectors."""

    model_config = ConfigDict(frozen=True)

    bracket_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)
    pool_settings: PoolSettings
    completion_mode: CompletionMode = CompletionMode.MANUAL
    dataset_hash: str = Field(min_length=64, max_length=64)
    updated_at: datetime


class SavedBracket(BaseModel):
    """Full saved bracket record persisted to local disk."""

    model_config = ConfigDict(frozen=True)

    bracket_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)
    bracket: EditableBracket
    pool_settings: PoolSettings
    completion_mode: CompletionMode = CompletionMode.MANUAL
    dataset_hash: str = Field(min_length=64, max_length=64)
    updated_at: datetime

    def to_summary(self) -> SavedBracketSummary:
        """Return dropdown-friendly metadata for this saved bracket."""

        return SavedBracketSummary(
            bracket_id=self.bracket_id,
            name=self.name,
            pool_settings=self.pool_settings,
            completion_mode=self.completion_mode,
            dataset_hash=self.dataset_hash,
            updated_at=self.updated_at,
        )


class SavedBracketList(BaseModel):
    """Response payload for listing saved local bracket drafts."""

    model_config = ConfigDict(frozen=True)

    brackets: list[SavedBracketSummary]
