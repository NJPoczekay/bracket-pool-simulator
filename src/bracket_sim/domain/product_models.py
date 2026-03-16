"""Shared product-facing models for the web/API surface."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    AP_POLL = "ap_poll"
    NCAA_NET = "ncaa_net"
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
    public_percentile: float = Field(ge=0.0, le=1.0)
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
    implemented: bool = False


class CachePolicy(BaseModel):
    """Document the shared dataset-hash and cache-key strategy."""

    model_config = ConfigDict(frozen=True)

    manifest_schema_version: int = 1
    dataset_hash_rule: str = Field(min_length=1)
    cache_key_rule: str = Field(min_length=1)
    artifact_kinds: list[CacheArtifactKind]


class ProductFoundation(BaseModel):
    """Bootstrap payload for the phase-0 web shell."""

    model_config = ConfigDict(frozen=True)

    app_name: str = Field(min_length=1)
    roadmap_phase: str = Field(min_length=1)
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
