"""Shared phase-0 product metadata for the web/API layer."""

from __future__ import annotations

from bracket_sim.domain.product_models import (
    CacheArtifactKind,
    CacheKeyPreview,
    CacheKeyPreviewRequest,
    CachePolicy,
    CompletionMode,
    CompletionModeOption,
    ProductFoundation,
    ScoringSystem,
    ScoringSystemKey,
)
from bracket_sim.infrastructure.storage.cache_keys import build_cache_key


def build_product_foundation() -> ProductFoundation:
    """Return the shared product metadata exposed by the phase-0 API."""

    return ProductFoundation(
        app_name="Bracket Pool Simulator",
        roadmap_phase="phase_0_foundation",
        scoring_systems=_scoring_systems(),
        completion_modes=_completion_modes(),
        cache_policy=CachePolicy(
            dataset_hash_rule=(
                "Hash every top-level .json/.csv/.parquet dataset artifact in sorted filename "
                "order, then hash that file-hash manifest."
            ),
            cache_key_rule=(
                "Combine artifact kind, dataset hash, and the JSON-serialized settings payload "
                "into a deterministic SHA-256 key."
            ),
            artifact_kinds=[
                CacheArtifactKind.ANALYSIS,
                CacheArtifactKind.OPTIMIZATION,
            ],
        ),
    )


def preview_cache_key(request: CacheKeyPreviewRequest) -> CacheKeyPreview:
    """Return a stable cache key preview for analysis or optimization settings."""

    cache_key = build_cache_key(
        artifact_kind=request.artifact_kind.value,
        dataset_hash=request.dataset_hash,
        settings={
            "pool_settings": request.pool_settings,
            "completion_mode": request.completion_mode,
        },
    )
    return CacheKeyPreview(
        artifact_kind=request.artifact_kind,
        dataset_hash=request.dataset_hash,
        cache_key=cache_key,
    )


def _scoring_systems() -> list[ScoringSystem]:
    return [
        ScoringSystem(
            key=ScoringSystemKey.ESPN,
            label="ESPN",
            round_values=(1, 2, 4, 8, 16, 32),
            implemented=True,
            description="Current simulator scoring and the phase-0 default.",
        ),
        ScoringSystem(
            key=ScoringSystemKey.LINEAR,
            label="Linear",
            round_values=(1, 2, 3, 4, 5, 6),
            description="Planned for phase 2 analyzer support.",
        ),
        ScoringSystem(
            key=ScoringSystemKey.FIBONACCI,
            label="Fibonacci",
            round_values=(2, 3, 5, 8, 13, 21),
            description="Planned for phase 2 analyzer support.",
        ),
        ScoringSystem(
            key=ScoringSystemKey.ROUND_PLUS_SEED,
            label="Round Plus Seed",
            round_values=(1, 2, 3, 4, 5, 6),
            seed_bonus=True,
            description="Planned for phase 2 analyzer support with a seed bonus overlay.",
        ),
    ]


def _completion_modes() -> list[CompletionModeOption]:
    return [
        CompletionModeOption(
            mode=CompletionMode.MANUAL,
            label="Manual Picks",
            description="Edit the bracket directly without auto-completion.",
            implemented=True,
        ),
        CompletionModeOption(
            mode=CompletionMode.TOURNAMENT_SEEDS,
            label="Tournament Seeds",
            description="Auto-complete by seed order.",
        ),
        CompletionModeOption(
            mode=CompletionMode.POPULAR_PICKS,
            label="Popular Picks",
            description="Auto-complete from public pick rates.",
        ),
        CompletionModeOption(
            mode=CompletionMode.INTERNAL_MODEL_RANK,
            label="Model Rank",
            description="Auto-complete using the simulator's internal ranking model.",
        ),
        CompletionModeOption(
            mode=CompletionMode.KENPOM,
            label="KenPom",
            description="Auto-complete from KenPom rankings.",
        ),
        CompletionModeOption(
            mode=CompletionMode.AP_POLL,
            label="AP Poll",
            description="Auto-complete from AP Poll rankings.",
        ),
        CompletionModeOption(
            mode=CompletionMode.NCAA_NET,
            label="NCAA NET",
            description="Auto-complete from NCAA NET rankings.",
        ),
        CompletionModeOption(
            mode=CompletionMode.PICK_FOUR,
            label="Pick Four",
            description="Seed the search from Final Four choices before auto-completion.",
        ),
    ]
