"""Application orchestration for one configured pool pipeline run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bracket_sim.application.generate_reports import generate_reports
from bracket_sim.application.prepare_data import PrepareDataSummary, prepare_data
from bracket_sim.application.refresh_data import RefreshDataSummary, refresh_data
from bracket_sim.domain.models import ReportBundleResult, ReportConfig
from bracket_sim.domain.scoring_systems import ScoringSystemKey
from bracket_sim.infrastructure.providers.contracts import (
    EntriesProvider,
    RatingsProvider,
    ResultsProvider,
)
from bracket_sim.infrastructure.storage.report_bundle import publish_latest_report

_ALLOWED_ENGINES = {"numpy", "numba"}
REPORT_DIR_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


class PoolPipelineConfig(BaseModel):
    """Read-only configuration for one end-to-end pool pipeline run."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    group_url: str = Field(min_length=1)
    raw_dir: Path
    prepared_dir: Path
    reports_root: Path
    ratings_file: Path | None = None
    use_kenpom: bool = False
    min_usable_entries: int = Field(default=1, ge=1)
    n_sims: int = Field(gt=0)
    seed: int
    batch_size: int | None = Field(default=None, gt=0)
    engine: str = Field(default="numpy")
    scoring_system: ScoringSystemKey = Field(default=ScoringSystemKey.ESPN)

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, value: str) -> str:
        """Normalize and validate engine selection."""

        normalized = value.strip().lower()
        if normalized not in _ALLOWED_ENGINES:
            msg = f"Unsupported engine {value!r}; expected one of {sorted(_ALLOWED_ENGINES)}"
            raise ValueError(msg)
        return normalized

    @model_validator(mode="after")
    def validate_output_dirs(self) -> PoolPipelineConfig:
        """Reject obviously conflicting directory choices."""

        if self.raw_dir == self.prepared_dir:
            msg = "raw_dir and prepared_dir must point to different directories"
            raise ValueError(msg)
        return self


@dataclass(frozen=True)
class PoolPipelineResult:
    """Artifacts produced by one successful configured pool run."""

    config: PoolPipelineConfig
    raw_dir: Path
    prepared_dir: Path
    report_dir: Path
    refresh_summary: RefreshDataSummary
    prepare_summary: PrepareDataSummary
    report_result: ReportBundleResult


def create_report_output_dir(*, reports_root: Path, started_at: datetime) -> Path:
    """Return a timestamped report output directory that will not collide."""

    if started_at.tzinfo is None:
        msg = "started_at must be timezone-aware"
        raise ValueError(msg)

    base_name = started_at.strftime(REPORT_DIR_TIMESTAMP_FORMAT)
    candidate = reports_root / base_name
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = reports_root / f"{base_name}-{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def run_pool_pipeline(
    config: PoolPipelineConfig,
    *,
    started_at: datetime | None = None,
    results_provider: ResultsProvider | None = None,
    entries_provider: EntriesProvider | None = None,
    ratings_provider: RatingsProvider | None = None,
) -> PoolPipelineResult:
    """Refresh raw data, prepare normalized inputs, and build a report bundle."""

    run_started_at = started_at or datetime.now(UTC)
    if run_started_at.tzinfo is None:
        msg = "started_at must be timezone-aware"
        raise ValueError(msg)

    report_dir = create_report_output_dir(
        reports_root=config.reports_root,
        started_at=run_started_at,
    )
    refresh_summary = refresh_data(
        group_url=config.group_url,
        raw_dir=config.raw_dir,
        ratings_file=config.ratings_file,
        use_kenpom=config.use_kenpom,
        min_usable_entries=config.min_usable_entries,
        results_provider=results_provider,
        entries_provider=entries_provider,
        ratings_provider=ratings_provider,
        fetched_at=run_started_at.astimezone(UTC),
    )
    prepare_summary = prepare_data(raw_dir=config.raw_dir, out_dir=config.prepared_dir)
    report_result = generate_reports(
        ReportConfig(
            input_dir=config.prepared_dir,
            output_dir=report_dir,
            n_sims=config.n_sims,
            seed=config.seed,
            batch_size=config.batch_size,
            engine=config.engine,
            scoring_system=config.scoring_system,
        )
    )
    publish_latest_report(
        archive_dir=report_dir,
        latest_dir=config.reports_root / "latest",
    )
    return PoolPipelineResult(
        config=config,
        raw_dir=refresh_summary.output_dir,
        prepared_dir=prepare_summary.output_dir,
        report_dir=report_dir,
        refresh_summary=refresh_summary,
        prepare_summary=prepare_summary,
        report_result=report_result,
    )
