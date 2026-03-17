"""Core service layer for the Pool Tracker portion of the integrated app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Protocol

from bracket_sim.application.run_pool_pipeline import (
    REPORT_DIR_TIMESTAMP_FORMAT,
    PoolPipelineResult,
    run_pool_pipeline,
)
from bracket_sim.domain.models import ReportSummary
from bracket_sim.infrastructure.web.config import PoolProfile, PoolRegistry

REPORT_ARTIFACT_FILENAMES = (
    "summary.json",
    "manifest.json",
    "entry_summary.csv",
    "team_advancement_odds.csv",
    "champion_sensitivity.csv",
)


class PoolRunBusyError(RuntimeError):
    """Raised when another pool pipeline run is already active."""


class UnknownPoolError(KeyError):
    """Raised when a configured pool id is not found."""


class PoolRunner(Protocol):
    """Callable interface for running one configured pool pipeline."""

    def __call__(
        self,
        config: PoolProfile,
        *,
        started_at: datetime,
    ) -> PoolPipelineResult:
        """Run the configured pipeline and return its outputs."""


@dataclass(frozen=True)
class LatestReport:
    """Most recent successful report bundle discovered for one pool."""

    pool_id: str
    report_dir: Path
    report_timestamp: datetime
    summary: ReportSummary

    @property
    def artifact_paths(self) -> dict[str, Path]:
        """Return available canonical artifact paths inside the report directory."""

        return {
            filename: self.report_dir / filename
            for filename in REPORT_ARTIFACT_FILENAMES
            if (self.report_dir / filename).exists()
        }


class PoolService:
    """Local synchronous controller for Pool Tracker access."""

    def __init__(
        self,
        registry: PoolRegistry,
        *,
        runner: PoolRunner = run_pool_pipeline,
    ) -> None:
        self._pools_by_id = {pool.id: pool for pool in registry.pools}
        self._runner = runner
        self._run_lock = Lock()

    def list_pools(self) -> list[PoolProfile]:
        """Return configured pools in config order."""

        return list(self._pools_by_id.values())

    def get_pool(self, pool_id: str) -> PoolProfile:
        """Return one configured pool or raise an explicit lookup error."""

        try:
            return self._pools_by_id[pool_id]
        except KeyError as exc:
            raise UnknownPoolError(pool_id) from exc

    def is_busy(self) -> bool:
        """Return whether a manual or scheduled run is active."""

        return self._run_lock.locked()

    def run_pool(
        self,
        pool_id: str,
        *,
        now: datetime | None = None,
    ) -> PoolPipelineResult:
        """Run one pool pipeline synchronously under the global process lock."""

        pool = self.get_pool(pool_id)
        started_at = _pool_now(pool=pool, base_now=now or datetime.now(UTC))

        if not self._run_lock.acquire(blocking=False):
            msg = "Another pool run is already in progress"
            raise PoolRunBusyError(msg)

        try:
            return self._runner(pool, started_at=started_at)
        finally:
            self._run_lock.release()

    def get_latest_report(self, pool_id: str) -> LatestReport | None:
        """Return the newest successful report bundle for one pool."""

        return find_latest_report(self.get_pool(pool_id))

    def run_due_pools(
        self,
        *,
        now: datetime | None = None,
    ) -> list[str]:
        """Run any scheduled pools that are due and not already satisfied today."""

        current = now or datetime.now(UTC)
        ran_pool_ids: list[str] = []
        for pool in self.list_pools():
            latest_report = find_latest_report(pool)
            if not is_pool_due_today(pool=pool, latest_report=latest_report, now=current):
                continue
            try:
                self.run_pool(pool.id, now=current)
            except PoolRunBusyError:
                continue
            ran_pool_ids.append(pool.id)
        return ran_pool_ids


def find_latest_report(pool: PoolProfile) -> LatestReport | None:
    """Scan one pool's reports root and return the newest successful bundle."""

    if not pool.reports_root.exists():
        return None

    candidates: list[tuple[datetime, str, LatestReport]] = []
    for child in pool.reports_root.iterdir():
        if not child.is_dir():
            continue

        summary_path = child / "summary.json"
        if not summary_path.exists():
            continue

        report_timestamp = parse_report_dir_timestamp(child.name)
        if report_timestamp is None:
            continue

        try:
            summary = ReportSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
        except ValueError:
            continue

        candidates.append(
            (
                report_timestamp,
                child.name,
                LatestReport(
                    pool_id=pool.id,
                    report_dir=child,
                    report_timestamp=report_timestamp,
                    summary=summary,
                ),
            )
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def parse_report_dir_timestamp(dir_name: str) -> datetime | None:
    """Parse the sortable timestamp prefix used for report bundle directories."""

    prefix_length = len("20260315-143022")
    if len(dir_name) < prefix_length:
        return None

    prefix = dir_name[:prefix_length]
    try:
        return datetime.strptime(prefix, REPORT_DIR_TIMESTAMP_FORMAT)
    except ValueError:
        return None


def is_pool_due_today(
    *,
    pool: PoolProfile,
    latest_report: LatestReport | None,
    now: datetime,
) -> bool:
    """Return whether the pool's daily schedule should run now."""

    if now.tzinfo is None:
        msg = "now must be timezone-aware"
        raise ValueError(msg)

    schedule = pool.schedule
    if schedule is None or not schedule.enabled or schedule.daily_time is None:
        return False

    local_now = now.astimezone(schedule.zoneinfo)
    if local_now.time() < schedule.daily_time:
        return False

    return not (
        latest_report is not None and latest_report.report_timestamp.date() == local_now.date()
    )


def _pool_now(*, pool: PoolProfile, base_now: datetime) -> datetime:
    if base_now.tzinfo is None:
        msg = "base_now must be timezone-aware"
        raise ValueError(msg)

    zone = pool.scheduler_zoneinfo()
    if zone is None:
        return base_now.astimezone(UTC)
    return base_now.astimezone(zone)
