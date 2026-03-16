"""Configuration loading for the local multi-pool web wrapper."""

from __future__ import annotations

import tomllib
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bracket_sim.application.run_pool_pipeline import PoolPipelineConfig


class PoolSchedule(BaseModel):
    """Optional once-per-day schedule for one configured pool."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    daily_time: time | None = None
    timezone: str | None = None

    @model_validator(mode="after")
    def validate_schedule(self) -> PoolSchedule:
        """Require complete and valid scheduling inputs when enabled."""

        if self.enabled and self.daily_time is None:
            msg = "enabled schedules require daily_time"
            raise ValueError(msg)
        if self.enabled and self.timezone is None:
            msg = "enabled schedules require timezone"
            raise ValueError(msg)
        if self.timezone is not None:
            try:
                ZoneInfo(self.timezone)
            except ZoneInfoNotFoundError as exc:
                msg = f"Unknown timezone {self.timezone!r}"
                raise ValueError(msg) from exc
        return self

    @property
    def zoneinfo(self) -> ZoneInfo:
        """Return the configured schedule timezone."""

        if self.timezone is None:
            msg = "timezone is not configured"
            raise ValueError(msg)
        return ZoneInfo(self.timezone)


class PoolProfile(PoolPipelineConfig):
    """One configured bracket pool exposed by the web wrapper."""

    schedule: PoolSchedule | None = None

    def scheduler_zoneinfo(self) -> ZoneInfo | None:
        """Return the pool-specific scheduler timezone when scheduling is enabled."""

        if self.schedule is None or not self.schedule.enabled:
            return None
        return self.schedule.zoneinfo


class PoolRegistry(BaseModel):
    """Collection wrapper for all configured web pools."""

    model_config = ConfigDict(frozen=True)

    pools: list[PoolProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_pool_ids(self) -> PoolRegistry:
        """Reject duplicate pool identifiers."""

        seen: set[str] = set()
        for pool in self.pools:
            if pool.id in seen:
                msg = f"Duplicate pool id {pool.id!r}"
                raise ValueError(msg)
            seen.add(pool.id)
        return self


def load_pool_registry(path: Path) -> PoolRegistry:
    """Load and validate the pool registry TOML file."""

    if not path.exists():
        msg = f"Pool config file does not exist: {path}"
        raise ValueError(msg)

    with path.open("rb") as handle:
        try:
            payload = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            msg = f"Invalid TOML in pool config {path.name}: {exc}"
            raise ValueError(msg) from exc

    try:
        registry = PoolRegistry.model_validate(payload)
    except ValidationError as exc:
        msg = f"Invalid pool config {path.name}: {exc}"
        raise ValueError(msg) from exc

    base_dir = path.parent.resolve()
    resolved_pools = [
        pool.model_copy(
            update={
                "raw_dir": _resolve_path(base_dir=base_dir, value=pool.raw_dir),
                "prepared_dir": _resolve_path(base_dir=base_dir, value=pool.prepared_dir),
                "reports_root": _resolve_path(base_dir=base_dir, value=pool.reports_root),
                "ratings_file": (
                    _resolve_path(base_dir=base_dir, value=pool.ratings_file)
                    if pool.ratings_file is not None
                    else None
                ),
            }
        )
        for pool in registry.pools
    ]
    return registry.model_copy(update={"pools": resolved_pools})


def _resolve_path(*, base_dir: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return (base_dir / value).resolve()
