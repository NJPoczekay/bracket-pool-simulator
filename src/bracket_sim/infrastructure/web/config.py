"""Configuration loading for the Pool Tracker portion of the integrated web app."""

from __future__ import annotations

import tomllib
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bracket_sim.application.run_pool_pipeline import PoolPipelineConfig
from bracket_sim.infrastructure.providers.espn_api import parse_espn_group_url
from bracket_sim.infrastructure.storage.path_defaults import (
    build_tracker_paths,
    tracker_context_from_pool,
)


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
    """One configured tracker pool exposed by the integrated web app."""

    schedule: PoolSchedule | None = None

    def scheduler_zoneinfo(self) -> ZoneInfo | None:
        """Return the pool-specific scheduler timezone when scheduling is enabled."""

        if self.schedule is None or not self.schedule.enabled:
            return None
        return self.schedule.zoneinfo


class PoolRegistry(BaseModel):
    """Collection wrapper for all configured tracker pools."""

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


class PoolProfileSource(BaseModel):
    """Config-file representation before relative/default path resolution."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    group_url: str = Field(min_length=1)
    raw_dir: Path | None = None
    prepared_dir: Path | None = None
    reports_root: Path | None = None
    ratings_file: Path | None = None
    use_kenpom: bool = False
    min_usable_entries: int = Field(default=1, ge=1)
    n_sims: int = Field(gt=0)
    seed: int
    batch_size: int | None = Field(default=None, gt=0)
    engine: str = Field(default="numpy")
    schedule: PoolSchedule | None = None


class PoolRegistrySource(BaseModel):
    """Config-file wrapper before path defaults are materialized."""

    model_config = ConfigDict(frozen=True)

    pools: list[PoolProfileSource] = Field(min_length=1)


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
        source_registry = PoolRegistrySource.model_validate(payload)
    except ValidationError as exc:
        msg = f"Invalid pool config {path.name}: {exc}"
        raise ValueError(msg) from exc

    base_dir = path.parent.resolve()
    resolved_pools = []
    for pool in source_registry.pools:
        group_ref = parse_espn_group_url(pool.group_url)
        defaults = build_tracker_paths(
            base_dir=base_dir,
            context=tracker_context_from_pool(
                challenge_key=group_ref.challenge_key,
                pool_id=pool.id,
            ),
        )
        resolved_pools.append(
            PoolProfile.model_validate(
                {
                    "id": pool.id,
                    "name": pool.name,
                    "group_url": pool.group_url,
                    "raw_dir": _resolve_path(
                        base_dir=base_dir,
                        value=pool.raw_dir or defaults.raw_dir,
                    ),
                    "prepared_dir": _resolve_path(
                        base_dir=base_dir,
                        value=pool.prepared_dir or defaults.prepared_dir,
                    ),
                    "reports_root": _resolve_path(
                        base_dir=base_dir,
                        value=pool.reports_root or defaults.reports_root,
                    ),
                    "ratings_file": (
                        _resolve_path(base_dir=base_dir, value=pool.ratings_file)
                        if pool.ratings_file is not None
                        else None
                    ),
                    "use_kenpom": pool.use_kenpom,
                    "min_usable_entries": pool.min_usable_entries,
                    "n_sims": pool.n_sims,
                    "seed": pool.seed,
                    "batch_size": pool.batch_size,
                    "engine": pool.engine,
                    "schedule": pool.schedule,
                }
            )
        )
    return PoolRegistry(pools=resolved_pools)


def _resolve_path(*, base_dir: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return (base_dir / value).resolve()
