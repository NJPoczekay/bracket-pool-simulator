"""Shared helpers for safe default storage paths and storage-context inference."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_YEAR_SUFFIX_PATTERN = re.compile(r"(20\d{2})$")
_NON_TOKEN_PATTERN = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True)
class StorageContext:
    """Minimal metadata needed to derive default output locations."""

    workflow: str
    season: str
    dataset_slug: str

    def to_metadata(self) -> dict[str, str]:
        return {
            "workflow": self.workflow,
            "season": self.season,
            "dataset_slug": self.dataset_slug,
        }


@dataclass(frozen=True)
class TrackerStoragePaths:
    """Default tracker directories for one season + slug."""

    raw_dir: Path
    prepared_dir: Path
    reports_root: Path
    runs_root: Path


@dataclass(frozen=True)
class BracketLabStoragePaths:
    """Default Bracket Lab directories for one season + challenge."""

    raw_dir: Path
    prepared_dir: Path


@dataclass(frozen=True)
class ReportPublishTargets:
    """Default report root and latest directory for one prepared dataset."""

    context: StorageContext
    reports_root: Path
    latest_dir: Path


def safe_path_token(value: str, *, default: str) -> str:
    """Normalize free-form identifiers into safe path tokens."""

    normalized = _NON_TOKEN_PATTERN.sub("-", value.strip().lower())
    normalized = normalized.strip("-._")
    return normalized or default


def season_from_challenge_key(challenge_key: str) -> str:
    """Return the season token inferred from a challenge key."""

    match = _YEAR_SUFFIX_PATTERN.search(challenge_key.strip())
    if match is not None:
        return match.group(1)
    return str(datetime.now(UTC).year)


def tracker_context_from_group(*, challenge_key: str, group_id: str) -> StorageContext:
    """Build tracker storage context from an ESPN group reference."""

    return StorageContext(
        workflow="tracker",
        season=season_from_challenge_key(challenge_key),
        dataset_slug=safe_path_token(group_id, default="tracker"),
    )


def tracker_context_from_pool(*, challenge_key: str, pool_id: str) -> StorageContext:
    """Build tracker storage context from pool config identity."""

    return StorageContext(
        workflow="tracker",
        season=season_from_challenge_key(challenge_key),
        dataset_slug=safe_path_token(pool_id, default="pool"),
    )


def bracket_lab_context_from_challenge(challenge_key: str) -> StorageContext:
    """Build Bracket Lab storage context from one challenge key."""

    slug = safe_path_token(challenge_key, default="bracket-lab")
    return StorageContext(
        workflow="bracket-lab",
        season=season_from_challenge_key(challenge_key),
        dataset_slug=slug,
    )


def national_picks_context_from_challenge(challenge_key: str) -> StorageContext:
    """Build national-picks storage context from one challenge key."""

    slug = safe_path_token(challenge_key, default="national-picks")
    return StorageContext(
        workflow="national-picks",
        season=season_from_challenge_key(challenge_key),
        dataset_slug=slug,
    )


def build_tracker_paths(*, base_dir: Path, context: StorageContext) -> TrackerStoragePaths:
    """Return the default tracker raw/prepared/report/run directories."""

    season_root = base_dir / "data" / context.season / "tracker" / context.dataset_slug
    report_root = base_dir / "reports" / context.season / "tracker" / context.dataset_slug
    run_root = base_dir / "runs" / context.season / "tracker" / context.dataset_slug
    return TrackerStoragePaths(
        raw_dir=season_root / "raw",
        prepared_dir=season_root / "prepared",
        reports_root=report_root,
        runs_root=run_root,
    )


def build_bracket_lab_paths(*, base_dir: Path, context: StorageContext) -> BracketLabStoragePaths:
    """Return the default Bracket Lab raw/prepared directories."""

    season_root = base_dir / "data" / context.season / "bracket-lab" / context.dataset_slug
    return BracketLabStoragePaths(
        raw_dir=season_root / "raw",
        prepared_dir=season_root / "prepared",
    )


def build_national_picks_dir(*, base_dir: Path, context: StorageContext) -> Path:
    """Return the default national-picks directory."""

    return base_dir / "data" / context.season / "national-picks" / context.dataset_slug


def derive_prepared_out_dir(raw_dir: Path) -> Path:
    """Derive a sibling prepared directory from a raw directory."""

    if raw_dir.name == "raw":
        return raw_dir.with_name("prepared")
    return raw_dir.parent / "prepared"


def tracker_context_from_raw(
    *,
    raw_dir: Path,
    raw_metadata: dict[str, Any] | None,
) -> StorageContext:
    """Infer tracker storage context from raw metadata or path shape."""

    if raw_metadata is not None:
        source = raw_metadata.get("source")
        if isinstance(source, dict):
            challenge_key = source.get("challenge_key")
            group_id = source.get("group_id")
            if isinstance(challenge_key, str) and isinstance(group_id, str):
                return tracker_context_from_group(challenge_key=challenge_key, group_id=group_id)

    inferred = infer_storage_context_from_path(raw_dir)
    if inferred is not None and inferred.workflow == "tracker":
        return inferred

    slug_source = raw_dir.parent.name if raw_dir.name == "raw" else raw_dir.name
    return StorageContext(
        workflow="tracker",
        season=_infer_year_from_path(raw_dir),
        dataset_slug=safe_path_token(slug_source, default="tracker"),
    )


def bracket_lab_context_from_raw(
    *,
    raw_dir: Path,
    raw_metadata: dict[str, Any] | None,
) -> StorageContext:
    """Infer Bracket Lab storage context from raw metadata or path shape."""

    if raw_metadata is not None:
        source = raw_metadata.get("source")
        if isinstance(source, dict):
            challenge_key = source.get("challenge_key")
            if isinstance(challenge_key, str):
                return bracket_lab_context_from_challenge(challenge_key)

    inferred = infer_storage_context_from_path(raw_dir)
    if inferred is not None and inferred.workflow == "bracket-lab":
        return inferred

    slug_source = raw_dir.parent.name if raw_dir.name == "raw" else raw_dir.name
    return StorageContext(
        workflow="bracket-lab",
        season=_infer_year_from_path(raw_dir),
        dataset_slug=safe_path_token(slug_source, default="bracket-lab"),
    )


def report_publish_targets_for_input(*, input_dir: Path, base_dir: Path) -> ReportPublishTargets:
    """Infer the default report root and latest dir for one prepared dataset."""

    context = load_storage_context(input_dir)
    if context is None:
        inferred = infer_storage_context_from_path(input_dir)
        if inferred is not None:
            context = inferred
        else:
            context = StorageContext(
                workflow="tracker",
                season=_infer_year_from_path(input_dir),
                dataset_slug=safe_path_token(input_dir.name, default="report"),
            )

    reports_root = base_dir / "reports" / context.season / context.workflow / context.dataset_slug
    return ReportPublishTargets(
        context=context,
        reports_root=reports_root,
        latest_dir=reports_root / "latest",
    )


def load_storage_context(input_dir: Path) -> StorageContext | None:
    """Read optional prepared metadata and extract storage context when present."""

    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None

    storage = payload.get("storage")
    if not isinstance(storage, dict):
        return None

    workflow = storage.get("workflow")
    season = storage.get("season")
    dataset_slug = storage.get("dataset_slug")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (workflow, season, dataset_slug)
    ):
        return None

    return StorageContext(
        workflow=workflow.strip(),
        season=season.strip(),
        dataset_slug=dataset_slug.strip(),
    )


def infer_storage_context_from_path(path: Path) -> StorageContext | None:
    """Infer storage context from known season-first directory layouts."""

    parts = path.resolve().parts
    for index in range(len(parts) - 4):
        if parts[index] != "data":
            continue
        season = parts[index + 1]
        workflow = parts[index + 2]
        dataset_slug = parts[index + 3]
        leaf = parts[index + 4]
        if not _looks_like_year(season):
            continue
        if workflow not in {"tracker", "bracket-lab"}:
            continue
        if leaf != "prepared" and leaf != "raw":
            continue
        return StorageContext(
            workflow=workflow,
            season=season,
            dataset_slug=dataset_slug,
        )
    return None


def default_report_timestamp(*, now: datetime | None = None) -> datetime:
    """Return a timezone-aware timestamp for report directory generation."""

    return now or datetime.now(UTC)


def _infer_year_from_path(path: Path) -> str:
    for part in reversed(path.resolve().parts):
        if _looks_like_year(part):
            return part
    return str(datetime.now(UTC).year)


def _looks_like_year(value: str) -> bool:
    return bool(re.fullmatch(r"20\d{2}", value))
