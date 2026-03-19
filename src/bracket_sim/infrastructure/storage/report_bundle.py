"""Helpers for writing deterministic report bundle artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from bracket_sim import __version__
from bracket_sim.domain.models import (
    ChampionSensitivityRow,
    EntryReportRow,
    ReportArtifact,
    ReportBundleManifest,
    ReportConfig,
    ReportSummary,
    TeamAdvancementOddsRow,
)
from bracket_sim.infrastructure.storage.cache_keys import capture_dataset_hash
from bracket_sim.infrastructure.storage.run_artifacts import (
    capture_input_hashes,
    read_git_commit,
    utc_now,
    write_json_atomic,
)


@dataclass(frozen=True)
class ReportArtifactPaths:
    """Canonical file locations under one report output directory."""

    output_dir: Path
    manifest_path: Path
    summary_path: Path
    team_advancement_path: Path
    entry_summary_path: Path
    champion_sensitivity_path: Path


def build_report_artifact_paths(output_dir: Path) -> ReportArtifactPaths:
    """Return the standard artifact paths for one report bundle."""

    return ReportArtifactPaths(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        summary_path=output_dir / "summary.json",
        team_advancement_path=output_dir / "team_advancement_odds.csv",
        entry_summary_path=output_dir / "entry_summary.csv",
        champion_sensitivity_path=output_dir / "champion_sensitivity.csv",
    )


def ensure_report_output_dir(output_dir: Path) -> None:
    """Create the report output directory if needed."""

    output_dir.mkdir(parents=True, exist_ok=True)


def ensure_fresh_report_output_dir(paths: ReportArtifactPaths) -> None:
    """Reject report bundles that would overwrite canonical artifacts."""

    existing = [
        path
        for path in (
            paths.manifest_path,
            paths.summary_path,
            paths.team_advancement_path,
            paths.entry_summary_path,
            paths.champion_sensitivity_path,
        )
        if path.exists()
    ]
    if existing:
        msg = (
            "Report output directory already contains artifacts; choose a new --out directory: "
            f"{existing[0].parent}"
        )
        raise ValueError(msg)


def generate_report_id(*, config: ReportConfig) -> str:
    """Return a deterministic identifier for one report configuration."""

    payload = {
        "input_hashes": capture_input_hashes(config.input_dir),
        "n_sims": config.n_sims,
        "seed": config.seed,
        "rating_scale": config.rating_scale,
        "batch_size": config.effective_batch_size,
        "engine": config.engine,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_report_manifest(
    *,
    config: ReportConfig,
    report_id: str,
    entry_ids: list[str],
    team_ids: list[str],
    artifacts: list[ReportArtifact],
) -> ReportBundleManifest:
    """Capture reproducibility metadata for a report bundle."""

    return ReportBundleManifest(
        report_id=report_id,
        created_at=utc_now(),
        code_version=__version__,
        git_commit=read_git_commit(),
        input_dir=config.input_dir,
        dataset_hash=capture_dataset_hash(config.input_dir),
        input_hashes=capture_input_hashes(config.input_dir),
        output_dir=config.output_dir,
        n_sims=config.n_sims,
        seed=config.seed,
        rating_scale=config.rating_scale,
        batch_size=config.effective_batch_size,
        engine=config.engine,
        entry_ids=entry_ids,
        team_ids=team_ids,
        artifacts=artifacts,
    )


def write_report_summary(path: Path, summary: ReportSummary) -> ReportArtifact:
    """Persist the summary JSON and return artifact metadata."""

    write_json_atomic(path=path, payload=summary.model_dump(mode="json"))
    return ReportArtifact(
        name=path.name,
        path=path,
        kind="json",
        sha256=_sha256_path(path),
    )


def write_team_advancement_csv(
    path: Path,
    rows: list[TeamAdvancementOddsRow],
) -> ReportArtifact:
    """Persist team advancement probabilities as CSV."""

    return _write_csv_rows(
        path=path,
        rows=[row.model_dump(mode="json") for row in rows],
    )


def write_entry_summary_csv(path: Path, rows: list[EntryReportRow]) -> ReportArtifact:
    """Persist entry summary metrics as CSV."""

    return _write_csv_rows(
        path=path,
        rows=[
            {
                "rank": row.rank,
                "entry_id": row.entry_id,
                "entry_name": row.entry_name,
                "win_percentage": row.win_share * 100,
                "average_score": row.average_score,
            }
            for row in rows
        ],
    )


def write_champion_sensitivity_csv(
    path: Path,
    rows: list[ChampionSensitivityRow],
) -> ReportArtifact:
    """Persist champion-conditioned sensitivity metrics as CSV."""

    return _write_csv_rows(
        path=path,
        rows=[
            {
                "champion_team_id": row.champion_team_id,
                "champion_team_name": row.champion_team_name,
                "champion_probability": row.champion_probability,
                "champion_simulations": row.champion_simulations,
                "entry_rank": row.entry_rank,
                "entry_id": row.entry_id,
                "entry_name": row.entry_name,
                "baseline_win_percentage": row.baseline_win_share * 100,
                "conditional_win_percentage": row.conditional_win_share * 100,
                "win_percentage_point_delta": row.win_share_delta * 100,
                "baseline_average_score": row.baseline_average_score,
                "conditional_average_score": row.conditional_average_score,
                "average_score_delta": row.average_score_delta,
            }
            for row in rows
        ],
    )


def write_report_manifest(path: Path, manifest: ReportBundleManifest) -> None:
    """Persist the report manifest atomically."""

    write_json_atomic(path=path, payload=manifest.model_dump(mode="json"))


def publish_latest_report(*, archive_dir: Path, latest_dir: Path) -> None:
    """Refresh a materialized latest-report directory from one archive bundle."""

    latest_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = latest_dir.parent / f".{latest_dir.name}.tmp-{uuid4().hex}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
        for source_path in _canonical_report_paths(archive_dir):
            shutil.copy2(source_path, staging_dir / source_path.name)

        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        staging_dir.rename(latest_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def _write_csv_rows(*, path: Path, rows: list[dict[str, object]]) -> ReportArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    fieldnames = list(rows[0]) if rows else []

    with staging_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    staging_path.replace(path)
    return ReportArtifact(
        name=path.name,
        path=path,
        kind="csv",
        sha256=_sha256_path(path),
        row_count=len(rows),
    )


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_report_paths(output_dir: Path) -> tuple[Path, ...]:
    paths = build_report_artifact_paths(output_dir)
    return (
        paths.manifest_path,
        paths.summary_path,
        paths.team_advancement_path,
        paths.entry_summary_path,
        paths.champion_sensitivity_path,
    )
