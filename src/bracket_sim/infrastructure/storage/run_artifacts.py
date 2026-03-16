"""Run manifest, checkpoint, and result artifact helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from bracket_sim import __version__
from bracket_sim.domain.models import RunCheckpoint, RunManifest, SimulationConfig, SimulationResult
from bracket_sim.infrastructure.storage.cache_keys import (
    capture_dataset_file_hashes,
    capture_dataset_hash,
)


@dataclass(frozen=True)
class RunArtifactPaths:
    """Canonical file locations under a simulation run directory."""

    run_dir: Path
    manifest_path: Path
    checkpoint_path: Path
    result_path: Path
    log_path: Path


def build_run_artifact_paths(run_dir: Path) -> RunArtifactPaths:
    """Return the standard artifact paths for one run directory."""

    return RunArtifactPaths(
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        checkpoint_path=run_dir / "checkpoint.json",
        result_path=run_dir / "result.json",
        log_path=run_dir / "log.jsonl",
    )


def ensure_run_dir(run_dir: Path) -> None:
    """Create the run directory if it does not already exist."""

    run_dir.mkdir(parents=True, exist_ok=True)


def generate_run_id(*, config: SimulationConfig) -> str:
    """Return a deterministic identifier for one simulation configuration."""

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


def capture_input_hashes(input_dir: Path) -> dict[str, str]:
    """Hash normalized input artifacts for reproducibility checks."""

    return capture_dataset_file_hashes(input_dir)


def build_run_manifest(
    *,
    config: SimulationConfig,
    run_id: str,
    entry_ids: list[str],
    team_ids: list[str],
) -> RunManifest:
    """Capture reproducibility metadata for a simulation run."""

    return RunManifest(
        run_id=run_id,
        created_at=utc_now(),
        code_version=__version__,
        git_commit=read_git_commit(),
        input_dir=config.input_dir,
        dataset_hash=capture_dataset_hash(config.input_dir),
        input_hashes=capture_input_hashes(config.input_dir),
        n_sims=config.n_sims,
        seed=config.seed,
        rating_scale=config.rating_scale,
        batch_size=config.effective_batch_size,
        engine=config.engine,
        log_level=config.log_level,
        entry_ids=entry_ids,
        team_ids=team_ids,
    )


def verify_run_manifest(
    *,
    manifest: RunManifest,
    config: SimulationConfig,
    entry_ids: list[str],
    team_ids: list[str],
) -> None:
    """Reject resume attempts that do not match the captured run manifest."""

    mismatches: list[str] = []
    expected_hashes = capture_input_hashes(config.input_dir)
    current_git_commit = read_git_commit()

    comparisons = {
        "code_version": (manifest.code_version, __version__),
        "git_commit": (manifest.git_commit, current_git_commit),
        "dataset_hash": (manifest.dataset_hash, capture_dataset_hash(config.input_dir)),
        "n_sims": (manifest.n_sims, config.n_sims),
        "seed": (manifest.seed, config.seed),
        "rating_scale": (manifest.rating_scale, config.rating_scale),
        "batch_size": (manifest.batch_size, config.effective_batch_size),
        "engine": (manifest.engine, config.engine),
        "input_hashes": (manifest.input_hashes, expected_hashes),
        "entry_ids": (manifest.entry_ids, entry_ids),
        "team_ids": (manifest.team_ids, team_ids),
    }
    for name, (expected, actual) in comparisons.items():
        if expected != actual:
            mismatches.append(name)

    if mismatches:
        joined = ", ".join(sorted(mismatches))
        msg = f"Run manifest verification failed for: {joined}"
        raise ValueError(msg)


def load_run_manifest(path: Path) -> RunManifest:
    """Read a persisted run manifest from disk."""

    return RunManifest.model_validate(load_json(path))


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    """Persist a run manifest atomically."""

    write_json_atomic(path=path, payload=manifest.model_dump(mode="json"))


def load_run_checkpoint(path: Path) -> RunCheckpoint:
    """Read a persisted run checkpoint from disk."""

    return RunCheckpoint.model_validate(load_json(path))


def write_run_checkpoint(path: Path, checkpoint: RunCheckpoint) -> None:
    """Persist a run checkpoint atomically."""

    write_json_atomic(path=path, payload=checkpoint.model_dump(mode="json"))


def write_simulation_result(path: Path, result: SimulationResult) -> None:
    """Persist a completed simulation result atomically."""

    write_json_atomic(path=path, payload=result.model_dump(mode="json"))


def load_simulation_result(path: Path) -> SimulationResult:
    """Read a previously persisted simulation result from disk."""

    return SimulationResult.model_validate(load_json(path))


def write_json_atomic(*, path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = path.parent / f".{path.name}.tmp-{uuid4().hex}"

    with staging_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    staging_path.replace(path)


def load_json(path: Path) -> object:
    if not path.exists():
        msg = f"Run artifact is missing: {path}"
        raise ValueError(msg)

    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in run artifact {path.name}: {exc.msg}"
            raise ValueError(msg) from exc


def utc_now() -> datetime:
    return datetime.now(UTC)


def read_git_commit() -> str | None:
    project_root = _find_project_root()
    if project_root is None:
        return None

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    commit = result.stdout.strip()
    return commit or None


def _find_project_root() -> Path | None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None
