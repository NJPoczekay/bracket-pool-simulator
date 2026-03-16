"""Shared dataset hash and cache key helpers."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

_DATASET_SUFFIXES = {".csv", ".json", ".parquet"}


def capture_dataset_file_hashes(input_dir: Path) -> dict[str, str]:
    """Hash the canonical top-level dataset files for one prepared input directory."""

    if not input_dir.exists() or not input_dir.is_dir():
        msg = f"Input directory does not exist: {input_dir}"
        raise ValueError(msg)

    hashes: dict[str, str] = {}
    for path in sorted(input_dir.iterdir(), key=lambda candidate: candidate.name):
        if not path.is_file() or path.name.startswith(".") or path.suffix not in _DATASET_SUFFIXES:
            continue
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    if not hashes:
        msg = f"No dataset artifacts found in {input_dir}"
        raise ValueError(msg)

    return hashes


def capture_dataset_hash(input_dir: Path) -> str:
    """Return a deterministic hash for the dataset inputs used by cached artifacts."""

    payload = {"files": capture_dataset_file_hashes(input_dir)}
    return _hash_payload(payload)


def build_cache_key(*, artifact_kind: str, dataset_hash: str, settings: object) -> str:
    """Return a deterministic key for reusable analysis or optimization artifacts."""

    normalized_kind = artifact_kind.strip().lower().replace("_", "-")
    if not normalized_kind:
        msg = "artifact_kind must not be blank"
        raise ValueError(msg)

    payload = {
        "artifact_kind": normalized_kind,
        "dataset_hash": dataset_hash,
        "settings": _normalize_json_payload(settings),
    }
    return f"{normalized_kind}-{_hash_payload(payload)[:16]}"


def _hash_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_json_payload(payload: object) -> object:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    return payload


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    msg = f"Object of type {type(value).__name__} is not JSON serializable"
    raise TypeError(msg)
