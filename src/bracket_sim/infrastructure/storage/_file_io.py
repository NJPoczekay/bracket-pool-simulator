"""Shared file loading helpers for storage adapters."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_required_json(
    path: Path,
    *,
    missing_prefix: str = "Required file is missing",
) -> object:
    """Load a required JSON file from disk."""

    if not path.exists():
        msg = f"{missing_prefix}: {path.name}"
        raise ValueError(msg)

    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {path.name}: {exc.msg}"
            raise ValueError(msg) from exc


def load_required_csv_rows(
    path: Path,
    *,
    missing_prefix: str = "Required file is missing",
) -> tuple[list[dict[str, str]], list[str]]:
    """Load a required CSV file as dict rows plus field names."""

    if not path.exists():
        msg = f"{missing_prefix}: {path.name}"
        raise ValueError(msg)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    return rows, fieldnames
