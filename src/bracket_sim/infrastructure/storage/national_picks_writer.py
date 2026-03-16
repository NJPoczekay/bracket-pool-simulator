"""Atomic writer for national picks snapshots."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from bracket_sim.infrastructure.providers.contracts import RawNationalPickRow


@dataclass(frozen=True)
class NationalPicksDataset:
    """Complete national-picks dataset written by refresh-national-picks."""

    rows: list[RawNationalPickRow]
    metadata: dict[str, Any]
    snapshots: dict[str, dict[str, Any]]


def write_national_picks_dataset(*, out_dir: Path, dataset: NationalPicksDataset) -> None:
    """Write national-picks artifacts atomically."""

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = out_dir.parent / f".{out_dir.name}.tmp-{uuid4().hex}"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
        _write_dataset(staging_dir=staging_dir, dataset=dataset)

        if out_dir.exists():
            shutil.rmtree(out_dir)
        staging_dir.rename(out_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def _write_dataset(*, staging_dir: Path, dataset: NationalPicksDataset) -> None:
    _write_national_picks_csv(staging_dir / "national_picks.csv", dataset.rows)
    _write_json(path=staging_dir / "metadata.json", payload=dataset.metadata)

    if dataset.snapshots:
        snapshots_dir = staging_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in sorted(dataset.snapshots.items()):
            _write_json(path=snapshots_dir / f"{name}.json", payload=payload)


def _write_national_picks_csv(path: Path, rows: list[RawNationalPickRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "game_id",
                "round",
                "display_order",
                "outcome_id",
                "team_id",
                "team_name",
                "seed",
                "region",
                "matchup_position",
                "pick_count",
                "pick_percentage",
            ]
        )
        for row in sorted(
            rows,
            key=lambda item: (
                item.round,
                item.display_order,
                item.matchup_position,
                item.outcome_id,
            ),
        ):
            writer.writerow(
                [
                    row.game_id,
                    row.round,
                    row.display_order,
                    row.outcome_id,
                    row.team_id,
                    row.team_name,
                    row.seed,
                    row.region,
                    row.matchup_position,
                    row.pick_count,
                    f"{row.pick_percentage:.12g}",
                ]
            )


def _write_json(*, path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
