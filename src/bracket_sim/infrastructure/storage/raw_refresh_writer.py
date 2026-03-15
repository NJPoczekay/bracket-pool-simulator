"""Atomic writer for refreshed canonical raw datasets and snapshots."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from bracket_sim.infrastructure.providers.contracts import (
    RawAliasRow,
    RawConstraintRow,
    RawEntryRow,
    RawGameRow,
    RawRatingRow,
    RawTeamRow,
)


@dataclass(frozen=True)
class RefreshedRawDataset:
    """Complete canonical raw dataset written by refresh-data."""

    teams: list[RawTeamRow]
    games: list[RawGameRow]
    entries: list[RawEntryRow]
    constraints: list[RawConstraintRow]
    ratings: list[RawRatingRow]
    aliases: list[RawAliasRow]
    metadata: dict[str, Any]
    snapshots: dict[str, dict[str, Any]]


def write_refreshed_raw_dataset(*, out_dir: Path, dataset: RefreshedRawDataset) -> None:
    """Write canonical raw files and snapshots atomically."""

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


def _write_dataset(*, staging_dir: Path, dataset: RefreshedRawDataset) -> None:
    _write_teams_csv(staging_dir / "teams.csv", dataset.teams)
    _write_games_csv(staging_dir / "games.csv", dataset.games)
    _write_entries_json(staging_dir / "entries.json", dataset.entries)
    _write_constraints_json(staging_dir / "constraints.json", dataset.constraints)
    _write_ratings_csv(staging_dir / "ratings.csv", dataset.ratings)

    if dataset.aliases:
        _write_aliases_csv(staging_dir / "aliases.csv", dataset.aliases)

    _write_json(path=staging_dir / "metadata.json", payload=dataset.metadata)

    if dataset.snapshots:
        snapshots_dir = staging_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in sorted(dataset.snapshots.items()):
            _write_json(path=snapshots_dir / f"{name}.json", payload=payload)


def _write_teams_csv(path: Path, rows: list[RawTeamRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["team_id", "name", "seed", "region"])
        for row in sorted(rows, key=lambda item: item.team_id):
            writer.writerow([row.team_id, row.name, row.seed, row.region])


def _write_games_csv(path: Path, rows: list[RawGameRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "game_id",
                "round",
                "left_team_id",
                "right_team_id",
                "left_game_id",
                "right_game_id",
            ]
        )
        for row in sorted(rows, key=lambda item: (item.round, item.game_id)):
            writer.writerow(
                [
                    row.game_id,
                    row.round,
                    row.left_team_id or "",
                    row.right_team_id or "",
                    row.left_game_id or "",
                    row.right_game_id or "",
                ]
            )


def _write_entries_json(path: Path, rows: list[RawEntryRow]) -> None:
    payload: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: item.entry_id):
        payload.append(
            {
                "entry_id": row.entry_id,
                "entry_name": row.entry_name,
                "picks": {game_id: row.picks[game_id] for game_id in sorted(row.picks)},
            }
        )

    _write_json(path=path, payload=payload)


def _write_constraints_json(path: Path, rows: list[RawConstraintRow]) -> None:
    payload = [
        {"game_id": row.game_id, "winner": row.winner}
        for row in sorted(rows, key=lambda item: item.game_id)
    ]
    _write_json(path=path, payload=payload)


def _write_ratings_csv(path: Path, rows: list[RawRatingRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["team", "rating", "tempo"])
        for row in sorted(rows, key=lambda item: item.team):
            writer.writerow([row.team, f"{row.rating:.12g}", f"{row.tempo:.12g}"])


def _write_aliases_csv(path: Path, rows: list[RawAliasRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alias", "team_id"])
        for row in sorted(rows, key=lambda item: item.alias.casefold()):
            writer.writerow([row.alias, row.team_id])


def _write_json(*, path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
