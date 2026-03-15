"""Load normalized local input files for simulation."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter

from bracket_sim.domain.models import (
    CompletedGameConstraint,
    EntryPick,
    Game,
    PoolEntry,
    RatingRecord,
    RatingSnapshot,
    Team,
)
from bracket_sim.infrastructure.storage._file_io import load_required_csv_rows, load_required_json


@dataclass(frozen=True)
class NormalizedInput:
    """All normalized datasets required for a simulation run."""

    teams: list[Team]
    games: list[Game]
    entries: list[PoolEntry]
    constraints: list[CompletedGameConstraint]
    ratings: RatingSnapshot


def load_normalized_input(input_dir: Path) -> NormalizedInput:
    """Load and parse normalized input directory files."""

    if not input_dir.exists() or not input_dir.is_dir():
        msg = f"Input directory does not exist: {input_dir}"
        raise ValueError(msg)

    teams = _load_json_list(input_dir / "teams.json", list[Team])
    games = _load_json_list(input_dir / "games.json", list[Game])
    entries = _load_entries(input_dir / "entries.json")

    constraints_path = input_dir / "constraints.json"
    if constraints_path.exists():
        constraints = _load_json_list(constraints_path, list[CompletedGameConstraint])
    else:
        constraints = []

    ratings = RatingSnapshot(records=_load_ratings_csv(input_dir / "ratings.csv"))

    return NormalizedInput(
        teams=teams,
        games=games,
        entries=entries,
        constraints=constraints,
        ratings=ratings,
    )


def _load_json_list(path: Path, expected_type: type[list[Any]]) -> list[Any]:
    """Read JSON list from disk and validate with pydantic adapters."""

    payload = load_required_json(path, missing_prefix="Required input file is missing")

    adapter = TypeAdapter(expected_type)
    return cast(list[Any], adapter.validate_python(payload))


def _load_entries(path: Path) -> list[PoolEntry]:
    """Load entries where picks are encoded as game_id -> winner_team_id maps."""

    payload = load_required_json(path, missing_prefix="Required input file is missing")

    if not isinstance(payload, list):
        msg = "entries.json must contain a list"
        raise ValueError(msg)

    entries: list[PoolEntry] = []
    for row in payload:
        if not isinstance(row, dict):
            msg = "Each entries.json row must be an object"
            raise ValueError(msg)

        picks_raw = row.get("picks")
        if not isinstance(picks_raw, dict):
            msg = "Each entries.json row must include a picks object"
            raise ValueError(msg)

        picks = [
            EntryPick(game_id=str(game_id), winner_team_id=str(winner_team_id))
            for game_id, winner_team_id in sorted(picks_raw.items())
        ]

        entries.append(
            PoolEntry(
                entry_id=str(row.get("entry_id", "")),
                entry_name=str(row.get("entry_name", "")),
                picks=picks,
            )
        )

    return entries


def _load_ratings_csv(path: Path) -> list[RatingRecord]:
    """Load ratings.csv into validated rating records."""

    rows, fieldnames = load_required_csv_rows(path, missing_prefix="Required input file is missing")

    expected_columns = {"team_id", "rating", "tempo"}
    if set(fieldnames) != expected_columns:
        msg = (
            f"ratings.csv must have columns {sorted(expected_columns)}, got "
            f"{fieldnames}"
        )
        raise ValueError(msg)

    records: list[RatingRecord] = []
    for row in rows:
        team_id = html.unescape(str(row["team_id"]).strip())
        records.append(
            RatingRecord(
                team_id=team_id,
                rating=float(str(row["rating"]).replace("+", "")),
                tempo=float(row["tempo"]),
            )
        )

    return records
