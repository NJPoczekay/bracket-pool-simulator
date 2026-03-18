"""Load canonical raw local files for data preparation."""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from bracket_sim.domain.models import Game, Team
from bracket_sim.infrastructure.storage._file_io import load_required_csv_rows, load_required_json


@dataclass(frozen=True)
class RawEntry:
    """Raw entry document with picks encoded as game_id -> winner strings."""

    entry_id: str
    entry_name: str
    picks: dict[str, str]


@dataclass(frozen=True)
class RawConstraint:
    """Raw completed-game winner constraint."""

    game_id: str
    winner: str


@dataclass(frozen=True)
class RawAlias:
    """Alias row mapping an external alias to a canonical team id."""

    alias: str
    team_id: str


@dataclass(frozen=True)
class RawRatingRecord:
    """Raw team rating row keyed by alias-like team text."""

    team: str
    rating: float
    tempo: float


@dataclass(frozen=True)
class RawInput:
    """Complete raw input bundle used by the prepare-data pipeline."""

    teams: list[Team]
    games: list[Game]
    entries: list[RawEntry]
    constraints: list[RawConstraint]
    ratings: list[RawRatingRecord]
    aliases: list[RawAlias]
    metadata: dict[str, Any] | None


def load_raw_input(raw_dir: Path) -> RawInput:
    """Load and validate canonical raw files from disk."""

    if not raw_dir.exists() or not raw_dir.is_dir():
        msg = f"Raw input directory does not exist: {raw_dir}"
        raise ValueError(msg)

    teams = _load_teams_csv(raw_dir / "teams.csv")
    games = _load_games_csv(raw_dir / "games.csv")
    entries = _load_entries_json(raw_dir / "entries.json")
    ratings = _load_ratings_csv(raw_dir / "ratings.csv")

    constraints_path = raw_dir / "constraints.json"
    constraints = _load_constraints_json(constraints_path) if constraints_path.exists() else []

    aliases_path = raw_dir / "aliases.csv"
    aliases = _load_aliases_csv(aliases_path) if aliases_path.exists() else []

    metadata_path = raw_dir / "metadata.json"
    metadata = _load_metadata(metadata_path) if metadata_path.exists() else None

    return RawInput(
        teams=teams,
        games=games,
        entries=entries,
        constraints=constraints,
        ratings=ratings,
        aliases=aliases,
        metadata=metadata,
    )


def _load_teams_csv(path: Path) -> list[Team]:
    rows, fieldnames = load_required_csv_rows(path, missing_prefix="Required raw file is missing")
    expected = {"team_id", "name", "seed", "region"}
    _validate_columns(path=path, expected_columns=expected, fieldnames=fieldnames)

    teams: list[Team] = []
    for row in rows:
        teams.append(
            Team(
                team_id=_require_non_empty(row, "team_id", path),
                name=_require_non_empty(row, "name", path),
                seed=int(_require_non_empty(row, "seed", path)),
                region=_require_non_empty(row, "region", path),
            )
        )

    if not teams:
        msg = f"{path.name} must contain at least one team row"
        raise ValueError(msg)

    return teams


def _load_games_csv(path: Path) -> list[Game]:
    rows, fieldnames = load_required_csv_rows(path, missing_prefix="Required raw file is missing")
    expected = {
        "game_id",
        "round",
        "left_team_id",
        "right_team_id",
        "left_game_id",
        "right_game_id",
    }
    _validate_columns(path=path, expected_columns=expected, fieldnames=fieldnames)

    games: list[Game] = []
    for row in rows:
        games.append(
            Game(
                game_id=_require_non_empty(row, "game_id", path),
                round=int(_require_non_empty(row, "round", path)),
                left_team_id=_optional_text(row.get("left_team_id")),
                right_team_id=_optional_text(row.get("right_team_id")),
                left_game_id=_optional_text(row.get("left_game_id")),
                right_game_id=_optional_text(row.get("right_game_id")),
            )
        )

    if not games:
        msg = f"{path.name} must contain at least one game row"
        raise ValueError(msg)

    return games


def _load_entries_json(path: Path) -> list[RawEntry]:
    payload = load_required_json(path, missing_prefix="Required raw file is missing")
    if not isinstance(payload, list):
        msg = "entries.json must contain a list"
        raise ValueError(msg)

    entries: list[RawEntry] = []
    for idx, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            msg = f"entries.json row {idx} must be an object"
            raise ValueError(msg)

        entry_id = _require_non_empty(row, "entry_id", path)
        entry_name = _require_non_empty(row, "entry_name", path)

        picks_raw = row.get("picks")
        if not isinstance(picks_raw, dict):
            msg = f"entries.json row {idx} must include an object field 'picks'"
            raise ValueError(msg)

        picks: dict[str, str] = {}
        for game_id_raw, winner_raw in picks_raw.items():
            game_id = _sanitize_text(game_id_raw)
            winner = _sanitize_text(winner_raw)
            if not game_id:
                msg = f"entries.json row {idx} has blank game id in picks"
                raise ValueError(msg)
            if not winner:
                msg = f"entries.json row {idx} has blank winner for game {game_id}"
                raise ValueError(msg)
            picks[game_id] = winner

        entries.append(RawEntry(entry_id=entry_id, entry_name=entry_name, picks=picks))

    if not entries:
        msg = "entries.json must contain at least one entry"
        raise ValueError(msg)

    return entries


def _load_constraints_json(path: Path) -> list[RawConstraint]:
    payload = load_required_json(path, missing_prefix="Required raw file is missing")
    if not isinstance(payload, list):
        msg = "constraints.json must contain a list"
        raise ValueError(msg)

    constraints: list[RawConstraint] = []
    for idx, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            msg = f"constraints.json row {idx} must be an object"
            raise ValueError(msg)

        constraints.append(
            RawConstraint(
                game_id=_require_non_empty(row, "game_id", path),
                winner=_require_non_empty(row, "winner", path),
            )
        )

    return constraints


def _load_ratings_csv(path: Path) -> list[RawRatingRecord]:
    rows, fieldnames = load_required_csv_rows(path, missing_prefix="Required raw file is missing")
    expected = {"team", "rating", "tempo"}
    _validate_columns(path=path, expected_columns=expected, fieldnames=fieldnames)

    ratings: list[RawRatingRecord] = []
    for row in rows:
        team = _require_non_empty(row, "team", path)
        rating_raw = _require_non_empty(row, "rating", path).replace("+", "")
        tempo_raw = _require_non_empty(row, "tempo", path)
        ratings.append(RawRatingRecord(team=team, rating=float(rating_raw), tempo=float(tempo_raw)))

    if not ratings:
        msg = "ratings.csv must contain at least one rating row"
        raise ValueError(msg)

    return ratings


def _load_aliases_csv(path: Path) -> list[RawAlias]:
    rows, fieldnames = load_required_csv_rows(path, missing_prefix="Required raw file is missing")
    expected = {"alias", "team_id"}
    _validate_columns(path=path, expected_columns=expected, fieldnames=fieldnames)

    aliases: list[RawAlias] = []
    for row in rows:
        aliases.append(
            RawAlias(
                alias=_require_non_empty(row, "alias", path),
                team_id=_require_non_empty(row, "team_id", path),
            )
        )
    return aliases


def _load_metadata(path: Path) -> dict[str, Any]:
    payload = load_required_json(path, missing_prefix="Required raw file is missing")
    if not isinstance(payload, dict):
        msg = "metadata.json must contain an object"
        raise ValueError(msg)
    return cast(dict[str, Any], payload)


def _validate_columns(path: Path, expected_columns: set[str], fieldnames: list[str]) -> None:
    if set(fieldnames) != expected_columns:
        msg = (
            f"{path.name} must have columns {sorted(expected_columns)}, "
            f"got {fieldnames}"
        )
        raise ValueError(msg)


def _require_non_empty(row: Mapping[str, object], key: str, path: Path) -> str:
    value = _sanitize_text(row.get(key))
    if not value:
        msg = f"{path.name} contains blank value for required field '{key}'"
        raise ValueError(msg)
    return value


def _optional_text(value: object) -> str | None:
    sanitized = _sanitize_text(value)
    if sanitized == "":
        return None
    return sanitized


def _sanitize_text(value: object) -> str:
    return html.unescape(str(value or "")).strip()
