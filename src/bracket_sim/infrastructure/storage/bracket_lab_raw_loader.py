"""Load raw Bracket Lab datasets from disk."""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter

from bracket_sim.domain.bracket_lab_models import PublicPickRecord
from bracket_sim.domain.models import Game, Team
from bracket_sim.infrastructure.providers.contracts import RawRatingRow
from bracket_sim.infrastructure.storage._file_io import load_required_csv_rows, load_required_json
from bracket_sim.infrastructure.storage.raw_loader import RawAlias, RawConstraint


@dataclass(frozen=True)
class BracketLabRawInput:
    """Complete raw input bundle used by Bracket Lab preparation."""

    teams: list[Team]
    games: list[Game]
    constraints: list[RawConstraint]
    public_picks: list[PublicPickRecord]
    kenpom_rows: list[RawRatingRow]
    aliases: list[RawAlias]
    metadata: dict[str, Any]


def load_bracket_lab_raw_input(raw_dir: Path) -> BracketLabRawInput:
    """Load and validate raw Bracket Lab files from disk."""

    if not raw_dir.exists() or not raw_dir.is_dir():
        msg = f"Raw input directory does not exist: {raw_dir}"
        raise ValueError(msg)

    teams = _load_jsonless_csv(raw_dir / "teams.csv", list[Team], "Required raw file is missing")
    games = _load_jsonless_csv(raw_dir / "games.csv", list[Game], "Required raw file is missing")
    public_picks = _load_public_picks_csv(raw_dir / "national_picks.csv")
    kenpom_rows = _load_kenpom_csv(raw_dir / "kenpom.csv")

    constraints_path = raw_dir / "constraints.json"
    constraints = _load_constraints_json(constraints_path) if constraints_path.exists() else []

    aliases_path = raw_dir / "aliases.csv"
    aliases = _load_aliases_csv(aliases_path) if aliases_path.exists() else []

    metadata_raw = load_required_json(
        raw_dir / "metadata.json",
        missing_prefix="Required raw file is missing",
    )
    if not isinstance(metadata_raw, dict):
        msg = "metadata.json must contain an object"
        raise ValueError(msg)

    return BracketLabRawInput(
        teams=teams,
        games=games,
        constraints=constraints,
        public_picks=public_picks,
        kenpom_rows=kenpom_rows,
        aliases=aliases,
        metadata=cast(dict[str, Any], metadata_raw),
    )


def _load_jsonless_csv(
    path: Path,
    expected_type: type[list[Any]],
    missing_prefix: str,
) -> list[Any]:
    if path.name == "teams.csv":
        rows, fieldnames = load_required_csv_rows(path, missing_prefix=missing_prefix)
        if set(fieldnames) != {"team_id", "name", "seed", "region"}:
            msg = (
                f"{path.name} must have columns ['name', 'region', 'seed', 'team_id'], "
                f"got {fieldnames}"
            )
            raise ValueError(msg)
        payload = [
            {
                "team_id": _require_text(row, "team_id", path),
                "name": _require_text(row, "name", path),
                "seed": int(_require_text(row, "seed", path)),
                "region": _require_text(row, "region", path),
            }
            for row in rows
        ]
    else:
        rows, fieldnames = load_required_csv_rows(path, missing_prefix=missing_prefix)
        expected = {
            "game_id",
            "round",
            "left_team_id",
            "right_team_id",
            "left_game_id",
            "right_game_id",
        }
        if set(fieldnames) != expected:
            msg = f"{path.name} must have columns {sorted(expected)}, got {fieldnames}"
            raise ValueError(msg)
        payload = [
            {
                "game_id": _require_text(row, "game_id", path),
                "round": int(_require_text(row, "round", path)),
                "left_team_id": _optional_text(row.get("left_team_id")),
                "right_team_id": _optional_text(row.get("right_team_id")),
                "left_game_id": _optional_text(row.get("left_game_id")),
                "right_game_id": _optional_text(row.get("right_game_id")),
            }
            for row in rows
        ]

    adapter = TypeAdapter(expected_type)
    return cast(list[Any], adapter.validate_python(payload))


def _load_public_picks_csv(path: Path) -> list[PublicPickRecord]:
    rows, fieldnames = load_required_csv_rows(path, missing_prefix="Required raw file is missing")
    expected = {
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
    }
    if set(fieldnames) != expected:
        msg = f"{path.name} must have columns {sorted(expected)}, got {fieldnames}"
        raise ValueError(msg)

    payload = [
        PublicPickRecord(
            game_id=_require_text(row, "game_id", path),
            round=int(_require_text(row, "round", path)),
            display_order=int(_require_text(row, "display_order", path)),
            outcome_id=_require_text(row, "outcome_id", path),
            team_id=_require_text(row, "team_id", path),
            team_name=_require_text(row, "team_name", path),
            seed=int(_require_text(row, "seed", path)),
            region=_require_text(row, "region", path),
            matchup_position=int(_require_text(row, "matchup_position", path)),
            pick_count=int(_require_text(row, "pick_count", path)),
            pick_percentage=float(_require_text(row, "pick_percentage", path)),
        )
        for row in rows
    ]
    return payload


def _load_kenpom_csv(path: Path) -> list[RawRatingRow]:
    rows, fieldnames = load_required_csv_rows(path, missing_prefix="Required raw file is missing")
    expected = {"team", "rating", "tempo"}
    if set(fieldnames) != expected:
        msg = f"{path.name} must have columns {sorted(expected)}, got {fieldnames}"
        raise ValueError(msg)

    payload = [
        RawRatingRow(
            team=_require_text(row, "team", path),
            rating=float(_require_text(row, "rating", path).replace("+", "")),
            tempo=float(_require_text(row, "tempo", path)),
        )
        for row in rows
    ]
    return payload


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
                game_id=_require_text(row, "game_id", path),
                winner=_require_text(row, "winner", path),
            )
        )
    return constraints


def _load_aliases_csv(path: Path) -> list[RawAlias]:
    rows, fieldnames = load_required_csv_rows(path, missing_prefix="Required raw file is missing")
    expected = {"alias", "team_id"}
    if set(fieldnames) != expected:
        msg = f"{path.name} must have columns {sorted(expected)}, got {fieldnames}"
        raise ValueError(msg)
    return [
        RawAlias(
            alias=_require_text(row, "alias", path),
            team_id=_require_text(row, "team_id", path),
        )
        for row in rows
    ]


def _require_text(row: Mapping[str, object], key: str, path: Path) -> str:
    value = _optional_text(row.get(key))
    if value is None:
        msg = f"{path.name} contains blank value for required field '{key}'"
        raise ValueError(msg)
    return value


def _optional_text(value: object) -> str | None:
    normalized = html.unescape(str(value or "")).strip()
    if normalized == "":
        return None
    return normalized
