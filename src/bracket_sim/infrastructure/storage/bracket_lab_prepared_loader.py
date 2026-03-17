"""Load prepared Bracket Lab datasets from disk."""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter

from bracket_sim.domain.bracket_lab_models import CompletionInputs, PlayInSlot, PublicPickRecord
from bracket_sim.domain.models import CompletedGameConstraint, Game, RatingRecord, Team
from bracket_sim.infrastructure.storage._file_io import load_required_csv_rows, load_required_json


@dataclass(frozen=True)
class BracketLabPreparedInput:
    """All prepared Bracket Lab artifacts required by later product phases."""

    teams: list[Team]
    games: list[Game]
    constraints: list[CompletedGameConstraint]
    public_picks: list[PublicPickRecord]
    ratings: list[RatingRecord]
    completion_inputs: CompletionInputs
    play_in_slots: list[PlayInSlot]
    metadata: dict[str, Any]


def load_bracket_lab_prepared_input(input_dir: Path) -> BracketLabPreparedInput:
    """Load and validate a prepared Bracket Lab dataset."""

    if not input_dir.exists() or not input_dir.is_dir():
        msg = f"Input directory does not exist: {input_dir}"
        raise ValueError(msg)

    teams = _load_json_list(input_dir / "teams.json", list[Team])
    games = _load_json_list(input_dir / "games.json", list[Game])
    public_picks = _load_public_picks_csv(input_dir / "public_picks.csv")
    ratings = _load_ratings_csv(input_dir / "ratings.csv")
    completion_inputs_payload = load_required_json(
        input_dir / "completion_inputs.json",
        missing_prefix="Required input file is missing",
    )
    completion_inputs = TypeAdapter(CompletionInputs).validate_python(completion_inputs_payload)

    constraints_path = input_dir / "constraints.json"
    constraints = (
        _load_json_list(constraints_path, list[CompletedGameConstraint])
        if constraints_path.exists()
        else []
    )

    play_in_slots_path = input_dir / "play_in_slots.json"
    play_in_slots = (
        _load_json_list(play_in_slots_path, list[PlayInSlot])
        if play_in_slots_path.exists()
        else []
    )

    metadata_raw = load_required_json(
        input_dir / "metadata.json",
        missing_prefix="Required input file is missing",
    )
    if not isinstance(metadata_raw, dict):
        msg = "metadata.json must contain an object"
        raise ValueError(msg)

    return BracketLabPreparedInput(
        teams=teams,
        games=games,
        constraints=constraints,
        public_picks=public_picks,
        ratings=ratings,
        completion_inputs=completion_inputs,
        play_in_slots=play_in_slots,
        metadata=cast(dict[str, Any], metadata_raw),
    )


def _load_json_list(path: Path, expected_type: type[list[Any]]) -> list[Any]:
    payload = load_required_json(path, missing_prefix="Required input file is missing")
    adapter = TypeAdapter(expected_type)
    return cast(list[Any], adapter.validate_python(payload))


def _load_public_picks_csv(path: Path) -> list[PublicPickRecord]:
    rows, fieldnames = load_required_csv_rows(path, missing_prefix="Required input file is missing")
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

    return [
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


def _load_ratings_csv(path: Path) -> list[RatingRecord]:
    rows, fieldnames = load_required_csv_rows(path, missing_prefix="Required input file is missing")
    expected = {"team_id", "rating", "tempo"}
    if set(fieldnames) != expected:
        msg = f"{path.name} must have columns {sorted(expected)}, got {fieldnames}"
        raise ValueError(msg)

    return [
        RatingRecord(
            team_id=_require_text(row, "team_id", path),
            rating=float(_require_text(row, "rating", path).replace("+", "")),
            tempo=float(_require_text(row, "tempo", path)),
        )
        for row in rows
    ]


def _require_text(row: Mapping[str, object], key: str, path: Path) -> str:
    value = html.unescape(str(row.get(key) or "")).strip()
    if value == "":
        msg = f"{path.name} contains blank value for required field '{key}'"
        raise ValueError(msg)
    return value
