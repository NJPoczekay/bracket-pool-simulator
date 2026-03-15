from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from bracket_sim.domain.bracket_graph import BracketGraph, build_bracket_graph
from bracket_sim.domain.constraints import validate_constraints
from bracket_sim.infrastructure.storage.normalized_loader import (
    NormalizedInput,
    load_normalized_input,
)


@pytest.fixture(scope="session")
def synthetic_input_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "synthetic_64"


@pytest.fixture(scope="session")
def raw_canonical_dir(
    synthetic_input_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    raw_dir = tmp_path_factory.mktemp("raw_canonical")

    teams = json.loads((synthetic_input_dir / "teams.json").read_text(encoding="utf-8"))
    games = json.loads((synthetic_input_dir / "games.json").read_text(encoding="utf-8"))
    entries = json.loads((synthetic_input_dir / "entries.json").read_text(encoding="utf-8"))
    constraints = json.loads((synthetic_input_dir / "constraints.json").read_text(encoding="utf-8"))

    with (raw_dir / "teams.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["team_id", "name", "seed", "region"])
        for team in teams:
            writer.writerow([team["team_id"], team["name"], team["seed"], team["region"]])

    with (raw_dir / "games.csv").open("w", encoding="utf-8", newline="") as handle:
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
        for game in games:
            writer.writerow(
                [
                    game["game_id"],
                    game["round"],
                    game["left_team_id"] or "",
                    game["right_team_id"] or "",
                    game["left_game_id"] or "",
                    game["right_game_id"] or "",
                ]
            )

    (raw_dir / "entries.json").write_text(
        json.dumps(entries, indent=2) + "\n",
        encoding="utf-8",
    )

    raw_constraints = [
        {"game_id": row["game_id"], "winner": row["winner_team_id"]}
        for row in constraints
    ]
    (raw_dir / "constraints.json").write_text(
        json.dumps(raw_constraints, indent=2) + "\n",
        encoding="utf-8",
    )

    ratings_text = (synthetic_input_dir / "ratings.csv").read_text(encoding="utf-8")
    (raw_dir / "ratings.csv").write_text(ratings_text, encoding="utf-8")

    return raw_dir


@pytest.fixture(scope="session")
def normalized_input(synthetic_input_dir: Path) -> NormalizedInput:
    return load_normalized_input(synthetic_input_dir)


@pytest.fixture(scope="session")
def graph(normalized_input: NormalizedInput) -> BracketGraph:
    return build_bracket_graph(teams=normalized_input.teams, games=normalized_input.games)


@pytest.fixture(scope="session")
def constraint_map(normalized_input: NormalizedInput, graph: BracketGraph) -> dict[str, str]:
    return validate_constraints(constraints=normalized_input.constraints, graph=graph)
