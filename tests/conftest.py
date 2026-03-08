from __future__ import annotations

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
def normalized_input(synthetic_input_dir: Path) -> NormalizedInput:
    return load_normalized_input(synthetic_input_dir)


@pytest.fixture(scope="session")
def graph(normalized_input: NormalizedInput) -> BracketGraph:
    return build_bracket_graph(teams=normalized_input.teams, games=normalized_input.games)


@pytest.fixture(scope="session")
def constraint_map(normalized_input: NormalizedInput, graph: BracketGraph) -> dict[str, str]:
    return validate_constraints(constraints=normalized_input.constraints, graph=graph)
