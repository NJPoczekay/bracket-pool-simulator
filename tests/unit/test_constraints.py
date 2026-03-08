from __future__ import annotations

import pytest

from bracket_sim.domain.bracket_graph import BracketGraph
from bracket_sim.domain.constraints import validate_constraints
from bracket_sim.domain.models import CompletedGameConstraint
from bracket_sim.infrastructure.storage.normalized_loader import NormalizedInput


def test_valid_constraints_pass(normalized_input: NormalizedInput, graph: BracketGraph) -> None:
    result = validate_constraints(normalized_input.constraints, graph)
    assert len(result) == len(normalized_input.constraints)


def test_unknown_game_constraint_rejected(graph: BracketGraph) -> None:
    constraints = [CompletedGameConstraint(game_id="missing", winner_team_id="east-01")]
    with pytest.raises(ValueError, match="unknown game"):
        validate_constraints(constraints, graph)


def test_impossible_winner_constraint_rejected(graph: BracketGraph) -> None:
    first_round_game_id = next(
        game_id for game_id, game in graph.games_by_id.items() if game.round == 1
    )
    constraints = [
        CompletedGameConstraint(game_id=first_round_game_id, winner_team_id="west-01")
    ]

    with pytest.raises(ValueError, match="not possible"):
        validate_constraints(constraints, graph)


def test_higher_round_constraint_requires_children(graph: BracketGraph) -> None:
    round_two_game_id = next(
        game_id for game_id, game in graph.games_by_id.items() if game.round == 2
    )
    constraints = [CompletedGameConstraint(game_id=round_two_game_id, winner_team_id="east-01")]

    with pytest.raises(ValueError, match="requires constrained child games"):
        validate_constraints(constraints, graph)
