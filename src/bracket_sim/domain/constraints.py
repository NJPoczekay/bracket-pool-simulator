"""Validation and normalization for completed-game constraints."""

from __future__ import annotations

from bracket_sim.domain.bracket_graph import BracketGraph
from bracket_sim.domain.models import CompletedGameConstraint


def validate_constraints(
    constraints: list[CompletedGameConstraint],
    graph: BracketGraph,
) -> dict[str, str]:
    """Return validated constraint map game_id -> winner_team_id."""

    constraint_map: dict[str, str] = {}
    for constraint in constraints:
        if constraint.game_id not in graph.games_by_id:
            msg = f"Constraint references unknown game: {constraint.game_id}"
            raise ValueError(msg)
        if constraint.game_id in constraint_map:
            msg = f"Duplicate constraint for game: {constraint.game_id}"
            raise ValueError(msg)

        allowed_winners = graph.possible_teams_by_game_id[constraint.game_id]
        if constraint.winner_team_id not in allowed_winners:
            msg = (
                f"Constraint winner {constraint.winner_team_id} is not possible for game "
                f"{constraint.game_id}"
            )
            raise ValueError(msg)

        constraint_map[constraint.game_id] = constraint.winner_team_id

    for game_id, winner_team_id in constraint_map.items():
        game = graph.games_by_id[game_id]
        if game.round == 1:
            continue

        child_game_ids = graph.children_by_game_id[game_id]
        missing_children = [
            child_id for child_id in child_game_ids if child_id not in constraint_map
        ]
        if missing_children:
            msg = (
                f"Constraint for game {game_id} winner {winner_team_id} requires constrained child "
                f"games, missing: {missing_children}"
            )
            raise ValueError(msg)

    _validate_constraint_consistency(constraint_map=constraint_map, graph=graph)
    return constraint_map


def _validate_constraint_consistency(
    constraint_map: dict[str, str],
    graph: BracketGraph,
) -> None:
    """Reject mutually inconsistent constraints across game dependencies."""

    feasible_winners: dict[str, set[str]] = {}
    for game_id in graph.topological_game_ids:
        game = graph.games_by_id[game_id]
        if game.round == 1:
            feasible = set(graph.possible_teams_by_game_id[game_id])
        else:
            left_game_id, right_game_id = graph.children_by_game_id[game_id]
            feasible = feasible_winners[left_game_id] | feasible_winners[right_game_id]

        if game_id in constraint_map:
            winner = constraint_map[game_id]
            if winner not in feasible:
                msg = (
                    f"Constraint winner {winner} for game {game_id} conflicts with upstream "
                    "constraints"
                )
                raise ValueError(msg)
            feasible = {winner}

        feasible_winners[game_id] = feasible
