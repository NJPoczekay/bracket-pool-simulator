from __future__ import annotations

import numpy as np

from bracket_sim.domain.bracket_graph import BracketGraph
from bracket_sim.domain.simulator import simulate_tournament
from bracket_sim.infrastructure.storage.normalized_loader import NormalizedInput


def _ratings_by_team_id(
    normalized_input: NormalizedInput,
    graph: BracketGraph,
) -> dict[str, float]:
    ratings_by_name = {record.team: record.rating for record in normalized_input.ratings.records}
    return {team_id: ratings_by_name[team.name] for team_id, team in graph.teams_by_id.items()}


def test_simulation_is_deterministic_for_same_seed(
    normalized_input: NormalizedInput,
    graph: BracketGraph,
    constraint_map: dict[str, str],
) -> None:
    ratings_by_team_id = _ratings_by_team_id(normalized_input, graph)

    first = simulate_tournament(
        graph=graph,
        ratings_by_team_id=ratings_by_team_id,
        constraints_by_game_id=constraint_map,
        n_sims=300,
        seed=42,
        rating_scale=10.0,
    )
    second = simulate_tournament(
        graph=graph,
        ratings_by_team_id=ratings_by_team_id,
        constraints_by_game_id=constraint_map,
        n_sims=300,
        seed=42,
        rating_scale=10.0,
    )

    assert np.array_equal(first.team_wins, second.team_wins)
    assert np.array_equal(first.champions, second.champions)


def test_simulation_changes_for_different_seed(
    normalized_input: NormalizedInput,
    graph: BracketGraph,
    constraint_map: dict[str, str],
) -> None:
    ratings_by_team_id = _ratings_by_team_id(normalized_input, graph)

    first = simulate_tournament(
        graph=graph,
        ratings_by_team_id=ratings_by_team_id,
        constraints_by_game_id=constraint_map,
        n_sims=300,
        seed=42,
        rating_scale=10.0,
    )
    second = simulate_tournament(
        graph=graph,
        ratings_by_team_id=ratings_by_team_id,
        constraints_by_game_id=constraint_map,
        n_sims=300,
        seed=43,
        rating_scale=10.0,
    )

    assert not np.array_equal(first.champions, second.champions)


def test_simulation_invariants(
    normalized_input: NormalizedInput,
    graph: BracketGraph,
    constraint_map: dict[str, str],
) -> None:
    ratings_by_team_id = _ratings_by_team_id(normalized_input, graph)
    simulation = simulate_tournament(
        graph=graph,
        ratings_by_team_id=ratings_by_team_id,
        constraints_by_game_id=constraint_map,
        n_sims=200,
        seed=99,
        rating_scale=10.0,
    )

    assert simulation.team_wins.shape == (200, 64)
    assert simulation.champions.shape == (200,)
    assert np.all(simulation.team_wins >= 0)
    assert np.all(simulation.team_wins <= 6)
    assert np.all(np.sum(simulation.team_wins, axis=1) == 63)
