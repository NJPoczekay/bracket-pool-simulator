from __future__ import annotations

import numpy as np

from bracket_sim.domain.bracket_graph import BracketGraph
from bracket_sim.domain.models import RatingRecord
from bracket_sim.domain.simulator import simulate_tournament
from bracket_sim.infrastructure.storage.normalized_loader import NormalizedInput


def _rating_records_by_team_id(
    normalized_input: NormalizedInput,
    graph: BracketGraph,
) -> dict[str, RatingRecord]:
    records_by_team_id = {
        record.team_id: record for record in normalized_input.ratings.records
    }
    return {team_id: records_by_team_id[team_id] for team_id in graph.teams_by_id}


def test_simulation_is_deterministic_for_same_seed(
    normalized_input: NormalizedInput,
    graph: BracketGraph,
    constraint_map: dict[str, str],
) -> None:
    rating_records_by_team_id = _rating_records_by_team_id(normalized_input, graph)

    first = simulate_tournament(
        graph=graph,
        rating_records_by_team_id=rating_records_by_team_id,
        constraints_by_game_id=constraint_map,
        n_sims=300,
        seed=42,
        point_spread_std_dev=11.0,
    )
    second = simulate_tournament(
        graph=graph,
        rating_records_by_team_id=rating_records_by_team_id,
        constraints_by_game_id=constraint_map,
        n_sims=300,
        seed=42,
        point_spread_std_dev=11.0,
    )

    assert np.array_equal(first.team_wins, second.team_wins)
    assert np.array_equal(first.champions, second.champions)


def test_simulation_changes_for_different_seed(
    normalized_input: NormalizedInput,
    graph: BracketGraph,
    constraint_map: dict[str, str],
) -> None:
    rating_records_by_team_id = _rating_records_by_team_id(normalized_input, graph)

    first = simulate_tournament(
        graph=graph,
        rating_records_by_team_id=rating_records_by_team_id,
        constraints_by_game_id=constraint_map,
        n_sims=300,
        seed=42,
        point_spread_std_dev=11.0,
    )
    second = simulate_tournament(
        graph=graph,
        rating_records_by_team_id=rating_records_by_team_id,
        constraints_by_game_id=constraint_map,
        n_sims=300,
        seed=43,
        point_spread_std_dev=11.0,
    )

    assert not np.array_equal(first.champions, second.champions)


def test_simulation_invariants(
    normalized_input: NormalizedInput,
    graph: BracketGraph,
    constraint_map: dict[str, str],
) -> None:
    rating_records_by_team_id = _rating_records_by_team_id(normalized_input, graph)
    simulation = simulate_tournament(
        graph=graph,
        rating_records_by_team_id=rating_records_by_team_id,
        constraints_by_game_id=constraint_map,
        n_sims=200,
        seed=99,
        point_spread_std_dev=11.0,
    )

    assert simulation.team_wins.shape == (200, 64)
    assert simulation.champions.shape == (200,)
    assert np.all(simulation.team_wins >= 0)
    assert np.all(simulation.team_wins <= 6)
    assert np.all(np.sum(simulation.team_wins, axis=1) == 63)

    champion_indices = simulation.champions.astype(np.intp, copy=False)
    assert np.all(champion_indices >= 0)
    assert np.all(champion_indices < simulation.team_wins.shape[1])

    sim_indices = np.arange(simulation.team_wins.shape[0], dtype=np.intp)
    champion_wins = simulation.team_wins[sim_indices, champion_indices]
    assert np.all(champion_wins == 6)

    champion_mask = np.zeros_like(simulation.team_wins, dtype=bool)
    champion_mask[sim_indices, champion_indices] = True
    assert np.all(simulation.team_wins[~champion_mask] <= 5)
