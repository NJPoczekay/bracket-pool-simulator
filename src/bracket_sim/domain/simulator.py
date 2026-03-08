"""Deterministic Monte Carlo tournament simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bracket_sim.domain.bracket_graph import BracketGraph
from bracket_sim.domain.probability_model import logistic_win_probability


@dataclass(frozen=True)
class TournamentSimulation:
    """Vectorized tournament simulation outputs."""

    team_ids: list[str]
    team_wins: npt.NDArray[np.int16]
    champions: npt.NDArray[np.int16]


def canonical_team_order(graph: BracketGraph) -> list[str]:
    """Return deterministic team ordering used by simulation arrays."""

    return sorted(graph.teams_by_id)


def simulate_tournament(
    graph: BracketGraph,
    ratings_by_team_id: dict[str, float],
    constraints_by_game_id: dict[str, str],
    n_sims: int,
    seed: int,
    rating_scale: float,
) -> TournamentSimulation:
    """Simulate bracket outcomes in a deterministic, seed-controlled manner."""

    team_ids = canonical_team_order(graph)
    team_index = {team_id: idx for idx, team_id in enumerate(team_ids)}

    ratings = np.zeros(len(team_ids), dtype=np.float64)
    for team_id, idx in team_index.items():
        if team_id not in ratings_by_team_id:
            msg = f"Missing rating for team id: {team_id}"
            raise ValueError(msg)
        ratings[idx] = ratings_by_team_id[team_id]

    rng = np.random.default_rng(seed)

    team_wins = np.zeros((n_sims, len(team_ids)), dtype=np.int16)
    winners_by_game_id: dict[str, npt.NDArray[np.int16]] = {}
    simulation_index = np.arange(n_sims)

    for game_id in graph.topological_game_ids:
        game = graph.games_by_id[game_id]

        if game.round == 1:
            assert game.left_team_id is not None
            assert game.right_team_id is not None
            left_winners = np.full(n_sims, team_index[game.left_team_id], dtype=np.int16)
            right_winners = np.full(n_sims, team_index[game.right_team_id], dtype=np.int16)
        else:
            child_left_id, child_right_id = graph.children_by_game_id[game_id]
            left_winners = winners_by_game_id[child_left_id]
            right_winners = winners_by_game_id[child_right_id]

        if game_id in constraints_by_game_id:
            winner_team_id = constraints_by_game_id[game_id]
            winner_idx = team_index[winner_team_id]
            winners = np.full(n_sims, winner_idx, dtype=np.int16)
        else:
            left_ratings = ratings[left_winners]
            right_ratings = ratings[right_winners]
            left_probs = logistic_win_probability(
                left_ratings=left_ratings,
                right_ratings=right_ratings,
                rating_scale=rating_scale,
            )
            draws = rng.random(n_sims)
            winners = np.where(draws < left_probs, left_winners, right_winners).astype(np.int16)

        winners_by_game_id[game_id] = winners
        team_wins[simulation_index, winners] += 1

    champions = winners_by_game_id[graph.championship_game_id]
    return TournamentSimulation(team_ids=team_ids, team_wins=team_wins, champions=champions)
