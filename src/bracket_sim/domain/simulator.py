"""Deterministic Monte Carlo tournament simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from bracket_sim.domain.bracket_graph import BracketGraph
from bracket_sim.domain.probability_model import logistic_win_probability

try:
    from numba import njit  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised via feature-flag tests
    njit = None


@dataclass(frozen=True)
class TournamentSimulation:
    """Vectorized tournament simulation outputs."""

    team_ids: list[str]
    team_wins: npt.NDArray[np.int16]
    champions: npt.NDArray[np.int16]


@dataclass(frozen=True)
class CompiledTournament:
    """Array-friendly tournament topology used by both engines."""

    round1_mask: npt.NDArray[np.bool_]
    left_refs: npt.NDArray[np.int16]
    right_refs: npt.NDArray[np.int16]
    constrained_winners: npt.NDArray[np.int16]
    championship_game_index: int


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
    engine: str = "numpy",
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

    compiled = compile_tournament_arrays(
        graph=graph,
        team_index=team_index,
        constraints_by_game_id=constraints_by_game_id,
    )

    if engine == "numpy":
        team_wins, champions = _simulate_tournament_numpy(
            compiled=compiled,
            ratings=ratings,
            n_sims=n_sims,
            seed=seed,
            rating_scale=rating_scale,
        )
    elif engine == "numba":
        team_wins, champions = _simulate_tournament_numba(
            compiled=compiled,
            ratings=ratings,
            n_sims=n_sims,
            seed=seed,
            rating_scale=rating_scale,
        )
    else:
        msg = f"Unsupported simulation engine: {engine}"
        raise ValueError(msg)

    return TournamentSimulation(team_ids=team_ids, team_wins=team_wins, champions=champions)


def compile_tournament_arrays(
    *,
    graph: BracketGraph,
    team_index: dict[str, int],
    constraints_by_game_id: dict[str, str],
) -> CompiledTournament:
    """Compile graph topology into array references shared by execution engines."""

    game_index = {game_id: idx for idx, game_id in enumerate(graph.topological_game_ids)}
    n_games = len(graph.topological_game_ids)

    round1_mask = np.zeros(n_games, dtype=np.bool_)
    left_refs = np.zeros(n_games, dtype=np.int16)
    right_refs = np.zeros(n_games, dtype=np.int16)
    constrained_winners = np.full(n_games, -1, dtype=np.int16)

    for game_idx, game_id in enumerate(graph.topological_game_ids):
        game = graph.games_by_id[game_id]
        if game.round == 1:
            assert game.left_team_id is not None
            assert game.right_team_id is not None
            round1_mask[game_idx] = True
            left_refs[game_idx] = team_index[game.left_team_id]
            right_refs[game_idx] = team_index[game.right_team_id]
        else:
            child_left_id, child_right_id = graph.children_by_game_id[game_id]
            left_refs[game_idx] = game_index[child_left_id]
            right_refs[game_idx] = game_index[child_right_id]

        if game_id in constraints_by_game_id:
            constrained_winners[game_idx] = team_index[constraints_by_game_id[game_id]]

    return CompiledTournament(
        round1_mask=round1_mask,
        left_refs=left_refs,
        right_refs=right_refs,
        constrained_winners=constrained_winners,
        championship_game_index=game_index[graph.championship_game_id],
    )


def _simulate_tournament_numpy(
    *,
    compiled: CompiledTournament,
    ratings: npt.NDArray[np.float64],
    n_sims: int,
    seed: int,
    rating_scale: float,
) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
    rng = np.random.default_rng(seed)

    team_wins = np.zeros((n_sims, len(ratings)), dtype=np.int16)
    winners_by_game = np.zeros((len(compiled.left_refs), n_sims), dtype=np.int16)
    simulation_index = np.arange(n_sims)

    for game_idx in range(len(compiled.left_refs)):
        if compiled.round1_mask[game_idx]:
            left_winners = np.full(n_sims, compiled.left_refs[game_idx], dtype=np.int16)
            right_winners = np.full(n_sims, compiled.right_refs[game_idx], dtype=np.int16)
        else:
            left_winners = winners_by_game[compiled.left_refs[game_idx]]
            right_winners = winners_by_game[compiled.right_refs[game_idx]]

        if compiled.constrained_winners[game_idx] >= 0:
            winners = np.full(n_sims, compiled.constrained_winners[game_idx], dtype=np.int16)
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

        winners_by_game[game_idx] = winners
        team_wins[simulation_index, winners] += 1

    champions = winners_by_game[compiled.championship_game_index].copy()
    return team_wins, champions


def _simulate_tournament_numba(
    *,
    compiled: CompiledTournament,
    ratings: npt.NDArray[np.float64],
    n_sims: int,
    seed: int,
    rating_scale: float,
) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
    if njit is None:
        msg = "Numba engine requested but numba is not installed"
        raise ValueError(msg)

    return cast(
        tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]],
        _simulate_tournament_numba_core(
            ratings,
            compiled.round1_mask,
            compiled.left_refs,
            compiled.right_refs,
            compiled.constrained_winners,
            compiled.championship_game_index,
            n_sims,
            seed,
            rating_scale,
        ),
    )


if njit is not None:

    @njit(cache=True)  # type: ignore[untyped-decorator]
    def _simulate_tournament_numba_core(
        ratings: npt.NDArray[np.float64],
        round1_mask: npt.NDArray[np.bool_],
        left_refs: npt.NDArray[np.int16],
        right_refs: npt.NDArray[np.int16],
        constrained_winners: npt.NDArray[np.int16],
        championship_game_index: int,
        n_sims: int,
        seed: int,
        rating_scale: float,
    ) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
        np.random.seed(seed)
        team_wins = np.zeros((n_sims, len(ratings)), dtype=np.int16)
        winners_by_game = np.zeros((len(left_refs), n_sims), dtype=np.int16)

        for game_idx in range(len(left_refs)):
            for sim_idx in range(n_sims):
                if round1_mask[game_idx]:
                    left_winner = left_refs[game_idx]
                    right_winner = right_refs[game_idx]
                else:
                    left_winner = winners_by_game[left_refs[game_idx], sim_idx]
                    right_winner = winners_by_game[right_refs[game_idx], sim_idx]

                winner_idx = constrained_winners[game_idx]
                if winner_idx < 0:
                    delta = (ratings[left_winner] - ratings[right_winner]) / rating_scale
                    left_prob = 1.0 / (1.0 + np.exp(-delta))
                    winner_idx = left_winner if np.random.random() < left_prob else right_winner

                winners_by_game[game_idx, sim_idx] = winner_idx
                team_wins[sim_idx, winner_idx] += 1

        champions = winners_by_game[championship_game_index].copy()
        return team_wins, champions

else:

    def _simulate_tournament_numba_core(
        ratings: npt.NDArray[np.float64],
        round1_mask: npt.NDArray[np.bool_],
        left_refs: npt.NDArray[np.int16],
        right_refs: npt.NDArray[np.int16],
        constrained_winners: npt.NDArray[np.int16],
        championship_game_index: int,
        n_sims: int,
        seed: int,
        rating_scale: float,
    ) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
        raise AssertionError("numba core should not be called when numba is unavailable")
