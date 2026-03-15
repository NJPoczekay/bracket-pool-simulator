"""Simulation orchestration entrypoints."""

from __future__ import annotations

from collections import Counter

import numpy as np

from bracket_sim.domain.bracket_graph import build_bracket_graph
from bracket_sim.domain.constraints import validate_constraints
from bracket_sim.domain.models import SimulationConfig, SimulationEntryResult, SimulationResult
from bracket_sim.domain.scoring import (
    aggregate_win_shares,
    build_predicted_wins_matrix,
    score_entries,
    validate_entries,
)
from bracket_sim.domain.simulator import simulate_tournament
from bracket_sim.infrastructure.storage.normalized_loader import load_normalized_input


def simulate_pool(config: SimulationConfig) -> SimulationResult:
    """Load validated inputs, run deterministic simulations, and aggregate entry odds."""

    normalized = load_normalized_input(config.input_dir)
    graph = build_bracket_graph(teams=normalized.teams, games=normalized.games)

    constraints_by_game_id = validate_constraints(
        constraints=normalized.constraints,
        graph=graph,
    )

    validate_entries(entries=normalized.entries, graph=graph)
    _, team_ids, predicted_wins = build_predicted_wins_matrix(
        entries=normalized.entries,
        graph=graph,
    )

    ratings_by_team_id = {record.team_id: record.rating for record in normalized.ratings.records}
    missing_team_ids = sorted(set(team_ids) - set(ratings_by_team_id))
    if missing_team_ids:
        msg = f"Missing rating for team id(s): {missing_team_ids[:5]}"
        raise ValueError(msg)

    simulation = simulate_tournament(
        graph=graph,
        ratings_by_team_id=ratings_by_team_id,
        constraints_by_game_id=constraints_by_game_id,
        n_sims=config.n_sims,
        seed=config.seed,
        rating_scale=config.rating_scale,
    )

    scores = score_entries(predicted_wins=predicted_wins, actual_wins=simulation.team_wins)
    win_shares = aggregate_win_shares(scores)

    entry_results: list[SimulationEntryResult] = []
    average_scores = np.mean(scores, axis=1)
    for entry_idx, entry in enumerate(normalized.entries):
        entry_results.append(
            SimulationEntryResult(
                entry_id=entry.entry_id,
                entry_name=entry.entry_name,
                win_share=float(win_shares[entry_idx]),
                average_score=float(average_scores[entry_idx]),
            )
        )

    entry_results.sort(
        key=lambda result: (result.win_share, result.average_score, result.entry_id),
        reverse=True,
    )

    champion_counter = Counter(simulation.champions.tolist())
    champion_counts = {
        simulation.team_ids[team_idx]: int(count)
        for team_idx, count in sorted(champion_counter.items(), key=lambda item: item[0])
    }

    return SimulationResult(
        n_sims=config.n_sims,
        seed=config.seed,
        entry_results=entry_results,
        champion_counts=champion_counts,
    )
