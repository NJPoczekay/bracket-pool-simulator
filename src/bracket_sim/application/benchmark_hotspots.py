"""Hotspot benchmark entrypoints for simulation and scoring."""

from __future__ import annotations

from statistics import fmean
from time import perf_counter

from bracket_sim.domain.bracket_graph import build_bracket_graph
from bracket_sim.domain.constraints import validate_constraints
from bracket_sim.domain.models import BenchmarkConfig, BenchmarkMeasurement, BenchmarkReport
from bracket_sim.domain.scoring import build_predicted_wins_matrix, score_entries, validate_entries
from bracket_sim.domain.simulator import simulate_tournament
from bracket_sim.infrastructure.storage.normalized_loader import load_normalized_input


def benchmark_hotspots(config: BenchmarkConfig) -> BenchmarkReport:
    """Measure simulation and scoring hotspots against configured budgets."""

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

    simulation_timings_ms: list[float] = []
    scoring_timings_ms: list[float] = []

    for repeat_index in range(config.repeats):
        started_sim = perf_counter()
        simulation = simulate_tournament(
            graph=graph,
            ratings_by_team_id=ratings_by_team_id,
            constraints_by_game_id=constraints_by_game_id,
            n_sims=config.n_sims,
            seed=10_000 + repeat_index,
            rating_scale=config.rating_scale,
            engine=config.engine,
        )
        simulation_timings_ms.append((perf_counter() - started_sim) * 1000)

        started_scoring = perf_counter()
        score_entries(predicted_wins=predicted_wins, actual_wins=simulation.team_wins)
        scoring_timings_ms.append((perf_counter() - started_scoring) * 1000)

    return BenchmarkReport(
        n_sims=config.n_sims,
        repeats=config.repeats,
        engine=config.engine,
        simulation=_measurement(
            timings_ms=simulation_timings_ms,
            budget_ms=config.simulation_budget_ms,
        ),
        scoring=_measurement(
            timings_ms=scoring_timings_ms,
            budget_ms=config.scoring_budget_ms,
        ),
    )


def _measurement(*, timings_ms: list[float], budget_ms: float) -> BenchmarkMeasurement:
    return BenchmarkMeasurement(
        mean_ms=round(fmean(timings_ms), 3),
        min_ms=round(min(timings_ms), 3),
        budget_ms=budget_ms,
        within_budget=max(timings_ms) <= budget_ms,
    )
