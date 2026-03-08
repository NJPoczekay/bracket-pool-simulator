from __future__ import annotations

import math
from pathlib import Path

from bracket_sim.application.simulate_pool import simulate_pool
from bracket_sim.domain.models import SimulationConfig


def test_simulate_pool_is_deterministic(synthetic_input_dir: Path) -> None:
    config = SimulationConfig(input_dir=synthetic_input_dir, n_sims=400, seed=42)
    first = simulate_pool(config)
    second = simulate_pool(config)

    assert first == second
    assert first.n_sims == 400
    assert first.seed == 42


def test_simulate_pool_outputs_valid_probabilities(synthetic_input_dir: Path) -> None:
    config = SimulationConfig(input_dir=synthetic_input_dir, n_sims=250, seed=11)
    result = simulate_pool(config)

    total_share = sum(entry.win_share for entry in result.entry_results)
    assert math.isclose(total_share, 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert sum(result.champion_counts.values()) == 250

    shares = [entry.win_share for entry in result.entry_results]
    assert shares == sorted(shares, reverse=True)
