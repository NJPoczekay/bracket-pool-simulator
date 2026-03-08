from bracket_sim.application.simulate_pool import simulate_pool
from bracket_sim.domain.models import SimulationConfig


def test_simulate_pool_is_deterministic() -> None:
    config = SimulationConfig(n_sims=1000, seed=42)
    first = simulate_pool(config)
    second = simulate_pool(config)

    assert first == second
    assert first["seed"] == 42
    assert first["n_sims"] == 1000
