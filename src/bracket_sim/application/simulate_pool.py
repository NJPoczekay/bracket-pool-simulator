"""Simulation orchestration entrypoints."""

from bracket_sim.domain.models import SimulationConfig


class SimulationSummary(dict[str, int]):
    """Minimal typed summary returned by placeholder simulation use-case."""


def simulate_pool(config: SimulationConfig) -> SimulationSummary:
    """Return deterministic placeholder output for Phase 0 scaffolding."""

    checksum = (config.seed * 31 + config.n_sims) % 100_000
    return SimulationSummary(
        {
            "seed": config.seed,
            "n_sims": config.n_sims,
            "checksum": checksum,
        }
    )
