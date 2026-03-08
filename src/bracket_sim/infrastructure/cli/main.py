"""Typer CLI entrypoint."""

from typing import Annotated

import typer

from bracket_sim.application.simulate_pool import simulate_pool
from bracket_sim.domain.models import SimulationConfig

app = typer.Typer(help="Bracket pool simulator CLI")


@app.command()
def simulate(
    n_sims: Annotated[int, typer.Option(help="Number of simulations to run")] = 100_000,
    seed: Annotated[int, typer.Option(help="Deterministic random seed")] = 42,
) -> None:
    """Run the deterministic placeholder simulation command."""

    config = SimulationConfig(n_sims=n_sims, seed=seed)
    summary = simulate_pool(config)
    typer.echo(summary)


def main() -> None:
    """Console script entrypoint."""

    app()


if __name__ == "__main__":
    main()
