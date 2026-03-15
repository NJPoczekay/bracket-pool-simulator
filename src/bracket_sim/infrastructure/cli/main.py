"""Typer CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from bracket_sim.application.prepare_data import PrepareDataSummary, prepare_data
from bracket_sim.application.simulate_pool import simulate_pool
from bracket_sim.domain.models import SimulationConfig, SimulationResult

app = typer.Typer(no_args_is_help=True, help="Bracket pool simulator CLI")


@app.callback()
def root() -> None:
    """Root CLI callback."""


@app.command("simulate")
def simulate_command(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input",
            help="Directory containing normalized simulation inputs",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    n_sims: Annotated[int, typer.Option(help="Number of simulations to run")] = 100_000,
    seed: Annotated[int, typer.Option(help="Deterministic random seed")] = 42,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit structured JSON instead of table output"),
    ] = False,
) -> None:
    """Run deterministic pool simulation from normalized local inputs."""

    config = SimulationConfig(input_dir=input_dir, n_sims=n_sims, seed=seed)
    result = simulate_pool(config)

    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return

    typer.echo(_format_result_table(result))


@app.command("prepare-data")
def prepare_data_command(
    raw_dir: Annotated[
        Path,
        typer.Option(
            "--raw",
            help="Directory containing canonical raw preparation inputs",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Directory to write normalized simulation inputs",
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ],
) -> None:
    """Prepare normalized simulation inputs from canonical raw local files."""

    try:
        summary = prepare_data(raw_dir=raw_dir, out_dir=out_dir)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(_format_prepare_summary(summary))


def _format_result_table(result: SimulationResult) -> str:
    """Render compact human-readable table output."""

    lines = [
        f"Simulations: {result.n_sims}  Seed: {result.seed}",
        "",
        f"{'Entry':<24} {'Win Share':>10} {'Avg Score':>10}",
        f"{'-' * 24} {'-' * 10} {'-' * 10}",
    ]

    for entry in result.entry_results:
        lines.append(
            f"{entry.entry_name[:24]:<24} {entry.win_share:>10.4f} {entry.average_score:>10.2f}"
        )

    return "\n".join(lines)


def _format_prepare_summary(summary: PrepareDataSummary) -> str:
    """Render human-readable summary for prepare-data command."""

    lines = [
        f"Prepared dataset written to: {summary.output_dir}",
        (
            "Counts: "
            f"teams={summary.teams} games={summary.games} entries={summary.entries} "
            f"constraints={summary.constraints} ratings={summary.ratings} aliases={summary.aliases}"
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    """Console script entrypoint."""

    app()


if __name__ == "__main__":
    main()
