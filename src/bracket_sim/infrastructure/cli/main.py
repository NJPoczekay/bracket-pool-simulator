"""Typer CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from bracket_sim.application.benchmark_hotspots import benchmark_hotspots
from bracket_sim.application.prepare_data import PrepareDataSummary, prepare_data
from bracket_sim.application.refresh_data import RefreshDataSummary, refresh_data
from bracket_sim.application.refresh_national_picks import (
    RefreshNationalPicksSummary,
    refresh_national_picks,
)
from bracket_sim.application.simulate_pool import simulate_pool
from bracket_sim.domain.models import (
    BenchmarkConfig,
    BenchmarkMeasurement,
    BenchmarkReport,
    SimulationConfig,
    SimulationResult,
)

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
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            help="Optional simulations per batch for checkpointed runs",
            min=1,
        ),
    ] = None,
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir",
            help="Directory to write manifest, checkpoint, logs, and result artifacts",
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume an existing run from --run-dir artifacts"),
    ] = False,
    engine: Annotated[
        str,
        typer.Option("--engine", help="Simulation engine to use: numpy or numba"),
    ] = "numpy",
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Structured log verbosity: debug, info, warning, error"),
    ] = "warning",
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit structured JSON instead of table output"),
    ] = False,
) -> None:
    """Run deterministic pool simulation from normalized local inputs."""

    try:
        config = SimulationConfig(
            input_dir=input_dir,
            n_sims=n_sims,
            seed=seed,
            batch_size=batch_size,
            run_dir=run_dir,
            resume=resume,
            engine=engine,
            log_level=log_level,
        )
        result = simulate_pool(config)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return

    typer.echo(_format_result_table(result))


@app.command("benchmark")
def benchmark_command(
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
    n_sims: Annotated[int, typer.Option(help="Simulations per timing repeat")] = 5_000,
    repeats: Annotated[int, typer.Option(help="Number of timing repeats", min=1)] = 3,
    engine: Annotated[
        str,
        typer.Option("--engine", help="Simulation engine to benchmark: numpy or numba"),
    ] = "numpy",
    simulation_budget_ms: Annotated[
        float,
        typer.Option("--simulation-budget-ms", help="Maximum allowed simulation runtime in ms"),
    ] = 1_500.0,
    scoring_budget_ms: Annotated[
        float,
        typer.Option("--scoring-budget-ms", help="Maximum allowed scoring runtime in ms"),
    ] = 750.0,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit structured JSON instead of table output"),
    ] = False,
) -> None:
    """Benchmark simulation and scoring hotspots against performance budgets."""

    try:
        config = BenchmarkConfig(
            input_dir=input_dir,
            n_sims=n_sims,
            repeats=repeats,
            engine=engine,
            simulation_budget_ms=simulation_budget_ms,
            scoring_budget_ms=scoring_budget_ms,
        )
        report = benchmark_hotspots(config)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(_format_benchmark_report(report))

    if not (report.simulation.within_budget and report.scoring.within_budget):
        raise typer.Exit(code=1)


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


@app.command("refresh-data")
def refresh_data_command(
    group_url: Annotated[
        str,
        typer.Option(
            "--group-url",
            help="ESPN Tournament Challenge group URL",
        ),
    ],
    raw_dir: Annotated[
        Path,
        typer.Option(
            "--raw",
            help="Directory to write canonical raw preparation inputs",
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ],
    ratings_file: Annotated[
        Path | None,
        typer.Option(
            "--ratings-file",
            help=(
                "Local ratings CSV path (team,rating,tempo). "
                "Uses cached raw/ratings.csv if omitted."
            ),
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    kenpom: Annotated[
        bool,
        typer.Option(
            "--kenpom",
            help=(
                "Fetch ratings from KenPom using KENPOM_COOKIE "
                "instead of local file/cached snapshot"
            ),
        ),
    ] = False,
    min_usable_entries: Annotated[
        int,
        typer.Option(
            "--min-usable-entries",
            help="Fail if parsed usable entries are below this threshold after retry/skip handling",
            min=1,
        ),
    ] = 1,
) -> None:
    """Refresh canonical raw inputs from ESPN + ratings providers."""

    try:
        summary = refresh_data(
            group_url=group_url,
            raw_dir=raw_dir,
            ratings_file=ratings_file,
            use_kenpom=kenpom,
            min_usable_entries=min_usable_entries,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(_format_refresh_summary(summary))


@app.command("refresh-national-picks")
def refresh_national_picks_command(
    challenge: Annotated[
        str,
        typer.Option(
            "--challenge",
            help="ESPN bracket URL, group URL, or challenge key",
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Directory to write national pick-count artifacts",
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ],
) -> None:
    """Download canonical ESPN national pick counts into a local snapshot."""

    try:
        summary = refresh_national_picks(challenge=challenge, out_dir=out_dir)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(_format_refresh_national_picks_summary(summary))


def _format_result_table(result: SimulationResult) -> str:
    """Render compact human-readable table output."""

    lines = [
        (
            "Run ID: "
            f"{result.run_metadata.run_id}  Engine: {result.run_metadata.engine}  "
            f"Batch Size: {result.run_metadata.batch_size}  "
            f"Batches: {result.run_metadata.batches_completed}"
        ),
        f"Simulations: {result.n_sims}  Seed: {result.seed}",
    ]

    if result.run_metadata.resumed_from_checkpoint:
        lines.append("Resumed: yes")
    if result.run_metadata.run_dir is not None:
        lines.append(f"Artifacts: {result.run_metadata.run_dir}")

    lines.extend(
        [
            "",
            f"{'Entry':<24} {'Win Share':>10} {'Avg Score':>10}",
            f"{'-' * 24} {'-' * 10} {'-' * 10}",
        ]
    )

    for entry in result.entry_results:
        lines.append(
            f"{entry.entry_name[:24]:<24} {entry.win_share:>10.4f} {entry.average_score:>10.2f}"
        )

    return "\n".join(lines)


def _format_benchmark_report(report: BenchmarkReport) -> str:
    """Render human-readable benchmark output."""

    lines = [
        f"Engine: {report.engine}  Simulations/Repeat: {report.n_sims}  Repeats: {report.repeats}",
        "",
        f"{'Hotspot':<12} {'Mean (ms)':>10} {'Min (ms)':>10} {'Budget':>10} {'Status':>8}",
        f"{'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}",
        _format_benchmark_row("simulation", report.simulation),
        _format_benchmark_row("scoring", report.scoring),
    ]
    return "\n".join(lines)


def _format_benchmark_row(name: str, measurement: BenchmarkMeasurement) -> str:
    status = "PASS" if measurement.within_budget else "FAIL"
    return (
        f"{name:<12} {measurement.mean_ms:>10.3f} {measurement.min_ms:>10.3f} "
        f"{measurement.budget_ms:>10.3f} {status:>8}"
    )


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


def _format_refresh_summary(summary: RefreshDataSummary) -> str:
    """Render human-readable summary for refresh-data command."""

    lines = [
        f"Refreshed raw dataset written to: {summary.output_dir}",
        (
            "Counts: "
            f"teams={summary.teams} games={summary.games} entries={summary.entries} "
            f"constraints={summary.constraints} ratings={summary.ratings} aliases={summary.aliases}"
        ),
        (
            "Entry handling: "
            f"skipped={summary.skipped_entries} retry_attempted={summary.retry_attempted}"
        ),
    ]
    return "\n".join(lines)


def _format_refresh_national_picks_summary(summary: RefreshNationalPicksSummary) -> str:
    """Render human-readable summary for refresh-national-picks command."""

    lines = [
        f"Refreshed national picks written to: {summary.output_dir}",
        (
            "Counts: "
            f"games={summary.games} rows={summary.rows} total_brackets={summary.total_brackets}"
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    """Console script entrypoint."""

    app()


if __name__ == "__main__":
    main()
