"""Typer CLI entrypoint."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Annotated

import typer

from bracket_sim.application.benchmark_hotspots import benchmark_hotspots
from bracket_sim.application.generate_matchup_tables import generate_matchup_tables
from bracket_sim.application.generate_reports import generate_reports
from bracket_sim.application.prepare_bracket_lab_data import prepare_bracket_lab_data
from bracket_sim.application.prepare_data import prepare_data
from bracket_sim.application.refresh_bracket_lab_data import refresh_bracket_lab_data
from bracket_sim.application.refresh_data import refresh_data
from bracket_sim.application.refresh_national_picks import refresh_national_picks
from bracket_sim.application.run_pool_pipeline import create_report_output_dir, run_pool_pipeline
from bracket_sim.application.simulate_pool import simulate_pool
from bracket_sim.domain.models import (
    BenchmarkConfig,
    ReportConfig,
    SimulationConfig,
    SimulationResult,
)
from bracket_sim.domain.scoring_systems import ScoringSystemKey
from bracket_sim.infrastructure.cli.presenter import (
    format_benchmark_report,
    format_matchup_tables,
    format_prepare_bracket_lab_summary,
    format_prepare_summary,
    format_refresh_bracket_lab_summary,
    format_refresh_national_picks_summary,
    format_refresh_summary,
    format_report_summary,
    format_result_table,
)
from bracket_sim.infrastructure.providers.espn_api import (
    parse_espn_challenge_reference,
    parse_espn_group_url,
)
from bracket_sim.infrastructure.storage.path_defaults import (
    bracket_lab_context_from_challenge,
    build_bracket_lab_paths,
    build_national_picks_dir,
    build_tracker_paths,
    default_report_timestamp,
    derive_prepared_out_dir,
    infer_storage_context_from_path,
    load_storage_context,
    national_picks_context_from_challenge,
    report_publish_targets_for_input,
    tracker_context_from_group,
)
from bracket_sim.infrastructure.storage.report_bundle import publish_latest_report
from bracket_sim.infrastructure.web.config import PoolProfile, load_pool_registry
from bracket_sim.infrastructure.web.main import run_server

app = typer.Typer(no_args_is_help=True, help="Bracket pool simulator CLI")

_DEFAULT_BASE_DIR = Path(".")
_DEFAULT_POOL_CONFIG_PATH = Path("config/pools.toml")


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
    pool_name: Annotated[
        str | None,
        typer.Option(
            "--pool-name",
            help="Optional display name printed above the simulation result table",
        ),
    ] = None,
    scoring_system: Annotated[
        ScoringSystemKey,
        typer.Option(
            "--scoring-system",
            help=(
                "Pool scoring system: 1-2-4-8-16-32, 1-2-3-4-5-6, 2-3-5-8-13-21, "
                "round+seed, round-of-64-flat, round-of-64-seed"
            ),
        ),
    ] = ScoringSystemKey.ESPN,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit structured JSON instead of table output"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Include run metadata details above the simulation table",
        ),
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
            scoring_system=scoring_system,
        )
        result = simulate_pool(config)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return

    resolved_pool_name = _resolve_pool_name(input_dir=input_dir, pool_name=pool_name)
    typer.echo(
        format_result_table(
            result,
            pool_name=resolved_pool_name,
            verbose=verbose,
        )
    )


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
    scoring_system: Annotated[
        ScoringSystemKey,
        typer.Option(
            "--scoring-system",
            help=(
                "Pool scoring system: 1-2-4-8-16-32, 1-2-3-4-5-6, 2-3-5-8-13-21, "
                "round+seed, round-of-64-flat, round-of-64-seed"
            ),
        ),
    ] = ScoringSystemKey.ESPN,
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
            scoring_system=scoring_system,
        )
        report = benchmark_hotspots(config)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(format_benchmark_report(report))

    if not (report.simulation.within_budget and report.scoring.within_budget):
        raise typer.Exit(code=1)


@app.command("report")
def report_command(
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
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help=(
                "Directory to write report bundle artifacts. Defaults to "
                "reports/<season>/<workflow>/<dataset>/<timestamp>"
            ),
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = None,
    n_sims: Annotated[int, typer.Option(help="Number of simulations to run")] = 100_000,
    seed: Annotated[int, typer.Option(help="Deterministic random seed")] = 42,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            help="Optional simulations per batch for deterministic report generation",
            min=1,
        ),
    ] = None,
    engine: Annotated[
        str,
        typer.Option("--engine", help="Simulation engine to use: numpy or numba"),
    ] = "numpy",
    scoring_system: Annotated[
        ScoringSystemKey,
        typer.Option(
            "--scoring-system",
            help=(
                "Pool scoring system: 1-2-4-8-16-32, 1-2-3-4-5-6, 2-3-5-8-13-21, "
                "round+seed, round-of-64-flat, round-of-64-seed"
            ),
        ),
    ] = ScoringSystemKey.ESPN,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit bundle summary JSON instead of text output"),
    ] = False,
) -> None:
    """Generate deterministic offline report artifacts from normalized local inputs."""

    try:
        publish_latest = out_dir is None
        latest_dir = None
        resolved_out_dir = out_dir
        if resolved_out_dir is None:
            targets = report_publish_targets_for_input(
                input_dir=input_dir,
                base_dir=_DEFAULT_BASE_DIR,
            )
            resolved_out_dir = create_report_output_dir(
                reports_root=targets.reports_root,
                started_at=default_report_timestamp(),
            )
            latest_dir = targets.latest_dir

        config = ReportConfig(
            input_dir=input_dir,
            output_dir=resolved_out_dir,
            n_sims=n_sims,
            seed=seed,
            batch_size=batch_size,
            engine=engine,
            scoring_system=scoring_system,
        )
        result = generate_reports(config)
        if publish_latest and latest_dir is not None:
            publish_latest_report(archive_dir=resolved_out_dir, latest_dir=latest_dir)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(result.summary.model_dump_json(indent=2))
        return

    typer.echo(format_report_summary(result))


@app.command("matchup-table")
def matchup_table_command(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input",
            help="Directory containing prepared Bracket Lab inputs",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    round_number: Annotated[
        int,
        typer.Option(
            "--round",
            help="Round to report. Defaults to 1 for concrete opening-round matchups",
            min=1,
            max=6,
        ),
    ] = 1,
    all_rounds: Annotated[
        bool,
        typer.Option(
            "--all-rounds",
            help="Include all rounds instead of filtering to --round",
        ),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit structured JSON instead of table output"),
    ] = False,
) -> None:
    """Generate Bracket Lab matchup win-probability and value tables."""

    try:
        result = generate_matchup_tables(
            input_dir=input_dir,
            round_filter=None if all_rounds else round_number,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(json.dumps(result.to_payload(), indent=2))
        return

    typer.echo(format_matchup_tables(result))


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
        Path | None,
        typer.Option(
            "--out",
            help=(
                "Directory to write normalized simulation inputs. Defaults to a sibling "
                "prepared directory next to --raw"
            ),
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = None,
) -> None:
    """Prepare normalized simulation inputs from canonical raw local files."""

    try:
        resolved_out_dir = out_dir or derive_prepared_out_dir(raw_dir)
        if raw_dir.resolve() == resolved_out_dir.resolve():
            msg = f"Derived default --out collides with --raw: {resolved_out_dir}"
            raise ValueError(msg)
        summary = prepare_data(raw_dir=raw_dir, out_dir=resolved_out_dir)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(format_prepare_summary(summary))


@app.command("prepare-bracket-lab-data")
def prepare_bracket_lab_data_command(
    raw_dir: Annotated[
        Path,
        typer.Option(
            "--raw",
            help="Directory containing raw Bracket Lab preparation inputs",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help=(
                "Directory to write prepared Bracket Lab inputs. Defaults to a sibling "
                "prepared directory next to --raw"
            ),
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = None,
) -> None:
    """Prepare self-contained Bracket Lab inputs from raw challenge data."""

    try:
        resolved_out_dir = out_dir or derive_prepared_out_dir(raw_dir)
        if raw_dir.resolve() == resolved_out_dir.resolve():
            msg = f"Derived default --out collides with --raw: {resolved_out_dir}"
            raise ValueError(msg)
        summary = prepare_bracket_lab_data(raw_dir=raw_dir, out_dir=resolved_out_dir)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(format_prepare_bracket_lab_summary(summary))


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
        Path | None,
        typer.Option(
            "--raw",
            help=(
                "Directory to write canonical raw preparation inputs. Defaults to "
                "data/<season>/tracker/<group-id>/raw"
            ),
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = None,
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
                "Load a saved KenPom HTML snapshot from data/kenpom_snapshots "
                "instead of a local file/cached snapshot"
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
        group_ref = parse_espn_group_url(group_url)
        resolved_raw_dir = raw_dir or build_tracker_paths(
            base_dir=_DEFAULT_BASE_DIR,
            context=tracker_context_from_group(
                challenge_key=group_ref.challenge_key,
                group_id=group_ref.group_id,
            ),
        ).raw_dir
        summary = refresh_data(
            group_url=group_url,
            raw_dir=resolved_raw_dir,
            ratings_file=ratings_file,
            use_kenpom=kenpom,
            min_usable_entries=min_usable_entries,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(format_refresh_summary(summary))


@app.command("refresh-bracket-lab-data")
def refresh_bracket_lab_data_command(
    challenge: Annotated[
        str,
        typer.Option(
            "--challenge",
            help="ESPN bracket URL, group URL, or challenge key",
        ),
    ],
    raw_dir: Annotated[
        Path | None,
        typer.Option(
            "--raw",
            help=(
                "Directory to write raw Bracket Lab inputs. Defaults to "
                "data/<season>/bracket-lab/<challenge-key>/raw"
            ),
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = None,
    ratings_file: Annotated[
        Path | None,
        typer.Option(
            "--ratings-file",
            help="Local KenPom-style CSV path with columns team,rating,tempo",
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
            help="Load a saved KenPom HTML snapshot from data/kenpom_snapshots",
        ),
    ] = False,
) -> None:
    """Refresh raw Bracket Lab data from ESPN challenge APIs plus KenPom inputs."""

    try:
        challenge_ref = parse_espn_challenge_reference(challenge)
        resolved_raw_dir = raw_dir or build_bracket_lab_paths(
            base_dir=_DEFAULT_BASE_DIR,
            context=bracket_lab_context_from_challenge(challenge_ref.challenge_key),
        ).raw_dir
        summary = refresh_bracket_lab_data(
            challenge=challenge,
            raw_dir=resolved_raw_dir,
            ratings_file=ratings_file,
            use_kenpom=kenpom,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(format_refresh_bracket_lab_summary(summary))


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
        Path | None,
        typer.Option(
            "--out",
            help=(
                "Directory to write national pick-count artifacts. Defaults to "
                "data/<season>/national-picks/<challenge-key>"
            ),
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = None,
) -> None:
    """Download canonical ESPN national pick counts into a local snapshot."""

    try:
        challenge_ref = parse_espn_challenge_reference(challenge)
        resolved_out_dir = out_dir or build_national_picks_dir(
            base_dir=_DEFAULT_BASE_DIR,
            context=national_picks_context_from_challenge(challenge_ref.challenge_key),
        )
        summary = refresh_national_picks(challenge=challenge, out_dir=resolved_out_dir)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(format_refresh_national_picks_summary(summary))


@app.command("serve")
def serve_command(
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Optional pool config TOML to enable live pool tracking data",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    bracket_lab_input: Annotated[
        Path | None,
        typer.Option(
            "--bracket-lab-input",
            help="Optional prepared Bracket Lab directory to enable analyzer workflows",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = None,
    host: Annotated[
        str,
        typer.Option("--host", help="Host interface for the local web/API server"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", help="TCP port for the local web/API server", min=1, max=65535),
    ] = 8000,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Enable auto-reload for development"),
    ] = False,
) -> None:
    """Run the local web/API surface."""
    try:
        run_server(
            host=host,
            port=port,
            reload=reload,
            config_path=config_path,
            bracket_lab_input=bracket_lab_input,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("refresh-pools")
def refresh_pools_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Pool config TOML to run all configured tracker pools",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    simulate: Annotated[
        bool,
        typer.Option(
            "--simulate",
            help=(
                "Run refresh -> prepare -> simulate for each pool instead of "
                "refresh -> prepare -> report"
            ),
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="When --simulate is set, include run metadata above each simulation table",
        ),
    ] = False,
) -> None:
    """Run end-to-end tracker pipelines for every pool in one config file."""

    try:
        registry = load_pool_registry(config_path)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    failures: list[tuple[str, str]] = []
    for index, pool in enumerate(registry.pools):
        try:
            if simulate:
                simulation_result = _run_quietly(
                    lambda pool=pool: _refresh_prepare_and_simulate_pool(pool)
                )
            else:
                _run_quietly(lambda pool=pool: run_pool_pipeline(pool))
        except ValueError as exc:
            failures.append((pool.id, str(exc)))
            typer.echo(f"Failed {pool.id}: {exc}", err=True)
            continue

        if simulate:
            typer.echo(
                format_result_table(
                    simulation_result,
                    pool_name=pool.name,
                    verbose=verbose,
                )
            )
            if index < len(registry.pools) - 1:
                typer.echo("")

    if failures:
        typer.echo(
            f"Completed with failures: {len(failures)} pool(s) failed",
            err=True,
        )
        raise typer.Exit(code=1)


def main() -> None:
    """Console script entrypoint."""

    app()


def _resolve_pool_name(*, input_dir: Path, pool_name: str | None) -> str:
    if pool_name is not None and pool_name.strip() != "":
        return pool_name.strip()

    config_name = _pool_name_from_tracker_config(input_dir=input_dir)
    if config_name is not None:
        return config_name

    storage_context = load_storage_context(input_dir) or infer_storage_context_from_path(input_dir)
    if storage_context is not None:
        return storage_context.dataset_slug

    if input_dir.name in {"prepared", "raw"}:
        return input_dir.parent.name
    return input_dir.name


def _pool_name_from_tracker_config(*, input_dir: Path) -> str | None:
    config_path = _DEFAULT_POOL_CONFIG_PATH
    if not config_path.exists():
        return None

    try:
        registry = load_pool_registry(config_path)
    except ValueError:
        return None

    resolved_input = input_dir.resolve()
    for pool in registry.pools:
        if pool.prepared_dir.resolve() == resolved_input:
            return pool.name
    return None


def _refresh_prepare_and_simulate_pool(pool: PoolProfile) -> SimulationResult:
    refresh_data(
        group_url=pool.group_url,
        raw_dir=pool.raw_dir,
        ratings_file=pool.ratings_file,
        use_kenpom=pool.use_kenpom,
        min_usable_entries=pool.min_usable_entries,
    )
    prepare_summary = prepare_data(raw_dir=pool.raw_dir, out_dir=pool.prepared_dir)
    return simulate_pool(
        SimulationConfig(
            input_dir=prepare_summary.output_dir,
            n_sims=pool.n_sims,
            seed=pool.seed,
            batch_size=pool.batch_size,
            engine=pool.engine,
            scoring_system=pool.scoring_system,
        )
    )


def _run_quietly[T](task: Callable[[], T]) -> T:
    """Suppress stdout/stderr noise from provider/library internals."""

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return task()


if __name__ == "__main__":
    main()
