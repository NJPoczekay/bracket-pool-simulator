"""Human-readable CLI presentation helpers."""

from __future__ import annotations

from bracket_sim.application.entry_pivotal_outcomes import EntryPivotalOutcomesResult
from bracket_sim.application.generate_matchup_tables import MatchupTablesResult
from bracket_sim.application.prepare_bracket_lab_data import PrepareBracketLabDataSummary
from bracket_sim.application.prepare_data import PrepareDataSummary
from bracket_sim.application.refresh_bracket_lab_data import RefreshBracketLabDataSummary
from bracket_sim.application.refresh_data import RefreshDataSummary
from bracket_sim.application.refresh_national_picks import RefreshNationalPicksSummary
from bracket_sim.domain.models import (
    BenchmarkMeasurement,
    BenchmarkReport,
    ReportBundleResult,
    SimulationResult,
)


def format_result_table(
    result: SimulationResult,
    *,
    pool_name: str | None = None,
    verbose: bool = False,
) -> str:
    """Render compact human-readable simulation output."""

    lines: list[str] = []

    if verbose:
        lines.extend(
            [
                (
                    "Run ID: "
                    f"{result.run_metadata.run_id}  Engine: {result.run_metadata.engine}  "
                    f"Batch Size: {result.run_metadata.batch_size}  "
                    f"Batches: {result.run_metadata.batches_completed}"
                ),
                f"Simulations: {result.n_sims}  Seed: {result.seed}",
            ]
        )

    if result.run_metadata.resumed_from_checkpoint:
        lines.append("Resumed: yes")
    if result.run_metadata.run_dir is not None:
        lines.append(f"Artifacts: {result.run_metadata.run_dir}")
    if pool_name:
        lines.append(f"Pool: {pool_name}")

    if lines:
        lines.append("")

    lines.extend(
        [
            f"{'Entry':<24} {'Win %':>10} {'Avg Score':>10}",
            f"{'-' * 24} {'-' * 10} {'-' * 10}",
        ]
    )

    for entry in result.entry_results:
        lines.append(
            f"{entry.entry_name[:24]:<24} {entry.win_share * 100:>10.2f} "
            f"{entry.average_score:>10.2f}"
        )

    return "\n".join(lines)


def format_matchup_tables(result: MatchupTablesResult) -> str:
    """Render human-readable matchup win-probability and value tables."""

    round_label = result.round_filter if result.round_filter is not None else "all"
    lines = [
        f"Input: {result.input_dir}",
        (
            f"Round: {round_label}  Games: {len({row.game_id for row in result.matchup_rows})}  "
            f"Rows: {len(result.matchup_rows)}"
        ),
    ]

    lines.extend(
        [
            "",
            "Matchup Win Probabilities",
            (
                f"{'Game':<34} {'Team':<20} {'Seed':>4} "
                f"{'Win Prob':>10} {'Public':>10} {'Value':>10}"
            ),
            f"{'-' * 34} {'-' * 20} {'-' * 4} {'-' * 10} {'-' * 10} {'-' * 10}",
        ]
    )
    for row in result.matchup_rows:
        lines.append(
            f"{row.game_label[:34]:<34} {row.team_name[:20]:<20} {row.seed:>4} "
            f"{_format_percent(row.win_probability):>10} "
            f"{_format_percent(row.public_pick_rate):>10} "
            f"{_format_value(row.value):>10}"
        )

    lines.extend(
        [
            "",
            "Value Table",
            (
                f"{'Value':>10} {'Win Prob':>10} {'Public':>10} "
                f"{'Team':<20} {'Game':<34}"
            ),
            f"{'-' * 10} {'-' * 10} {'-' * 10} {'-' * 20} {'-' * 34}",
        ]
    )
    for row in result.value_rows:
        lines.append(
            f"{_format_value(row.value):>10} "
            f"{_format_percent(row.win_probability):>10} "
            f"{_format_percent(row.public_pick_rate):>10} "
            f"{row.team_name[:20]:<20} {row.game_label[:34]:<34}"
        )

    return "\n".join(lines)


def format_entry_pivotal_outcomes(result: EntryPivotalOutcomesResult) -> str:
    """Render per-entry pivotal outcomes as a compact table."""

    lines = [
        f"Input: {result.input_dir}",
        f"Report: {result.report_dir}",
        f"Round: {result.round_number}  Entries: {len(result.rows)}",
        "",
        f"{'Entry':<24} {'Outcome':<32} {'Win % Change':>14} {'Outcome Prob':>14}",
        f"{'-' * 24} {'-' * 32} {'-' * 14} {'-' * 14}",
    ]

    for row in result.rows:
        lines.append(
            f"{row.entry_name[:24]:<24} {row.outcome_label[:32]:<32} "
            f"{_format_signed_points(row.win_percentage_point_delta):>14} "
            f"{_format_percent(row.outcome_probability):>14}"
        )

    return "\n".join(lines)


def format_benchmark_report(report: BenchmarkReport) -> str:
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


def format_report_summary(result: ReportBundleResult) -> str:
    """Render human-readable report bundle output."""

    lines = [
        f"Report bundle written to: {result.summary.output_dir}",
        (
            f"Report ID: {result.summary.report_id}  Engine: {result.summary.engine}  "
            f"Simulations: {result.summary.n_sims}  Seed: {result.summary.seed}  "
            f"Batch Size: {result.summary.batch_size}"
        ),
        (
            f"Artifacts: {len(result.manifest.artifacts)} files  "
            f"Entries: {result.summary.entry_count}  Teams: {result.summary.team_count}"
        ),
    ]

    if result.summary.top_entries:
        lines.extend(
            [
                "",
                f"{'Top Entries':<24} {'Win %':>10} {'Avg Score':>10}",
                f"{'-' * 24} {'-' * 10} {'-' * 10}",
            ]
        )
        for entry in result.summary.top_entries:
            lines.append(
                f"{entry.entry_name[:24]:<24} {entry.win_share * 100:>10.2f} "
                f"{entry.average_score:>10.2f}"
            )

    if result.summary.top_champions:
        lines.extend(["", "Top Champions"])
        for champion in result.summary.top_champions:
            lines.append(
                f"{champion.rank:>2}. {champion.team_name} ({champion.probability:.4f})"
            )

    return "\n".join(lines)


def format_prepare_summary(summary: PrepareDataSummary) -> str:
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


def format_prepare_bracket_lab_summary(summary: PrepareBracketLabDataSummary) -> str:
    """Render human-readable summary for prepare-bracket-lab-data command."""

    lines = [
        f"Prepared Bracket Lab dataset written to: {summary.output_dir}",
        (
            "Counts: "
            f"teams={summary.teams} games={summary.games} constraints={summary.constraints} "
            f"public_picks={summary.public_picks} ratings={summary.ratings} "
            f"play_in_slots={summary.play_in_slots}"
        ),
    ]
    return "\n".join(lines)


def format_refresh_summary(summary: RefreshDataSummary) -> str:
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


def format_refresh_bracket_lab_summary(summary: RefreshBracketLabDataSummary) -> str:
    """Render human-readable summary for refresh-bracket-lab-data command."""

    lines = [
        f"Refreshed Bracket Lab raw dataset written to: {summary.output_dir}",
        (
            "Counts: "
            f"teams={summary.teams} games={summary.games} constraints={summary.constraints} "
            f"public_pick_rows={summary.public_pick_rows} kenpom_rows={summary.kenpom_rows} "
            f"aliases={summary.aliases}"
        ),
    ]
    return "\n".join(lines)


def format_refresh_national_picks_summary(summary: RefreshNationalPicksSummary) -> str:
    """Render human-readable summary for refresh-national-picks command."""

    lines = [
        f"Refreshed national picks written to: {summary.output_dir}",
        (
            "Counts: "
            f"games={summary.games} rows={summary.rows} total_brackets={summary.total_brackets}"
        ),
    ]
    return "\n".join(lines)


def _format_benchmark_row(name: str, measurement: BenchmarkMeasurement) -> str:
    status = "PASS" if measurement.within_budget else "FAIL"
    return (
        f"{name:<12} {measurement.mean_ms:>10.3f} {measurement.min_ms:>10.3f} "
        f"{measurement.budget_ms:>10.3f} {status:>8}"
    )


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def _format_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_signed_points(value: float) -> str:
    return f"{value:+.2f} pp"
