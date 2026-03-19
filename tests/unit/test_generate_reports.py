from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from bracket_sim.application.generate_reports import generate_reports
from bracket_sim.application.simulate_pool import simulate_pool
from bracket_sim.domain.models import ReportConfig, SimulationConfig
from bracket_sim.domain.scoring_systems import ScoringSystemKey


def test_generate_reports_writes_deterministic_bundle(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "report_bundle"
    result = generate_reports(
        ReportConfig(
            input_dir=synthetic_input_dir,
            output_dir=output_dir,
            n_sims=120,
            seed=13,
            batch_size=40,
        )
    )

    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    team_path = output_dir / "team_advancement_odds.csv"
    entry_path = output_dir / "entry_summary.csv"
    sensitivity_path = output_dir / "champion_sensitivity.csv"

    assert manifest_path.exists()
    assert summary_path.exists()
    assert team_path.exists()
    assert entry_path.exists()
    assert sensitivity_path.exists()

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    team_rows = _read_csv_rows(team_path)
    entry_rows = _read_csv_rows(entry_path)
    sensitivity_rows = _read_csv_rows(sensitivity_path)

    assert manifest_payload["report_id"] == result.summary.report_id
    assert manifest_payload["batch_size"] == 40
    assert len(manifest_payload["dataset_hash"]) == 64
    assert len(manifest_payload["artifacts"]) == 4
    assert summary_payload["report_id"] == result.summary.report_id
    assert summary_payload["n_sims"] == 120
    assert len(team_rows) == 64
    assert len(entry_rows) == 8
    champion_teams = {
        row["team_id"]
        for row in team_rows
        if float(row["win_championship"]) > 0.0
    }
    assert len(sensitivity_rows) == len(champion_teams) * len(entry_rows)

    championship_total = sum(float(row["win_championship"]) for row in team_rows)
    win_percentage_total = sum(float(row["win_percentage"]) for row in entry_rows)
    assert math.isclose(championship_total, 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(win_percentage_total, 100.0, rel_tol=1e-12, abs_tol=1e-12)


def test_generate_reports_entry_summary_matches_simulation_results(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "report_compare"
    report_result = generate_reports(
        ReportConfig(
            input_dir=synthetic_input_dir,
            output_dir=output_dir,
            n_sims=150,
            seed=17,
            batch_size=50,
        )
    )
    simulation_result = simulate_pool(
        SimulationConfig(
            input_dir=synthetic_input_dir,
            n_sims=150,
            seed=17,
            batch_size=50,
        )
    )

    entry_rows = _read_csv_rows(output_dir / "entry_summary.csv")
    actual_by_entry = {
        row["entry_id"]: (float(row["win_percentage"]), float(row["average_score"]))
        for row in entry_rows
    }

    for entry in simulation_result.entry_results:
        win_percentage, average_score = actual_by_entry[entry.entry_id]
        assert win_percentage == entry.win_share * 100
        assert average_score == entry.average_score

    champion_rows = report_result.summary.top_champions
    assert champion_rows
    assert champion_rows[0].probability >= champion_rows[-1].probability


def test_generate_reports_supports_round_of_64_seed_scoring(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "report_r64_seed"
    report_result = generate_reports(
        ReportConfig(
            input_dir=synthetic_input_dir,
            output_dir=output_dir,
            n_sims=180,
            seed=23,
            batch_size=60,
            scoring_system=ScoringSystemKey.ROUND_OF_64_SEED,
        )
    )
    simulation_result = simulate_pool(
        SimulationConfig(
            input_dir=synthetic_input_dir,
            n_sims=180,
            seed=23,
            batch_size=60,
            scoring_system=ScoringSystemKey.ROUND_OF_64_SEED,
        )
    )

    manifest_payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["scoring_system"] == "round-of-64-seed"

    entry_rows = _read_csv_rows(output_dir / "entry_summary.csv")
    actual_by_entry = {
        row["entry_id"]: (float(row["win_percentage"]), float(row["average_score"]))
        for row in entry_rows
    }
    for entry in simulation_result.entry_results:
        win_percentage, average_score = actual_by_entry[entry.entry_id]
        assert win_percentage == entry.win_share * 100
        assert average_score == entry.average_score

    assert report_result.summary.n_sims == 180


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
