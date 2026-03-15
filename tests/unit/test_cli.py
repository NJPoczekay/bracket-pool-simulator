from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from bracket_sim.infrastructure.cli.main import app


def test_simulate_command_runs_with_table_output(synthetic_input_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "simulate",
            "--input",
            str(synthetic_input_dir),
            "--n-sims",
            "100",
            "--seed",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert "Win Share" in result.stdout
    assert "Simulations: 100" in result.stdout


def test_simulate_command_json_output_is_stable(synthetic_input_dir: Path) -> None:
    runner = CliRunner()
    args = [
        "simulate",
        "--input",
        str(synthetic_input_dir),
        "--n-sims",
        "120",
        "--seed",
        "17",
        "--json",
    ]

    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first.stdout == second.stdout

    payload = json.loads(first.stdout)
    assert payload["n_sims"] == 120
    assert payload["seed"] == 17
    assert isinstance(payload["entry_results"], list)
    assert isinstance(payload["champion_counts"], dict)


def test_prepare_data_command_runs(raw_canonical_dir: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "prepared_cli"

    result = runner.invoke(
        app,
        [
            "prepare-data",
            "--raw",
            str(raw_canonical_dir),
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Prepared dataset written to:" in result.stdout
    assert (out_dir / "teams.json").exists()
    assert (out_dir / "cache" / "manifest.json").exists()


def test_prepare_data_command_surfaces_validation_errors(
    raw_canonical_dir: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw_bad"
    shutil.copytree(raw_canonical_dir, raw_dir)
    (raw_dir / "teams.csv").unlink()

    result = runner.invoke(
        app,
        [
            "prepare-data",
            "--raw",
            str(raw_dir),
            "--out",
            str(tmp_path / "prepared_bad"),
        ],
    )

    assert result.exit_code == 1
    assert "Error:" in result.stderr
