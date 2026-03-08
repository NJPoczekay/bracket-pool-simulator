from __future__ import annotations

import json
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
