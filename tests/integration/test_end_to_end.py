from __future__ import annotations

import json
import math
from pathlib import Path

from typer.testing import CliRunner

from bracket_sim.infrastructure.cli.main import app


def test_cli_end_to_end_deterministic_and_valid(synthetic_input_dir: Path) -> None:
    runner = CliRunner()
    args = [
        "simulate",
        "--input",
        str(synthetic_input_dir),
        "--n-sims",
        "300",
        "--seed",
        "99",
        "--json",
    ]

    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == 0
    assert first.stdout == second.stdout

    payload = json.loads(first.stdout)
    total_share = sum(result["win_share"] for result in payload["entry_results"])
    assert math.isclose(total_share, 1.0, rel_tol=1e-12, abs_tol=1e-12)

    champion_total = sum(payload["champion_counts"].values())
    assert champion_total == 300
