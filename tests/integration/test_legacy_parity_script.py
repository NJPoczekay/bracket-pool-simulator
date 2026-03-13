from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from bracket_sim.application.simulate_pool import simulate_pool
from bracket_sim.domain.models import SimulationConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_parity_script(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    script_path = _repo_root() / "scripts" / "check_legacy_parity.py"
    return subprocess.run(
        [sys.executable, str(script_path), *arguments],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_legacy_payload(path: Path, entry_win_shares: dict[str, float]) -> None:
    payload = {"entry_win_shares": entry_win_shares}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _current_entry_win_shares(n_sims: int, seed: int) -> dict[str, float]:
    result = simulate_pool(
        SimulationConfig(
            input_dir=Path("tests/fixtures/synthetic_64"),
            n_sims=n_sims,
            seed=seed,
            rating_scale=10.0,
        )
    )
    return {entry.entry_id: entry.win_share for entry in result.entry_results}


def test_parity_script_fails_for_missing_legacy_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_legacy.json"
    completed = _run_parity_script(["--legacy-json", str(missing_path)])

    assert completed.returncode == 2
    assert "does not exist" in completed.stderr


def test_parity_script_passes_when_within_threshold(tmp_path: Path) -> None:
    n_sims = 300
    seed = 42
    legacy_path = tmp_path / "legacy_within_threshold.json"
    _write_legacy_payload(legacy_path, _current_entry_win_shares(n_sims=n_sims, seed=seed))

    completed = _run_parity_script(
        [
            "--legacy-json",
            str(legacy_path),
            "--n-sims",
            str(n_sims),
            "--seed",
            str(seed),
            "--top-n",
            "10",
            "--max-delta",
            "0.0",
        ]
    )

    assert completed.returncode == 0
    assert "PASS:" in completed.stdout


def test_parity_script_fails_when_delta_exceeds_threshold(tmp_path: Path) -> None:
    n_sims = 300
    seed = 42
    entry_win_shares = _current_entry_win_shares(n_sims=n_sims, seed=seed)
    top_entry_id = next(iter(entry_win_shares))
    entry_win_shares[top_entry_id] += 0.5

    legacy_path = tmp_path / "legacy_out_of_threshold.json"
    _write_legacy_payload(legacy_path, entry_win_shares)

    completed = _run_parity_script(
        [
            "--legacy-json",
            str(legacy_path),
            "--n-sims",
            str(n_sims),
            "--seed",
            str(seed),
            "--top-n",
            "10",
            "--max-delta",
            "0.02",
        ]
    )

    assert completed.returncode == 1
    assert "FAIL:" in completed.stdout
