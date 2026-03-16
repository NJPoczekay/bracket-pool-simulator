from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
from pathlib import Path

import pytest

from bracket_sim.application.simulate_pool import simulate_pool
from bracket_sim.domain.bracket_graph import BracketGraph
from bracket_sim.domain.models import SimulationConfig
from bracket_sim.domain.simulator import (
    TournamentSimulation,
)
from bracket_sim.domain.simulator import (
    simulate_tournament as real_simulate_tournament,
)


def test_simulate_pool_is_deterministic(synthetic_input_dir: Path) -> None:
    config = SimulationConfig(input_dir=synthetic_input_dir, n_sims=400, seed=42)
    first = simulate_pool(config)
    second = simulate_pool(config)

    assert first == second
    assert first.n_sims == 400
    assert first.seed == 42
    assert first.run_metadata.engine == "numpy"
    assert first.run_metadata.batch_size == 400
    assert first.run_metadata.batches_completed == 1


def test_simulate_pool_outputs_valid_probabilities(synthetic_input_dir: Path) -> None:
    config = SimulationConfig(input_dir=synthetic_input_dir, n_sims=250, seed=11)
    result = simulate_pool(config)

    total_share = sum(entry.win_share for entry in result.entry_results)
    assert math.isclose(total_share, 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert sum(result.champion_counts.values()) == 250

    shares = [entry.win_share for entry in result.entry_results]
    assert shares == sorted(shares, reverse=True)


def test_simulate_pool_requires_complete_team_id_ratings(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input_missing_rating"
    shutil.copytree(synthetic_input_dir, input_dir)

    ratings_path = input_dir / "ratings.csv"
    with ratings_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    rows[0]["team_id"] = "missing-team-id"
    with ratings_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Missing rating for team id"):
        simulate_pool(SimulationConfig(input_dir=input_dir, n_sims=100, seed=7))


def test_simulate_pool_writes_manifest_checkpoint_result_and_logs(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "phase5_run"
    result = simulate_pool(
        SimulationConfig(
            input_dir=synthetic_input_dir,
            n_sims=120,
            seed=13,
            batch_size=50,
            run_dir=run_dir,
            log_level="info",
        )
    )

    manifest_path = run_dir / "manifest.json"
    checkpoint_path = run_dir / "checkpoint.json"
    result_path = run_dir / "result.json"
    log_path = run_dir / "log.jsonl"

    assert manifest_path.exists()
    assert checkpoint_path.exists()
    assert result_path.exists()
    assert log_path.exists()

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    log_events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    assert manifest_payload["run_id"] == result.run_metadata.run_id
    assert manifest_payload["batch_size"] == 50
    assert len(manifest_payload["dataset_hash"]) == 64
    assert manifest_payload["input_hashes"]["teams.json"]
    assert checkpoint_payload["completed_sims"] == 120
    assert checkpoint_payload["completed_batches"] == 3
    assert result_payload == result.model_dump(mode="json")
    assert any(event["event"] == "simulation_started" for event in log_events)
    assert any(event["event"] == "simulation_completed" for event in log_events)
    assert all(event["run_id"] == result.run_metadata.run_id for event in log_events)


def test_simulate_pool_can_resume_from_checkpoint(
    synthetic_input_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "resume_run"
    config = SimulationConfig(
        input_dir=synthetic_input_dir,
        n_sims=90,
        seed=21,
        batch_size=30,
        run_dir=run_dir,
        log_level="info",
    )

    call_count = 0

    def flaky_simulate_tournament(
        graph: BracketGraph,
        ratings_by_team_id: dict[str, float],
        constraints_by_game_id: dict[str, str],
        n_sims: int,
        seed: int,
        rating_scale: float,
        engine: str = "numpy",
    ) -> TournamentSimulation:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("synthetic crash before batch 2")
        return real_simulate_tournament(
            graph=graph,
            ratings_by_team_id=ratings_by_team_id,
            constraints_by_game_id=constraints_by_game_id,
            n_sims=n_sims,
            seed=seed,
            rating_scale=rating_scale,
            engine=engine,
        )

    monkeypatch.setattr(
        "bracket_sim.application.simulate_pool.simulate_tournament",
        flaky_simulate_tournament,
    )
    with pytest.raises(RuntimeError, match="synthetic crash"):
        simulate_pool(config)

    checkpoint_payload = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint_payload["completed_sims"] == 30
    assert checkpoint_payload["completed_batches"] == 1

    monkeypatch.setattr(
        "bracket_sim.application.simulate_pool.simulate_tournament",
        real_simulate_tournament,
    )
    resumed = simulate_pool(config.model_copy(update={"resume": True}))
    fresh = simulate_pool(
        SimulationConfig(
            input_dir=synthetic_input_dir,
            n_sims=90,
            seed=21,
            batch_size=30,
        )
    )

    assert resumed.entry_results == fresh.entry_results
    assert resumed.champion_counts == fresh.champion_counts
    assert resumed.run_metadata.resumed_from_checkpoint is True
    assert resumed.run_metadata.batches_completed == 3


def test_simulate_pool_resume_rejects_manifest_mismatch(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "mismatch_run"
    simulate_pool(
        SimulationConfig(
            input_dir=synthetic_input_dir,
            n_sims=80,
            seed=5,
            batch_size=40,
            run_dir=run_dir,
        )
    )

    with pytest.raises(ValueError, match="Run manifest verification failed"):
        simulate_pool(
            SimulationConfig(
                input_dir=synthetic_input_dir,
                n_sims=80,
                seed=6,
                batch_size=40,
                run_dir=run_dir,
                resume=True,
            )
        )


def test_simulate_pool_numba_feature_flag(synthetic_input_dir: Path) -> None:
    numpy_config = SimulationConfig(
        input_dir=synthetic_input_dir,
        n_sims=60,
        seed=19,
        batch_size=20,
        engine="numpy",
    )
    numba_config = numpy_config.model_copy(update={"engine": "numba"})

    if importlib.util.find_spec("numba") is None:
        with pytest.raises(ValueError, match="numba is not installed"):
            simulate_pool(numba_config)
        return

    numpy_result = simulate_pool(numpy_config)
    numba_result = simulate_pool(numba_config)

    assert numba_result.entry_results == numpy_result.entry_results
    assert numba_result.champion_counts == numpy_result.champion_counts
