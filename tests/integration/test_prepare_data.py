from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from bracket_sim.application.prepare_data import prepare_data
from bracket_sim.application.simulate_pool import simulate_pool
from bracket_sim.domain.models import SimulationConfig
from bracket_sim.infrastructure.storage.normalized_loader import load_normalized_input


def test_prepare_data_generates_simulate_compatible_dataset(
    raw_canonical_dir: Path,
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "prepared"
    summary = prepare_data(raw_dir=raw_canonical_dir, out_dir=out_dir)

    assert summary.output_dir == out_dir
    assert summary.teams == 64
    assert summary.games == 63
    assert summary.entries > 0
    assert summary.ratings == 64

    loaded = load_normalized_input(out_dir)
    assert len(loaded.teams) == 64
    assert len(loaded.games) == 63

    prepared_result = simulate_pool(
        SimulationConfig(input_dir=out_dir, n_sims=250, seed=21, rating_scale=10.0)
    )
    baseline_result = simulate_pool(
        SimulationConfig(input_dir=synthetic_input_dir, n_sims=250, seed=21, rating_scale=10.0)
    )
    assert prepared_result.entry_results == baseline_result.entry_results
    assert prepared_result.champion_counts == baseline_result.champion_counts

    with (out_dir / "ratings.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["team_id", "rating", "tempo"]
        first_row = next(reader)
        assert first_row["team_id"]
        assert first_row["rating"]
        assert first_row["tempo"]

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "prepare-data.v1"
    assert metadata["storage"]["workflow"] == "tracker"


def test_prepare_data_is_deterministic(raw_canonical_dir: Path, tmp_path: Path) -> None:
    out_a = tmp_path / "prepared_a"
    out_b = tmp_path / "prepared_b"

    prepare_data(raw_dir=raw_canonical_dir, out_dir=out_a)
    prepare_data(raw_dir=raw_canonical_dir, out_dir=out_b)

    for filename in (
        "teams.json",
        "games.json",
        "entries.json",
        "constraints.json",
        "ratings.csv",
        "metadata.json",
    ):
        assert (out_a / filename).read_text(encoding="utf-8") == (
            out_b / filename
        ).read_text(encoding="utf-8")


def test_prepare_data_fails_on_unknown_alias(raw_canonical_dir: Path, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_bad_alias"
    shutil.copytree(raw_canonical_dir, raw_dir)

    entries = json.loads((raw_dir / "entries.json").read_text(encoding="utf-8"))
    first_entry = entries[0]
    first_game_id = sorted(first_entry["picks"])[0]
    first_entry["picks"][first_game_id] = "mystery-team"
    (raw_dir / "entries.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown alias"):
        prepare_data(raw_dir=raw_dir, out_dir=tmp_path / "prepared")
