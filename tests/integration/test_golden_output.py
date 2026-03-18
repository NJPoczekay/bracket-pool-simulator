from __future__ import annotations

import json
from pathlib import Path

from bracket_sim.application.simulate_pool import simulate_pool
from bracket_sim.domain.models import SimulationConfig


def test_golden_output_matches_expected_snapshot(synthetic_input_dir: Path) -> None:
    expected_path = (
        Path(__file__).resolve().parents[1] / "expected" / "synthetic_64_seed99_n300.json"
    )
    expected_payload = json.loads(expected_path.read_text(encoding="utf-8"))

    result = simulate_pool(
        SimulationConfig(
            input_dir=synthetic_input_dir,
            n_sims=300,
            seed=99,
            rating_scale=11.0,
        )
    )

    assert result.model_dump(mode="json") == expected_payload
