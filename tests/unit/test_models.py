from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from bracket_sim.domain.models import Game, SimulationConfig, Team


def test_team_validation() -> None:
    team = Team(team_id="east-01", name="East Team 1", seed=1, region="east")
    assert team.seed == 1


def test_team_seed_out_of_range() -> None:
    with pytest.raises(ValidationError):
        Team(team_id="east-99", name="Invalid", seed=17, region="east")


def test_round_one_game_must_have_team_sources() -> None:
    with pytest.raises(ValidationError):
        Game(
            game_id="g001",
            round=1,
            left_team_id=None,
            right_team_id="east-16",
            left_game_id=None,
            right_game_id=None,
        )


def test_later_round_game_must_have_game_sources() -> None:
    with pytest.raises(ValidationError):
        Game(
            game_id="g033",
            round=2,
            left_team_id="east-01",
            right_team_id="east-16",
            left_game_id=None,
            right_game_id=None,
        )


def test_simulation_config_requires_positive_sims(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        SimulationConfig(input_dir=tmp_path, n_sims=0, seed=10)
