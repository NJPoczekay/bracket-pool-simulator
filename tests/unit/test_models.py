from pydantic import ValidationError

from bracket_sim.domain.models import SimulationConfig, Team


def test_team_validation() -> None:
    team = Team(team_id="duke", name="Duke", seed=1)
    assert team.seed == 1


def test_team_seed_out_of_range() -> None:
    try:
        Team(team_id="duke", name="Duke", seed=17)
    except ValidationError:
        return

    msg = "Expected validation failure for invalid seed"
    raise AssertionError(msg)


def test_simulation_config_positive_sims() -> None:
    config = SimulationConfig(n_sims=10, seed=123)
    assert config.n_sims == 10
