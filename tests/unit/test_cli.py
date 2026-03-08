from typer.testing import CliRunner

from bracket_sim.infrastructure.cli.main import app


def test_simulate_command_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--n-sims", "10", "--seed", "7"])

    assert result.exit_code == 0
    assert "'seed': 7" in result.stdout
