from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from bracket_sim.application.prepare_bracket_lab_data import PrepareBracketLabDataSummary
from bracket_sim.application.refresh_bracket_lab_data import RefreshBracketLabDataSummary
from bracket_sim.application.refresh_data import RefreshDataSummary
from bracket_sim.application.refresh_national_picks import RefreshNationalPicksSummary
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
    assert "Run ID:" in result.stdout
    assert "Win %" in result.stdout
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
    assert payload["run_metadata"]["engine"] == "numpy"


def test_simulate_command_supports_run_artifact_flags(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    run_dir = tmp_path / "cli_run"

    result = runner.invoke(
        app,
        [
            "simulate",
            "--input",
            str(synthetic_input_dir),
            "--n-sims",
            "90",
            "--seed",
            "12",
            "--batch-size",
            "30",
            "--run-dir",
            str(run_dir),
            "--log-level",
            "info",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["run_metadata"]["batch_size"] == 30
    assert payload["run_metadata"]["run_dir"] == str(run_dir)
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "checkpoint.json").exists()
    assert (run_dir / "result.json").exists()
    assert (run_dir / "log.jsonl").exists()


def test_benchmark_command_runs_and_emits_json(synthetic_input_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--input",
            str(synthetic_input_dir),
            "--n-sims",
            "150",
            "--repeats",
            "1",
            "--simulation-budget-ms",
            "10000",
            "--scoring-budget-ms",
            "10000",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["n_sims"] == 150
    assert payload["simulation"]["within_budget"] is True
    assert payload["scoring"]["within_budget"] is True


def test_report_command_writes_bundle_and_emits_json(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "report_cli"

    result = runner.invoke(
        app,
        [
            "report",
            "--input",
            str(synthetic_input_dir),
            "--out",
            str(out_dir),
            "--n-sims",
            "120",
            "--seed",
            "7",
            "--batch-size",
            "40",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["n_sims"] == 120
    assert payload["seed"] == 7
    assert payload["batch_size"] == 40
    assert payload["entry_count"] == 8
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "team_advancement_odds.csv").exists()
    assert (out_dir / "entry_summary.csv").exists()
    assert (out_dir / "champion_sensitivity.csv").exists()


def test_report_command_defaults_out_dir_and_publishes_latest(
    synthetic_input_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    input_dir = tmp_path / "data" / "2026" / "tracker" / "main" / "prepared"
    shutil.copytree(synthetic_input_dir, input_dir)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "report",
            "--input",
            str(input_dir),
            "--n-sims",
            "120",
            "--seed",
            "7",
            "--batch-size",
            "40",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    report_dir = Path(payload["output_dir"])
    assert report_dir.parent == Path("reports/2026/tracker/main")
    assert (tmp_path / report_dir / "summary.json").exists()
    assert (tmp_path / "reports/2026/tracker/main/latest/summary.json").exists()


def test_matchup_table_command_emits_json(
    prepared_bracket_lab_dir: Path,
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "matchup-table",
            "--input",
            str(prepared_bracket_lab_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["input_dir"] == str(prepared_bracket_lab_dir)
    assert payload["round"] == 1
    assert len(payload["matchup_rows"]) == 64
    assert len(payload["value_rows"]) == 64
    assert payload["matchup_rows"][0]["game_id"] == "g001"


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
    assert (out_dir / "ratings.csv").exists()


def test_prepare_data_command_defaults_out_dir(raw_canonical_dir: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "data" / "2026" / "tracker" / "main" / "raw"
    shutil.copytree(raw_canonical_dir, raw_dir)

    result = runner.invoke(
        app,
        [
            "prepare-data",
            "--raw",
            str(raw_dir),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "data/2026/tracker/main/prepared/teams.json").exists()


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


def test_prepare_bracket_lab_data_command_runs_with_monkeypatched_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import bracket_sim.infrastructure.cli.main as cli_main

    fake_summary = PrepareBracketLabDataSummary(
        output_dir=tmp_path / "prepared_lab",
        teams=64,
        games=63,
        constraints=0,
        public_picks=384,
        ratings=65,
        play_in_slots=1,
    )
    fake_prepare = Mock(return_value=fake_summary)
    monkeypatch.setattr(cli_main, "prepare_bracket_lab_data", fake_prepare)

    raw_dir = tmp_path / "raw_lab"
    raw_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "prepare-bracket-lab-data",
            "--raw",
            str(raw_dir),
            "--out",
            str(tmp_path / "prepared_lab"),
        ],
    )

    assert result.exit_code == 0
    assert "Prepared Bracket Lab dataset written to:" in result.stdout
    assert "play_in_slots=1" in result.stdout


def test_prepare_bracket_lab_data_command_defaults_out_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import bracket_sim.infrastructure.cli.main as cli_main

    expected_out_dir = tmp_path / "data" / "2026" / "bracket-lab" / "mock" / "prepared"
    fake_summary = PrepareBracketLabDataSummary(
        output_dir=expected_out_dir,
        teams=64,
        games=63,
        constraints=0,
        public_picks=384,
        ratings=65,
        play_in_slots=1,
    )
    fake_prepare = Mock(return_value=fake_summary)
    monkeypatch.setattr(cli_main, "prepare_bracket_lab_data", fake_prepare)

    raw_dir = tmp_path / "data" / "2026" / "bracket-lab" / "mock" / "raw"
    raw_dir.mkdir(parents=True)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "prepare-bracket-lab-data",
            "--raw",
            str(raw_dir),
        ],
    )

    assert result.exit_code == 0
    fake_prepare.assert_called_once_with(raw_dir=raw_dir, out_dir=expected_out_dir)


def test_refresh_data_command_runs_with_monkeypatched_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import bracket_sim.infrastructure.cli.main as cli_main

    fake_summary = RefreshDataSummary(
        output_dir=tmp_path / "raw",
        teams=64,
        games=63,
        entries=10,
        skipped_entries=1,
        constraints=20,
        ratings=64,
        aliases=0,
        retry_attempted=True,
    )
    fake_refresh = Mock(return_value=fake_summary)
    monkeypatch.setattr(cli_main, "refresh_data", fake_refresh)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "refresh-data",
            "--group-url",
            "https://fantasy.espn.com/games/mock-challenge-2026/group?id=mock-group-2026",
            "--raw",
            str(tmp_path / "raw"),
            "--min-usable-entries",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Refreshed raw dataset written to:" in result.stdout
    assert "retry_attempted=True" in result.stdout


def test_refresh_data_command_defaults_raw_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import bracket_sim.infrastructure.cli.main as cli_main

    expected_raw_dir = tmp_path / "data" / "2026" / "tracker" / "mock-group-2026" / "raw"
    fake_summary = RefreshDataSummary(
        output_dir=expected_raw_dir,
        teams=64,
        games=63,
        entries=10,
        skipped_entries=1,
        constraints=20,
        ratings=64,
        aliases=0,
        retry_attempted=True,
    )
    fake_refresh = Mock(return_value=fake_summary)
    monkeypatch.setattr(cli_main, "refresh_data", fake_refresh)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "refresh-data",
            "--group-url",
            "https://fantasy.espn.com/games/mock-challenge-2026/group?id=mock-group-2026",
            "--min-usable-entries",
            "1",
        ],
    )

    assert result.exit_code == 0
    fake_refresh.assert_called_once_with(
        group_url="https://fantasy.espn.com/games/mock-challenge-2026/group?id=mock-group-2026",
        raw_dir=Path("data/2026/tracker/mock-group-2026/raw"),
        ratings_file=None,
        use_kenpom=False,
        min_usable_entries=1,
    )


def test_refresh_bracket_lab_data_command_runs_with_monkeypatched_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import bracket_sim.infrastructure.cli.main as cli_main

    fake_summary = RefreshBracketLabDataSummary(
        output_dir=tmp_path / "raw_lab",
        teams=64,
        games=63,
        constraints=0,
        public_pick_rows=384,
        kenpom_rows=65,
        aliases=1,
    )
    fake_refresh = Mock(return_value=fake_summary)
    monkeypatch.setattr(cli_main, "refresh_bracket_lab_data", fake_refresh)

    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text("team,rating,tempo\nA,1,60\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "refresh-bracket-lab-data",
            "--challenge",
            "tournament-challenge-bracket-2026",
            "--raw",
            str(tmp_path / "raw_lab"),
            "--ratings-file",
            str(ratings_path),
        ],
    )

    assert result.exit_code == 0
    assert "Refreshed Bracket Lab raw dataset written to:" in result.stdout
    assert "public_pick_rows=384" in result.stdout


def test_refresh_bracket_lab_data_command_defaults_raw_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import bracket_sim.infrastructure.cli.main as cli_main

    expected_raw_dir = (
        tmp_path / "data" / "2026" / "bracket-lab" / "tournament-challenge-bracket-2026" / "raw"
    )
    fake_summary = RefreshBracketLabDataSummary(
        output_dir=expected_raw_dir,
        teams=64,
        games=63,
        constraints=0,
        public_pick_rows=384,
        kenpom_rows=65,
        aliases=1,
    )
    fake_refresh = Mock(return_value=fake_summary)
    monkeypatch.setattr(cli_main, "refresh_bracket_lab_data", fake_refresh)
    monkeypatch.chdir(tmp_path)

    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text("team,rating,tempo\nA,1,60\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "refresh-bracket-lab-data",
            "--challenge",
            "tournament-challenge-bracket-2026",
            "--ratings-file",
            str(ratings_path),
        ],
    )

    assert result.exit_code == 0
    fake_refresh.assert_called_once_with(
        challenge="tournament-challenge-bracket-2026",
        raw_dir=Path("data/2026/bracket-lab/tournament-challenge-bracket-2026/raw"),
        ratings_file=ratings_path,
        use_kenpom=False,
    )


def test_refresh_national_picks_command_runs_with_monkeypatched_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import bracket_sim.infrastructure.cli.main as cli_main

    fake_summary = RefreshNationalPicksSummary(
        output_dir=tmp_path / "national",
        games=63,
        rows=384,
        total_brackets=1_000,
    )
    fake_refresh = Mock(return_value=fake_summary)
    monkeypatch.setattr(cli_main, "refresh_national_picks", fake_refresh)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "refresh-national-picks",
            "--challenge",
            "tournament-challenge-bracket-2026",
            "--out",
            str(tmp_path / "national"),
        ],
    )

    assert result.exit_code == 0
    assert "Refreshed national picks written to:" in result.stdout
    assert "total_brackets=1000" in result.stdout


def test_refresh_national_picks_command_defaults_out_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import bracket_sim.infrastructure.cli.main as cli_main

    expected_out_dir = (
        tmp_path / "data" / "2026" / "national-picks" / "tournament-challenge-bracket-2026"
    )
    fake_summary = RefreshNationalPicksSummary(
        output_dir=expected_out_dir,
        games=63,
        rows=384,
        total_brackets=1_000,
    )
    fake_refresh = Mock(return_value=fake_summary)
    monkeypatch.setattr(cli_main, "refresh_national_picks", fake_refresh)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "refresh-national-picks",
            "--challenge",
            "tournament-challenge-bracket-2026",
        ],
    )

    assert result.exit_code == 0
    fake_refresh.assert_called_once_with(
        challenge="tournament-challenge-bracket-2026",
        out_dir=Path("data/2026/national-picks/tournament-challenge-bracket-2026"),
    )


def test_refresh_national_picks_command_help_mentions_challenge_input() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["refresh-national-picks", "--help"])

    assert result.exit_code == 0
    assert "ESPN bracket URL, group URL, or challenge" in result.stdout
    assert "key" in result.stdout
    assert "Defaults to" in result.stdout
    assert "data/<season>/national-picks/" in result.stdout


def test_refresh_bracket_lab_data_command_help_mentions_challenge_input() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["refresh-bracket-lab-data", "--help"])

    assert result.exit_code == 0
    assert "ESPN bracket URL, group URL, or" in result.stdout
    assert "challenge key" in result.stdout
    assert "ratings-file" in result.stdout
    assert "Defaults to" in result.stdout
    assert "data/<season>/bracket-lab/" in result.stdout


def test_serve_command_invokes_web_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import bracket_sim.infrastructure.cli.main as cli_main

    config_path = tmp_path / "pools.toml"
    config_path.write_text("pools = []\n", encoding="utf-8")
    bracket_lab_input = tmp_path / "bracket-lab"
    bracket_lab_input.mkdir()
    fake_run_server = Mock()
    monkeypatch.setattr(cli_main, "run_server", fake_run_server)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "serve",
            "--config",
            str(config_path),
            "--bracket-lab-input",
            str(bracket_lab_input),
            "--host",
            "127.0.0.1",
            "--port",
            "8123",
        ],
    )

    assert result.exit_code == 0
    fake_run_server.assert_called_once_with(
        host="127.0.0.1",
        port=8123,
        reload=False,
        config_path=config_path,
        bracket_lab_input=bracket_lab_input,
    )
