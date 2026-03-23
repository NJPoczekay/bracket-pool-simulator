from __future__ import annotations

import json
from datetime import UTC, datetime, time
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import cast
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from bracket_sim.application.run_pool_pipeline import PoolPipelineResult, create_report_output_dir
from bracket_sim.domain.models import ChampionOddsRow, EntryReportRow, ReportSummary
from bracket_sim.infrastructure.web.config import PoolProfile, PoolRegistry, PoolSchedule
from bracket_sim.infrastructure.web.main import create_app
from bracket_sim.infrastructure.web.service import PoolService


def test_pools_api_lists_multiple_pools_and_latest_reports(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    beta = registry.pools[1]
    _write_report_bundle(beta.reports_root / "20260315-090000", report_id="beta-report")
    service = PoolService(registry, runner=_successful_runner)

    app = create_app(
        config_path=tmp_path / "unused.toml",
        service=service,
        enable_scheduler=False,
    )
    with TestClient(app) as client:
        response = client.get("/api/pools")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["pools"]) == 2
    assert payload["pools"][0]["latest_report"] is None
    assert payload["pools"][1]["latest_report"]["summary"]["report_id"] == "beta-report"
    assert payload["pools"][1]["latest_report"]["entries"][1]["entry_name"] == "Entry Two"


def test_run_pool_api_executes_runner_and_returns_latest_report(tmp_path: Path) -> None:
    service = PoolService(_build_registry(tmp_path), runner=_successful_runner)

    app = create_app(
        config_path=tmp_path / "unused.toml",
        service=service,
        enable_scheduler=False,
    )
    with TestClient(app) as client:
        run_response = client.post("/api/pools/alpha/run")
        latest_response = client.get("/api/pools/alpha/latest-report")

    assert run_response.status_code == 200
    assert latest_response.status_code == 200
    run_payload = run_response.json()["pool"]["latest_report"]
    latest_payload = latest_response.json()["latest_report"]
    assert run_payload["summary"]["report_id"] == "latest-report"
    assert latest_payload["summary"]["report_id"] == "latest-report"
    assert latest_payload["entries"][0]["win_percentage"] == 55.0
    assert latest_payload["entries"][1]["entry_name"] == "Entry Two"
    assert "game_outcome_sensitivity.csv" in latest_payload["artifacts"]
    assert "pivotal_games.csv" in latest_payload["artifacts"]
    cache_buster = Path(latest_payload["report_dir"]).name
    assert latest_payload["history_plot"] is None or latest_payload["history_plot"]["url"].endswith(
        f"?v={cache_buster}"
    )
    assert latest_payload["artifacts"]["pivotal_games.csv"]["url"].endswith(f"?v={cache_buster}")


def test_run_pool_html_redirects_back_to_tracker_page_after_success(tmp_path: Path) -> None:
    service = PoolService(_build_registry(tmp_path), runner=_successful_runner)

    app = create_app(
        config_path=tmp_path / "unused.toml",
        service=service,
        enable_scheduler=False,
    )
    with TestClient(app) as client:
        response = client.post("/pools/alpha/run", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/#pool-tracker"


def test_run_pool_api_rejects_busy_and_unknown_pool(tmp_path: Path) -> None:
    started = Event()
    release = Event()

    def blocking_runner(config: PoolProfile, *, started_at: datetime) -> PoolPipelineResult:
        started.set()
        release.wait(timeout=5.0)
        return _successful_runner(config, started_at=started_at)

    service = PoolService(_build_registry(tmp_path), runner=blocking_runner)
    worker = Thread(target=service.run_pool, args=("alpha",), daemon=True)
    worker.start()
    assert started.wait(timeout=5.0)

    app = create_app(
        config_path=tmp_path / "unused.toml",
        service=service,
        enable_scheduler=False,
    )
    with TestClient(app) as client:
        busy_response = client.post("/api/pools/beta/run")
        missing_response = client.get("/api/pools/missing/latest-report")

    release.set()
    worker.join(timeout=5.0)

    assert busy_response.status_code == 409
    assert "already in progress" in busy_response.json()["detail"]
    assert missing_response.status_code == 404


def test_dashboard_renders_multiple_pools_and_request_local_error(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    beta = registry.pools[1]
    _write_report_bundle(beta.reports_root / "20260315-090000", report_id="beta-report")

    def failing_runner(config: PoolProfile, *, started_at: datetime) -> PoolPipelineResult:
        if config.id == "alpha":
            raise ValueError("refresh failed")
        return _successful_runner(config, started_at=started_at)

    service = PoolService(registry, runner=failing_runner)

    app = create_app(
        config_path=tmp_path / "unused.toml",
        service=service,
        enable_scheduler=False,
    )
    with TestClient(app) as client:
        dashboard = client.get("/")
        failed_run = client.post("/pools/alpha/run")

    assert dashboard.status_code == 200
    assert "Bracket Lab" in dashboard.text
    assert "Pool Tracker" in dashboard.text
    assert "Alpha Pool" in dashboard.text
    assert "Beta Pool" in dashboard.text
    assert "Entry One" in dashboard.text
    assert "Entry Two" in dashboard.text
    assert "Entry Odds" in dashboard.text
    assert "Top Champions" not in dashboard.text
    assert failed_run.status_code == 400
    assert "refresh failed" in failed_run.text


def test_dashboard_and_latest_report_render_viewing_guide_sections(tmp_path: Path) -> None:
    registry = PoolRegistry(
        pools=[
            PoolProfile(
                id="alpha",
                name="Alpha Pool",
                group_url="https://fantasy.espn.com/games/mock-challenge-2026/group?id=alpha",
                raw_dir=tmp_path / "raw-alpha",
                prepared_dir=tmp_path / "prepared-alpha",
                reports_root=tmp_path / "reports-alpha",
                ratings_file=tmp_path / "ratings.csv",
                use_kenpom=False,
                min_usable_entries=1,
                n_sims=100,
                seed=0,
                batch_size=50,
                engine="numpy",
                schedule=PoolSchedule(
                    enabled=True,
                    daily_time=time(hour=8, minute=0),
                    timezone="America/Los_Angeles",
                ),
            )
        ]
    )
    alpha = registry.pools[0]
    _write_dynamic_viewing_guide_input(alpha.prepared_dir)
    _write_dynamic_viewing_guide_bundle(
        alpha.reports_root / "20260322-090000",
        report_id="alpha-guide",
    )
    service = PoolService(registry, runner=_successful_runner)

    app = create_app(
        config_path=tmp_path / "unused.toml",
        service=service,
        enable_scheduler=False,
    )
    with TestClient(app) as client:
        dashboard = client.get("/")
        latest_response = client.get("/api/pools/alpha/latest-report")

    assert dashboard.status_code == 200
    assert latest_response.status_code == 200
    assert "Watchlist" in dashboard.text
    assert "My Rooting Guide" in dashboard.text
    assert "Top Game Per Entry" in dashboard.text
    assert 'data-viewing-guide-root' in dashboard.text
    payload = latest_response.json()["latest_report"]["viewing_guide"]
    assert payload is not None
    assert payload["default_entry_id"] == "entry-1"
    assert payload["watchlist"][0]["recommended_outcome_team_name"] == "Alpha"


def test_download_latest_artifact_returns_file_contents(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    alpha = registry.pools[0]
    _write_report_bundle(alpha.reports_root / "20260315-090000", report_id="alpha-report")
    service = PoolService(registry, runner=_successful_runner)

    app = create_app(
        config_path=tmp_path / "unused.toml",
        service=service,
        enable_scheduler=False,
    )
    with TestClient(app) as client:
        response = client.get("/pools/alpha/artifacts/pivotal_games.csv")

    assert response.status_code == 200
    assert "pivotal_outcome_team_id" in response.text


def _build_registry(tmp_path: Path) -> PoolRegistry:
    return PoolRegistry(
        pools=[
            PoolProfile(
                id="alpha",
                name="Alpha Pool",
                group_url="https://fantasy.espn.com/games/mock-challenge-2026/group?id=alpha",
                raw_dir=tmp_path / "raw-alpha",
                prepared_dir=tmp_path / "prepared-alpha",
                reports_root=tmp_path / "reports-alpha",
                ratings_file=tmp_path / "ratings.csv",
                use_kenpom=False,
                min_usable_entries=1,
                n_sims=100,
                seed=0,
                batch_size=50,
                engine="numpy",
            ),
            PoolProfile(
                id="beta",
                name="Beta Pool",
                group_url="https://fantasy.espn.com/games/mock-challenge-2026/group?id=beta",
                raw_dir=tmp_path / "raw-beta",
                prepared_dir=tmp_path / "prepared-beta",
                reports_root=tmp_path / "reports-beta",
                ratings_file=tmp_path / "ratings.csv",
                use_kenpom=False,
                min_usable_entries=1,
                n_sims=100,
                seed=0,
                batch_size=50,
                engine="numpy",
            ),
        ]
    )


def _successful_runner(config: PoolProfile, *, started_at: datetime) -> PoolPipelineResult:
    report_dir = create_report_output_dir(reports_root=config.reports_root, started_at=started_at)
    _write_report_bundle(report_dir, report_id="latest-report")
    return cast(PoolPipelineResult, SimpleNamespace(report_dir=report_dir))


def _write_report_bundle(report_dir: Path, *, report_id: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = ReportSummary(
        report_id=report_id,
        output_dir=report_dir,
        n_sims=100,
        seed=7,
        engine="numpy",
        batch_size=50,
        entry_count=2,
        team_count=1,
        top_entries=[
            EntryReportRow(
                rank=1,
                entry_id="entry-1",
                entry_name="Entry One",
                win_share=0.55,
                average_score=71.2,
            )
        ],
        top_champions=[
            ChampionOddsRow(
                rank=1,
                team_id="team-1",
                team_name="Team One",
                probability=0.24,
            )
        ],
    )
    (report_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "entry_summary.csv").write_text(
        (
            "rank,entry_id,entry_name,win_percentage,average_score\n"
            "1,entry-1,Entry One,55.0,71.2\n"
            "2,entry-2,Entry Two,45.0,68.4\n"
        ),
        encoding="utf-8",
    )
    (report_dir / "team_advancement_odds.csv").write_text(
        "team_id,win_championship\nteam-1,0.24\n",
        encoding="utf-8",
    )
    (report_dir / "champion_sensitivity.csv").write_text(
        "champion_team_id,entry_id\nteam-1,entry-1\n",
        encoding="utf-8",
    )
    (report_dir / "game_outcome_sensitivity.csv").write_text(
        "game_id,outcome_team_id,entry_id\ng001,team-1,entry-1\n",
        encoding="utf-8",
    )
    (report_dir / "pivotal_games.csv").write_text(
        "game_id,pivotal_outcome_team_id\ng001,team-1\n",
        encoding="utf-8",
    )


def _write_dynamic_viewing_guide_input(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    guide_zone = ZoneInfo("America/Los_Angeles")
    local_now = datetime.now(UTC).astimezone(guide_zone)
    tonight_tipoff = local_now.replace(hour=19, minute=30, second=0, microsecond=0)
    if tonight_tipoff.date() != local_now.date():
        tonight_tipoff = local_now
    tonight_tipoff_utc = tonight_tipoff.astimezone(UTC)
    completed_tipoff_utc = tonight_tipoff_utc.replace(hour=max(0, tonight_tipoff_utc.hour - 3))

    teams = [
        {"team_id": "t1", "name": "Alpha", "seed": 1, "region": "east"},
        {"team_id": "t2", "name": "Beta", "seed": 16, "region": "east"},
        {"team_id": "t3", "name": "Gamma", "seed": 8, "region": "west"},
        {"team_id": "t4", "name": "Delta", "seed": 9, "region": "west"},
    ]
    games = [
        {
            "game_id": "g001",
            "round": 1,
            "left_team_id": "t1",
            "right_team_id": "t2",
            "left_game_id": None,
            "right_game_id": None,
            "display_order": 1,
            "scheduled_at_utc": tonight_tipoff_utc.isoformat(),
            "completed_at_utc": None,
        },
        {
            "game_id": "g002",
            "round": 1,
            "left_team_id": "t3",
            "right_team_id": "t4",
            "left_game_id": None,
            "right_game_id": None,
            "display_order": 2,
            "scheduled_at_utc": completed_tipoff_utc.isoformat(),
            "completed_at_utc": tonight_tipoff_utc.isoformat(),
        },
        {
            "game_id": "g003",
            "round": 2,
            "left_team_id": None,
            "right_team_id": None,
            "left_game_id": "g001",
            "right_game_id": "g002",
            "display_order": 1,
            "scheduled_at_utc": None,
            "completed_at_utc": None,
        },
    ]
    entries = [
        {
            "entry_id": "entry-1",
            "entry_name": "Entry One",
            "picks": {"g001": "t1", "g002": "t3", "g003": "t1"},
        },
        {
            "entry_id": "entry-2",
            "entry_name": "Entry Two",
            "picks": {"g001": "t2", "g002": "t4", "g003": "t4"},
        },
    ]
    constraints = [{"game_id": "g002", "winner_team_id": "t3"}]

    (path / "teams.json").write_text(json.dumps(teams, indent=2) + "\n", encoding="utf-8")
    (path / "games.json").write_text(json.dumps(games, indent=2) + "\n", encoding="utf-8")
    (path / "entries.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    (path / "constraints.json").write_text(
        json.dumps(constraints, indent=2) + "\n",
        encoding="utf-8",
    )
    (path / "ratings.csv").write_text(
        (
            "team_id,rating,tempo\n"
            "t1,28.0,68.0\n"
            "t2,2.0,67.0\n"
            "t3,16.0,69.0\n"
            "t4,15.0,70.0\n"
        ),
        encoding="utf-8",
    )


def _write_dynamic_viewing_guide_bundle(report_dir: Path, *, report_id: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = ReportSummary(
        report_id=report_id,
        output_dir=report_dir,
        n_sims=100,
        seed=7,
        engine="numpy",
        batch_size=50,
        entry_count=2,
        team_count=4,
        top_entries=[
            EntryReportRow(
                rank=1,
                entry_id="entry-1",
                entry_name="Entry One",
                win_share=0.55,
                average_score=71.2,
            )
        ],
        top_champions=[
            ChampionOddsRow(
                rank=1,
                team_id="t1",
                team_name="Alpha",
                probability=0.24,
            )
        ],
    )
    (report_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "entry_summary.csv").write_text(
        (
            "rank,entry_id,entry_name,win_percentage,average_score\n"
            "1,entry-1,Entry One,55.0,71.2\n"
            "2,entry-2,Entry Two,45.0,68.4\n"
        ),
        encoding="utf-8",
    )
    (report_dir / "game_outcome_sensitivity.csv").write_text(
        (
            "game_id,round,game_label,outcome_team_id,outcome_team_name,outcome_probability,"
            "entry_id,baseline_win_percentage,conditional_win_percentage,"
            "win_percentage_point_delta,outcome_total_win_percentage_point_swing\n"
            "g001,1,Round 1 Game g001,t1,Alpha,0.61,entry-1,55.0,60.0,5.0,12.0\n"
            "g001,1,Round 1 Game g001,t1,Alpha,0.61,entry-2,45.0,39.5,-5.5,12.0\n"
            "g001,1,Round 1 Game g001,t2,Beta,0.39,entry-1,55.0,50.0,-5.0,10.5\n"
            "g001,1,Round 1 Game g001,t2,Beta,0.39,entry-2,45.0,50.5,5.5,10.5\n"
        ),
        encoding="utf-8",
    )
