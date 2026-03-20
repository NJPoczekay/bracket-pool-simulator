from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import cast

from fastapi.testclient import TestClient

from bracket_sim.application.run_pool_pipeline import PoolPipelineResult, create_report_output_dir
from bracket_sim.domain.models import ChampionOddsRow, EntryReportRow, ReportSummary
from bracket_sim.infrastructure.web.config import PoolProfile, PoolRegistry
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
    assert run_response.json()["pool"]["latest_report"]["summary"]["report_id"] == "latest-report"
    assert latest_response.json()["latest_report"]["summary"]["report_id"] == "latest-report"
    assert "game_outcome_sensitivity.csv" in latest_response.json()["latest_report"]["artifacts"]
    assert "pivotal_games.csv" in latest_response.json()["latest_report"]["artifacts"]


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
    assert failed_run.status_code == 400
    assert "refresh failed" in failed_run.text


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
        entry_count=1,
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
        "entry_id,win_percentage\nentry-1,55.0\n",
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
