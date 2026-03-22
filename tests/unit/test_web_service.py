from __future__ import annotations

from datetime import UTC, datetime, time
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import cast

import pytest

from bracket_sim.application.run_pool_pipeline import PoolPipelineResult, create_report_output_dir
from bracket_sim.domain.models import ChampionOddsRow, EntryReportRow, ReportSummary
from bracket_sim.infrastructure.web.app import PoolScheduler
from bracket_sim.infrastructure.web.config import PoolProfile, PoolRegistry, PoolSchedule
from bracket_sim.infrastructure.web.service import (
    PoolRunBusyError,
    PoolService,
    find_latest_report,
    is_pool_due_today,
)


def test_create_report_output_dir_adds_suffix_when_timestamp_exists(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    reports_root.mkdir()
    started_at = datetime(2026, 3, 15, 14, 30, 22, tzinfo=UTC)

    first = create_report_output_dir(reports_root=reports_root, started_at=started_at)
    first.mkdir()
    second = create_report_output_dir(reports_root=reports_root, started_at=started_at)

    assert first.name == "20260315-143022"
    assert second.name == "20260315-143022-01"


def test_find_latest_report_uses_newest_timestamped_directory(tmp_path: Path) -> None:
    pool = _build_registry(tmp_path).pools[0]
    older_dir = pool.reports_root / "20260315-080000"
    newer_dir = pool.reports_root / "20260315-093000"
    _write_report_bundle(older_dir, report_id="old-report")
    _write_report_bundle(newer_dir, report_id="new-report")

    latest = find_latest_report(pool)

    assert latest is not None
    assert latest.report_dir == newer_dir
    assert latest.summary.report_id == "new-report"
    assert [entry.entry_name for entry in latest.entries] == ["Entry One", "Entry Two"]


def test_find_latest_report_ignores_materialized_latest_directory(tmp_path: Path) -> None:
    pool = _build_registry(tmp_path).pools[0]
    timestamped_dir = pool.reports_root / "20260315-093000"
    latest_dir = pool.reports_root / "latest"
    _write_report_bundle(timestamped_dir, report_id="timestamped-report")
    _write_report_bundle(latest_dir, report_id="latest-copy")

    latest = find_latest_report(pool)

    assert latest is not None
    assert latest.report_dir == timestamped_dir


def test_is_pool_due_today_respects_local_time_and_existing_success(tmp_path: Path) -> None:
    schedule = PoolSchedule(
        enabled=True,
        daily_time=time(hour=8, minute=0),
        timezone="America/Los_Angeles",
    )
    pool = _build_registry(tmp_path, schedule=schedule).pools[0]
    before_due = datetime(2026, 3, 15, 14, 30, tzinfo=UTC)
    after_due = datetime(2026, 3, 15, 16, 30, tzinfo=UTC)

    assert is_pool_due_today(pool=pool, latest_report=None, now=before_due) is False
    assert is_pool_due_today(pool=pool, latest_report=None, now=after_due) is True

    today_dir = pool.reports_root / "20260315-091500"
    _write_report_bundle(today_dir, report_id="today-report")
    latest = find_latest_report(pool)
    assert latest is not None
    assert is_pool_due_today(pool=pool, latest_report=latest, now=after_due) is False


def test_pool_service_rejects_concurrent_runs(tmp_path: Path) -> None:
    started = Event()
    release = Event()

    def blocking_runner(config: PoolProfile, *, started_at: datetime) -> PoolPipelineResult:
        started.set()
        release.wait(timeout=5.0)
        report_dir = create_report_output_dir(
            reports_root=config.reports_root,
            started_at=started_at,
        )
        _write_report_bundle(report_dir, report_id="blocking")
        return cast(PoolPipelineResult, SimpleNamespace(report_dir=report_dir))

    service = PoolService(_build_registry(tmp_path), runner=blocking_runner)
    worker = Thread(target=service.run_pool, args=("alpha",), daemon=True)
    worker.start()
    assert started.wait(timeout=5.0)

    with pytest.raises(PoolRunBusyError, match="already in progress"):
        service.run_pool("alpha")

    release.set()
    worker.join(timeout=5.0)


def test_pool_scheduler_runs_due_pool_once_per_day(tmp_path: Path) -> None:
    schedule = PoolSchedule(
        enabled=True,
        daily_time=time(hour=8, minute=0),
        timezone="America/Los_Angeles",
    )

    def successful_runner(config: PoolProfile, *, started_at: datetime) -> PoolPipelineResult:
        report_dir = create_report_output_dir(
            reports_root=config.reports_root,
            started_at=started_at,
        )
        _write_report_bundle(report_dir, report_id=report_dir.name)
        return cast(PoolPipelineResult, SimpleNamespace(report_dir=report_dir))

    service = PoolService(_build_registry(tmp_path, schedule=schedule), runner=successful_runner)
    scheduler = PoolScheduler(service, poll_seconds=0.01)
    due_time = datetime(2026, 3, 15, 16, 30, tzinfo=UTC)

    first = scheduler.run_pending(now=due_time)
    second = scheduler.run_pending(now=due_time)

    assert first == ["alpha"]
    assert second == []


def test_pool_scheduler_skips_busy_lock_and_tries_again_later(tmp_path: Path) -> None:
    schedule = PoolSchedule(
        enabled=True,
        daily_time=time(hour=8, minute=0),
        timezone="America/Los_Angeles",
    )
    started = Event()
    release = Event()

    def blocking_runner(config: PoolProfile, *, started_at: datetime) -> PoolPipelineResult:
        started.set()
        release.wait(timeout=5.0)
        report_dir = create_report_output_dir(
            reports_root=config.reports_root,
            started_at=started_at,
        )
        _write_report_bundle(report_dir, report_id="scheduled-report")
        return cast(PoolPipelineResult, SimpleNamespace(report_dir=report_dir))

    service = PoolService(_build_registry(tmp_path, schedule=schedule), runner=blocking_runner)
    scheduler = PoolScheduler(service, poll_seconds=0.01)
    worker = Thread(
        target=service.run_pool,
        kwargs={"pool_id": "alpha", "now": datetime(2026, 3, 15, 16, 30, tzinfo=UTC)},
        daemon=True,
    )
    worker.start()
    assert started.wait(timeout=5.0)

    skipped = scheduler.run_pending(now=datetime(2026, 3, 15, 16, 31, tzinfo=UTC))

    release.set()
    worker.join(timeout=5.0)

    assert skipped == []


def _build_registry(tmp_path: Path, schedule: PoolSchedule | None = None) -> PoolRegistry:
    return PoolRegistry(
        pools=[
            PoolProfile(
                id="alpha",
                name="Alpha Pool",
                group_url="https://fantasy.espn.com/games/mock-challenge-2026/group?id=alpha",
                raw_dir=tmp_path / "raw",
                prepared_dir=tmp_path / "prepared",
                reports_root=tmp_path / "reports",
                ratings_file=tmp_path / "ratings.csv",
                use_kenpom=False,
                min_usable_entries=1,
                n_sims=100,
                seed=7,
                batch_size=50,
                engine="numpy",
                schedule=schedule,
            )
        ]
    )


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
