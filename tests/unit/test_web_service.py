from __future__ import annotations

import json
from datetime import UTC, datetime, time
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import cast

import pytest

from bracket_sim.application.run_pool_pipeline import PoolPipelineResult, create_report_output_dir
from bracket_sim.domain.models import (
    ChampionOddsRow,
    EntryReportRow,
    ReportArtifact,
    ReportBundleManifest,
    ReportSummary,
)
from bracket_sim.domain.scoring_systems import ScoringSystemKey
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


def test_pool_service_prefers_manifest_input_dir_for_viewing_guide(tmp_path: Path) -> None:
    fixed_now = datetime(2026, 3, 22, 20, 0, tzinfo=UTC)
    manifest_input_dir = tmp_path / "manifest-prepared"
    _write_viewing_guide_input_dir(manifest_input_dir)
    registry = PoolRegistry(
        pools=[
            PoolProfile(
                id="alpha",
                name="Alpha Pool",
                group_url="https://fantasy.espn.com/games/mock-challenge-2026/group?id=alpha",
                raw_dir=tmp_path / "raw",
                prepared_dir=tmp_path / "missing-prepared",
                reports_root=tmp_path / "reports",
                ratings_file=tmp_path / "ratings.csv",
                use_kenpom=False,
                min_usable_entries=1,
                n_sims=100,
                seed=7,
                batch_size=50,
                engine="numpy",
            )
        ]
    )
    report_dir = registry.pools[0].reports_root / "20260322-120000"
    _write_viewing_guide_report_bundle(
        report_dir,
        report_id="manifest-guide",
        input_dir=manifest_input_dir,
        valid_manifest=True,
        valid_game_outcome_csv=True,
    )

    service = PoolService(registry)
    latest = service.get_latest_report("alpha", now=fixed_now)

    assert latest is not None
    assert latest.viewing_guide is not None
    assert latest.viewing_guide.default_entry_id == "entry-1"
    assert latest.viewing_guide.watchlist[0].game_id == "g001"


def test_pool_service_falls_back_to_pool_prepared_dir_for_legacy_bundles(tmp_path: Path) -> None:
    fixed_now = datetime(2026, 3, 22, 20, 0, tzinfo=UTC)
    registry = _build_registry(tmp_path)
    _write_viewing_guide_input_dir(registry.pools[0].prepared_dir)
    report_dir = registry.pools[0].reports_root / "20260322-120000"
    _write_viewing_guide_report_bundle(
        report_dir,
        report_id="legacy-guide",
        input_dir=None,
        valid_manifest=False,
        valid_game_outcome_csv=True,
    )

    service = PoolService(registry)
    latest = service.get_latest_report("alpha", now=fixed_now)

    assert latest is not None
    assert latest.viewing_guide is not None
    assert latest.viewing_guide.watchlist[0].recommended_outcome_team_name == "Alpha"


def test_pool_service_degrades_gracefully_when_viewing_guide_inputs_are_invalid(
    tmp_path: Path,
) -> None:
    fixed_now = datetime(2026, 3, 22, 20, 0, tzinfo=UTC)
    registry = _build_registry(tmp_path)
    _write_viewing_guide_input_dir(registry.pools[0].prepared_dir)
    report_dir = registry.pools[0].reports_root / "20260322-120000"
    _write_viewing_guide_report_bundle(
        report_dir,
        report_id="broken-guide",
        input_dir=None,
        valid_manifest=False,
        valid_game_outcome_csv=False,
    )

    service = PoolService(registry)
    latest = service.get_latest_report("alpha", now=fixed_now)

    assert latest is not None
    assert latest.entries[0].entry_name == "Entry One"
    assert latest.viewing_guide is None


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


def _write_viewing_guide_input_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
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
            "scheduled_at_utc": "2026-03-23T02:30:00+00:00",
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
            "scheduled_at_utc": "2026-03-22T23:00:00+00:00",
            "completed_at_utc": "2026-03-23T01:00:00+00:00",
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


def _write_viewing_guide_report_bundle(
    report_dir: Path,
    *,
    report_id: str,
    input_dir: Path | None,
    valid_manifest: bool,
    valid_game_outcome_csv: bool,
) -> None:
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
    if valid_manifest and input_dir is not None:
        manifest = ReportBundleManifest(
            report_id=report_id,
            created_at=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
            code_version="test",
            git_commit=None,
            input_dir=input_dir,
            dataset_hash="a" * 64,
            input_hashes={},
            output_dir=report_dir,
            n_sims=100,
            seed=7,
            rating_scale=11.0,
            batch_size=50,
            engine="numpy",
            scoring_system=ScoringSystemKey.ESPN,
            entry_ids=["entry-1", "entry-2"],
            team_ids=["t1", "t2", "t3", "t4"],
            artifacts=[
                ReportArtifact(
                    name="entry_summary.csv",
                    path=report_dir / "entry_summary.csv",
                    kind="csv",
                    sha256="b" * 64,
                    row_count=2,
                ),
                ReportArtifact(
                    name="game_outcome_sensitivity.csv",
                    path=report_dir / "game_outcome_sensitivity.csv",
                    kind="csv",
                    sha256="c" * 64,
                    row_count=4,
                ),
            ],
        )
        (report_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        (report_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "entry_summary.csv").write_text(
        (
            "rank,entry_id,entry_name,win_percentage,average_score\n"
            "1,entry-1,Entry One,55.0,71.2\n"
            "2,entry-2,Entry Two,45.0,68.4\n"
        ),
        encoding="utf-8",
    )
    if valid_game_outcome_csv:
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
    else:
        (report_dir / "game_outcome_sensitivity.csv").write_text(
            "game_id,entry_id\ng001,entry-1\n",
            encoding="utf-8",
        )
