from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from bracket_sim.application.tracker_viewing_guide import build_tracker_viewing_guide


def test_build_tracker_viewing_guide_filters_to_tonight_and_derives_member_guides(
    tmp_path: Path,
) -> None:
    prepared_dir = tmp_path / "prepared"
    report_dir = tmp_path / "report"
    _write_prepared_input(prepared_dir)
    _write_report_bundle(report_dir)

    guide = build_tracker_viewing_guide(
        input_dir=prepared_dir,
        report_dir=report_dir,
        now=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
        timezone=ZoneInfo("America/Los_Angeles"),
    )

    assert guide.local_date == "2026-03-22"
    assert guide.timezone == "America/Los_Angeles"
    assert guide.default_entry_id == "entry-1"
    assert guide.unavailable_schedule_count == 1
    assert [item.game_id for item in guide.watchlist] == ["g001"]
    assert guide.watchlist[0].recommended_outcome_team_name == "Alpha"
    assert guide.watchlist[0].top_gainer_entry_name == "Entry One"
    assert guide.watchlist[0].top_loser_entry_name == "Entry Two"
    assert guide.guides_by_entry_id["entry-1"][0].recommended_outcome_team_name == "Alpha"
    assert guide.guides_by_entry_id["entry-2"][0].recommended_outcome_team_name == "Beta"
    assert guide.top_games_by_entry[0].entry_id == "entry-1"
    assert guide.top_games_by_entry[0].recommended_outcome_team_name == "Alpha"
    assert guide.top_games_by_entry[1].entry_id == "entry-2"
    assert guide.top_games_by_entry[1].recommended_outcome_team_name == "Beta"


def test_build_tracker_viewing_guide_rejects_missing_required_columns(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    report_dir = tmp_path / "report"
    _write_prepared_input(prepared_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "entry_summary.csv").write_text(
        (
            "rank,entry_id,entry_name,win_percentage\n"
            "1,entry-1,Entry One,55.0\n"
        ),
        encoding="utf-8",
    )
    (report_dir / "game_outcome_sensitivity.csv").write_text(
        "game_id,entry_id\ng001,entry-1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        build_tracker_viewing_guide(
            input_dir=prepared_dir,
            report_dir=report_dir,
            now=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
            timezone=ZoneInfo("America/Los_Angeles"),
        )


def _write_prepared_input(path: Path) -> None:
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


def _write_report_bundle(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "entry_summary.csv").write_text(
        (
            "rank,entry_id,entry_name,win_percentage,average_score\n"
            "1,entry-1,Entry One,55.0,72.0\n"
            "2,entry-2,Entry Two,45.0,69.5\n"
        ),
        encoding="utf-8",
    )
    (path / "game_outcome_sensitivity.csv").write_text(
        (
            "game_id,round,game_label,outcome_team_id,outcome_team_name,outcome_probability,"
            "entry_id,baseline_win_percentage,conditional_win_percentage,"
            "win_percentage_point_delta,outcome_total_win_percentage_point_swing\n"
            "g001,1,Round 1 Game g001,t1,Alpha,0.61,entry-1,55.0,60.0,5.0,12.0\n"
            "g001,1,Round 1 Game g001,t1,Alpha,0.61,entry-2,45.0,39.5,-5.5,12.0\n"
            "g001,1,Round 1 Game g001,t2,Beta,0.39,entry-1,55.0,50.0,-5.0,10.5\n"
            "g001,1,Round 1 Game g001,t2,Beta,0.39,entry-2,45.0,50.5,5.5,10.5\n"
            "g002,1,Round 1 Game g002,t3,Gamma,0.55,entry-1,55.0,58.0,3.0,7.0\n"
            "g003,2,Round 2 Game g003,t1,Alpha,0.52,entry-1,55.0,62.0,7.0,15.0\n"
        ),
        encoding="utf-8",
    )
