from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from bracket_sim.infrastructure.storage.path_defaults import (
    bracket_lab_context_from_challenge,
    build_bracket_lab_paths,
    build_tracker_paths,
    derive_prepared_out_dir,
    national_picks_context_from_challenge,
    report_publish_targets_for_input,
    season_from_challenge_key,
    tracker_context_from_group,
)


def test_season_from_challenge_key_parses_trailing_year() -> None:
    assert season_from_challenge_key("tournament-challenge-bracket-2026") == "2026"


def test_season_from_challenge_key_falls_back_to_current_year() -> None:
    assert season_from_challenge_key("mock-challenge") == str(datetime.now(UTC).year)


def test_tracker_default_paths_are_season_first() -> None:
    context = tracker_context_from_group(
        challenge_key="tournament-challenge-bracket-2026",
        group_id="Friends Pool",
    )

    paths = build_tracker_paths(base_dir=Path("."), context=context)

    assert paths.raw_dir == Path("data/2026/tracker/friends-pool/raw")
    assert paths.prepared_dir == Path("data/2026/tracker/friends-pool/prepared")
    assert paths.reports_root == Path("reports/2026/tracker/friends-pool")
    assert paths.runs_root == Path("runs/2026/tracker/friends-pool")


def test_bracket_lab_and_national_pick_paths_are_season_first() -> None:
    bracket_lab_paths = build_bracket_lab_paths(
        base_dir=Path("."),
        context=bracket_lab_context_from_challenge("tournament-challenge-bracket-2026"),
    )
    national_picks_context = national_picks_context_from_challenge(
        "tournament-challenge-bracket-2026"
    )

    assert bracket_lab_paths.raw_dir == Path(
        "data/2026/bracket-lab/tournament-challenge-bracket-2026/raw"
    )
    assert bracket_lab_paths.prepared_dir == Path(
        "data/2026/bracket-lab/tournament-challenge-bracket-2026/prepared"
    )
    assert Path("data/2026/national-picks/tournament-challenge-bracket-2026") == (
        Path(".")
        / "data"
        / national_picks_context.season
        / "national-picks"
        / national_picks_context.dataset_slug
    )


def test_derive_prepared_out_dir_prefers_sibling_prepared() -> None:
    assert derive_prepared_out_dir(Path("data/2026/tracker/main/raw")) == Path(
        "data/2026/tracker/main/prepared"
    )
    assert derive_prepared_out_dir(Path("custom/raw_inputs")) == Path("custom/prepared")


def test_report_publish_targets_prefer_prepared_metadata(tmp_path: Path) -> None:
    input_dir = tmp_path / "prepared"
    input_dir.mkdir()
    (input_dir / "metadata.json").write_text(
        json.dumps(
            {
                "storage": {
                    "workflow": "tracker",
                    "season": "2026",
                    "dataset_slug": "main-pool",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    targets = report_publish_targets_for_input(input_dir=input_dir, base_dir=Path("."))

    assert targets.reports_root == Path("reports/2026/tracker/main-pool")
    assert targets.latest_dir == Path("reports/2026/tracker/main-pool/latest")


def test_report_publish_targets_fall_back_to_known_path_shape(tmp_path: Path) -> None:
    input_dir = tmp_path / "data" / "2026" / "tracker" / "alpha" / "prepared"
    input_dir.mkdir(parents=True)

    targets = report_publish_targets_for_input(input_dir=input_dir, base_dir=Path("."))

    assert targets.reports_root == Path("reports/2026/tracker/alpha")
