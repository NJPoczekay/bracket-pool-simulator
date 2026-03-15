from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.helpers.mock_espn_payloads import build_mock_payloads

from bracket_sim.application.prepare_data import prepare_data
from bracket_sim.application.refresh_data import refresh_data
from bracket_sim.infrastructure.providers.espn_api import EspnApiProvider
from bracket_sim.infrastructure.providers.ratings import LocalRatingsProvider


def _build_provider(
    *,
    challenge_payload: dict[str, Any],
    group_payloads: list[dict[str, Any]],
) -> EspnApiProvider:
    group_call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal group_call_count

        if request.url.path == "/apis/v1/challenges/mock-challenge-2026":
            return httpx.Response(200, json=challenge_payload)

        if request.url.path == "/apis/v1/challenges/mock-challenge-2026/groups/mock-group-2026":
            payload = group_payloads[min(group_call_count, len(group_payloads) - 1)]
            group_call_count += 1
            return httpx.Response(200, json=payload)

        return httpx.Response(404, json={"error": "not-found"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return EspnApiProvider(
        group_url="https://fantasy.espn.com/games/mock-challenge-2026/group?id=mock-group-2026",
        api_base_url="https://mock.espn.test",
        client=client,
    )


def _write_team_id_ratings(path: Path, fixture_dir: Path) -> None:
    teams = json.loads((fixture_dir / "teams.json").read_text(encoding="utf-8"))
    lines = ["team,rating,tempo"]
    for idx, team in enumerate(sorted(teams, key=lambda row: row["team_id"]), start=1):
        lines.append(f"{team['team_id']},{200 - idx:.3f},{64 + (idx % 5):.1f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("completed_round_cutoff", "expected_constraints"),
    [
        (0, 0),
        (1, 32),
        (4, 60),
    ],
)
def test_refresh_data_outputs_prepare_compatible_raw_for_progress_states(
    synthetic_input_dir: Path,
    tmp_path: Path,
    completed_round_cutoff: int,
    expected_constraints: int,
) -> None:
    games = json.loads((synthetic_input_dir / "games.json").read_text(encoding="utf-8"))
    completed_game_ids = {
        game["game_id"]
        for game in games
        if int(game["round"]) <= completed_round_cutoff
    }

    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=completed_game_ids,
    )

    provider = _build_provider(
        challenge_payload=challenge_payload,
        group_payloads=[group_payload],
    )

    ratings_path = tmp_path / f"ratings_{completed_round_cutoff}.csv"
    _write_team_id_ratings(ratings_path, synthetic_input_dir)

    raw_dir = tmp_path / f"raw_{completed_round_cutoff}"
    summary = refresh_data(
        group_url="https://fantasy.espn.com/games/mock-challenge-2026/group?id=mock-group-2026",
        raw_dir=raw_dir,
        results_provider=provider,
        entries_provider=provider,
        ratings_provider=LocalRatingsProvider(ratings_file=ratings_path),
        min_usable_entries=1,
        fetched_at=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )

    assert summary.teams == 64
    assert summary.games == 63
    assert summary.constraints == expected_constraints
    assert summary.entries > 0

    for filename in (
        "teams.csv",
        "games.csv",
        "entries.json",
        "constraints.json",
        "ratings.csv",
        "metadata.json",
    ):
        assert (raw_dir / filename).exists(), filename

    prepared_dir = tmp_path / f"prepared_{completed_round_cutoff}"
    prepared = prepare_data(raw_dir=raw_dir, out_dir=prepared_dir)
    assert prepared.teams == 64
    assert prepared.games == 63
    assert prepared.constraints == expected_constraints


def test_refresh_data_is_deterministic_for_fixed_responses(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    games = json.loads((synthetic_input_dir / "games.json").read_text(encoding="utf-8"))
    completed_game_ids = {
        game["game_id"]
        for game in games
        if int(game["round"]) <= 1
    }

    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=completed_game_ids,
    )

    ratings_path = tmp_path / "ratings.csv"
    _write_team_id_ratings(ratings_path, synthetic_input_dir)

    fixed_now = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)

    raw_a = tmp_path / "raw_a"
    provider_a = _build_provider(
        challenge_payload=challenge_payload,
        group_payloads=[group_payload],
    )
    refresh_data(
        group_url="https://fantasy.espn.com/games/mock-challenge-2026/group?id=mock-group-2026",
        raw_dir=raw_a,
        results_provider=provider_a,
        entries_provider=provider_a,
        ratings_provider=LocalRatingsProvider(ratings_file=ratings_path),
        fetched_at=fixed_now,
    )

    raw_b = tmp_path / "raw_b"
    provider_b = _build_provider(
        challenge_payload=challenge_payload,
        group_payloads=[group_payload],
    )
    refresh_data(
        group_url="https://fantasy.espn.com/games/mock-challenge-2026/group?id=mock-group-2026",
        raw_dir=raw_b,
        results_provider=provider_b,
        entries_provider=provider_b,
        ratings_provider=LocalRatingsProvider(ratings_file=ratings_path),
        fetched_at=fixed_now,
    )

    for filename in (
        "teams.csv",
        "games.csv",
        "entries.json",
        "constraints.json",
        "ratings.csv",
        "metadata.json",
    ):
        assert (raw_a / filename).read_text(encoding="utf-8") == (
            raw_b / filename
        ).read_text(encoding="utf-8")


def test_refresh_data_enforces_min_usable_entries_threshold(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )

    broken_group_payload = copy.deepcopy(group_payload)
    assert isinstance(broken_group_payload["entries"], list)
    for entry in broken_group_payload["entries"]:
        entry["picks"] = None

    provider = _build_provider(
        challenge_payload=challenge_payload,
        group_payloads=[broken_group_payload, broken_group_payload],
    )

    ratings_path = tmp_path / "ratings.csv"
    _write_team_id_ratings(ratings_path, synthetic_input_dir)

    with pytest.raises(ValueError, match="Usable entries below threshold"):
        refresh_data(
            group_url="https://fantasy.espn.com/games/mock-challenge-2026/group?id=mock-group-2026",
            raw_dir=tmp_path / "raw",
            results_provider=provider,
            entries_provider=provider,
            ratings_provider=LocalRatingsProvider(ratings_file=ratings_path),
            min_usable_entries=1,
            fetched_at=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
        )
