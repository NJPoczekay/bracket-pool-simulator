from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from tests.helpers.mock_espn_payloads import build_mock_payloads

from bracket_sim.application.refresh_national_picks import refresh_national_picks
from bracket_sim.infrastructure.providers.espn_api import EspnApiProvider


def _build_provider(*, challenge_payload: dict[str, Any]) -> EspnApiProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/v1/challenges/mock-challenge-2026":
            return httpx.Response(200, json=challenge_payload)
        return httpx.Response(404, json={"error": "not-found"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return EspnApiProvider(
        challenge="mock-challenge-2026",
        api_base_url="https://mock.espn.test",
        client=client,
    )


def test_refresh_national_picks_writes_expected_artifacts(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    challenge_payload, _, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )

    out_dir = tmp_path / "national_picks"
    summary = refresh_national_picks(
        challenge="mock-challenge-2026",
        out_dir=out_dir,
        provider=_build_provider(challenge_payload=challenge_payload),
        fetched_at=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )

    assert summary.output_dir == out_dir
    assert summary.games == 63
    assert summary.rows == 384
    assert summary.total_brackets == 1_000

    for filename in ("national_picks.csv", "metadata.json", "snapshots/challenge.json"):
        assert (out_dir / filename).exists(), filename

    with (out_dir / "national_picks.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "game_id",
        "round",
        "display_order",
        "outcome_id",
        "team_id",
        "team_name",
        "seed",
        "region",
        "matchup_position",
        "pick_count",
        "pick_percentage",
    ]
    assert len(rows) == 384
    assert rows[0]["game_id"]
    assert rows[0]["pick_count"]

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "national-picks.v1"
    assert metadata["challenge_key"] == "mock-challenge-2026"
    assert metadata["challenge_id"] == 277
    assert metadata["games"] == 63
    assert metadata["rows"] == 384
    assert metadata["total_brackets"] == 1000
    assert metadata["source_url"] == "https://fantasy.espn.com/games/mock-challenge-2026/bracket"
    assert metadata["round_counts"] == {"1": 32, "2": 16, "3": 8, "4": 4, "5": 2, "6": 1}
    assert metadata["canonical_hash"]

    snapshot = json.loads((out_dir / "snapshots" / "challenge.json").read_text(encoding="utf-8"))
    assert snapshot == challenge_payload


def test_refresh_national_picks_is_deterministic(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    challenge_payload, _, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )
    fixed_now = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)

    out_a = tmp_path / "national_a"
    refresh_national_picks(
        challenge="mock-challenge-2026",
        out_dir=out_a,
        provider=_build_provider(challenge_payload=challenge_payload),
        fetched_at=fixed_now,
    )

    out_b = tmp_path / "national_b"
    refresh_national_picks(
        challenge="mock-challenge-2026",
        out_dir=out_b,
        provider=_build_provider(challenge_payload=challenge_payload),
        fetched_at=fixed_now,
    )

    for filename in ("national_picks.csv", "metadata.json", "snapshots/challenge.json"):
        assert (out_a / filename).read_text(encoding="utf-8") == (
            out_b / filename
        ).read_text(encoding="utf-8")
