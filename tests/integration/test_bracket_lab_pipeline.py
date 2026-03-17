from __future__ import annotations

import copy
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.helpers.mock_espn_payloads import build_mock_payloads

from bracket_sim.application.prepare_bracket_lab_data import prepare_bracket_lab_data
from bracket_sim.application.refresh_bracket_lab_data import refresh_bracket_lab_data
from bracket_sim.infrastructure.providers.espn_api import EspnApiProvider
from bracket_sim.infrastructure.providers.ratings import LocalRatingSourceProvider
from bracket_sim.infrastructure.storage.bracket_lab_prepared_loader import (
    load_bracket_lab_prepared_input,
)


def _build_challenge_provider(*, challenge_payload: dict[str, Any]) -> EspnApiProvider:
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


def _write_name_ratings(
    path: Path,
    fixture_dir: Path,
    *,
    replacement_name: str | None = None,
) -> None:
    teams = json.loads((fixture_dir / "teams.json").read_text(encoding="utf-8"))
    lines = ["team,rating,tempo"]

    for idx, team in enumerate(sorted(teams, key=lambda row: row["team_id"]), start=1):
        name = str(team["name"])
        if replacement_name is not None and name == replacement_name:
            continue
        lines.append(f"{name},{200 - idx:.3f},{64 + (idx % 5):.1f}")

    if replacement_name is not None:
        lines.append("Miami OH,19.400,67.0")
        lines.append("SMU,21.100,69.2")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mutate_placeholder_challenge(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    mutated = copy.deepcopy(payload)
    assert isinstance(mutated["propositions"], list)
    proposition = mutated["propositions"][0]
    assert isinstance(proposition, dict)
    assert isinstance(proposition["possibleOutcomes"], list)
    outcome = proposition["possibleOutcomes"][0]
    assert isinstance(outcome, dict)
    original_name = str(outcome["name"])
    original_team_id = next(
        str(mapping["value"])
        for mapping in outcome["mappings"]
        if isinstance(mapping, dict) and mapping.get("type") == "COMPETITOR_ID"
    )

    for proposition_payload in mutated["propositions"]:
        assert isinstance(proposition_payload, dict)
        assert isinstance(proposition_payload["possibleOutcomes"], list)
        for possible_outcome in proposition_payload["possibleOutcomes"]:
            assert isinstance(possible_outcome, dict)
            mappings = possible_outcome.get("mappings")
            if not isinstance(mappings, list):
                continue
            competitor_ids = [
                str(mapping["value"])
                for mapping in mappings
                if isinstance(mapping, dict) and mapping.get("type") == "COMPETITOR_ID"
            ]
            if original_team_id not in competitor_ids:
                continue

            possible_outcome["name"] = "M-OH/SMU"
            possible_outcome["description"] = "M-OH/SMU"
            possible_outcome["abbrev"] = "M-OH/SMU"
            possible_outcome["mappings"] = [
                mapping
                for mapping in mappings
                if not (isinstance(mapping, dict) and mapping.get("type") == "COMPETITOR_ID")
            ]
    return mutated, original_name


def test_refresh_bracket_lab_data_writes_expected_artifacts_and_preserves_aliases(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    challenge_payload, _, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )
    ratings_path = tmp_path / "ratings.csv"
    _write_name_ratings(ratings_path, synthetic_input_dir)

    raw_dir = tmp_path / "raw_lab"
    raw_dir.mkdir()
    (raw_dir / "aliases.csv").write_text(
        "alias,team_id\nLegacy Alias,east-01\n",
        encoding="utf-8",
    )

    summary = refresh_bracket_lab_data(
        challenge="mock-challenge-2026",
        raw_dir=raw_dir,
        challenge_provider=_build_challenge_provider(challenge_payload=challenge_payload),
        rating_source_provider=LocalRatingSourceProvider(ratings_file=ratings_path),
        fetched_at=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )

    assert summary.teams == 64
    assert summary.games == 63
    assert summary.public_pick_rows == 384
    assert summary.aliases == 1

    for filename in (
        "teams.csv",
        "games.csv",
        "national_picks.csv",
        "kenpom.csv",
        "aliases.csv",
        "metadata.json",
        "snapshots/challenge.json",
    ):
        assert (raw_dir / filename).exists(), filename

    with (raw_dir / "aliases.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows == [{"alias": "Legacy Alias", "team_id": "east-01"}]

    metadata = json.loads((raw_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "refresh-bracket-lab-data.v1"
    assert metadata["ratings_source"].startswith("local:")
    assert metadata["mode_aliases"] == {"internal_model_rank": "kenpom"}


def test_prepare_bracket_lab_data_outputs_self_contained_placeholder_artifacts(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    challenge_payload, _, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )
    mutated_payload, original_name = _mutate_placeholder_challenge(challenge_payload)
    ratings_path = tmp_path / "ratings.csv"
    _write_name_ratings(ratings_path, synthetic_input_dir, replacement_name=original_name)

    raw_dir = tmp_path / "raw_lab_placeholder"
    raw_dir.mkdir()
    (raw_dir / "aliases.csv").write_text(
        "alias,team_id\nMiami OH,playin-m-oh\n",
        encoding="utf-8",
    )

    refresh_bracket_lab_data(
        challenge="mock-challenge-2026",
        raw_dir=raw_dir,
        challenge_provider=_build_challenge_provider(challenge_payload=mutated_payload),
        rating_source_provider=LocalRatingSourceProvider(ratings_file=ratings_path),
        fetched_at=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )

    out_dir = tmp_path / "prepared_lab"
    summary = prepare_bracket_lab_data(raw_dir=raw_dir, out_dir=out_dir)

    assert summary.teams == 64
    assert summary.games == 63
    assert summary.play_in_slots == 1

    for filename in (
        "teams.json",
        "games.json",
        "public_picks.csv",
        "ratings.csv",
        "completion_inputs.json",
        "play_in_slots.json",
        "metadata.json",
    ):
        assert (out_dir / filename).exists(), filename

    prepared = load_bracket_lab_prepared_input(out_dir)
    assert prepared.metadata["schema_version"] == "prepare-bracket-lab-data.v1"
    assert prepared.completion_inputs.mode_aliases[0].mode.value == "internal_model_rank"
    assert prepared.completion_inputs.mode_aliases[0].alias_of.value == "kenpom"
    assert all(
        mode.value not in {"ap_poll", "ncaa_net"}
        for mode in prepared.completion_inputs.available_modes
    )
    assert prepared.play_in_slots[0].placeholder_team_id == "placeholder-m-oh-smu"
    assert [candidate.team_id for candidate in prepared.play_in_slots[0].candidates] == [
        "playin-m-oh",
        "playin-smu",
    ]
    assert abs(
        sum(candidate.advancement_probability for candidate in prepared.play_in_slots[0].candidates)
        - 1.0
    ) < 1e-9

    prepared_rating_ids = {row.team_id for row in prepared.ratings}
    assert "placeholder-m-oh-smu" not in prepared_rating_ids
    assert {"playin-m-oh", "playin-smu"}.issubset(prepared_rating_ids)


def test_refresh_bracket_lab_data_is_deterministic(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    challenge_payload, _, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )
    ratings_path = tmp_path / "ratings_deterministic.csv"
    _write_name_ratings(ratings_path, synthetic_input_dir)
    fixed_now = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)

    raw_a = tmp_path / "raw_a"
    refresh_bracket_lab_data(
        challenge="mock-challenge-2026",
        raw_dir=raw_a,
        challenge_provider=_build_challenge_provider(challenge_payload=challenge_payload),
        rating_source_provider=LocalRatingSourceProvider(ratings_file=ratings_path),
        fetched_at=fixed_now,
    )

    raw_b = tmp_path / "raw_b"
    refresh_bracket_lab_data(
        challenge="mock-challenge-2026",
        raw_dir=raw_b,
        challenge_provider=_build_challenge_provider(challenge_payload=challenge_payload),
        rating_source_provider=LocalRatingSourceProvider(ratings_file=ratings_path),
        fetched_at=fixed_now,
    )

    for filename in (
        "teams.csv",
        "games.csv",
        "national_picks.csv",
        "kenpom.csv",
        "metadata.json",
    ):
        assert (raw_a / filename).read_text(encoding="utf-8") == (
            raw_b / filename
        ).read_text(encoding="utf-8")


def test_prepare_bracket_lab_data_fails_when_placeholder_candidate_cannot_resolve(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    challenge_payload, _, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )
    mutated_payload, original_name = _mutate_placeholder_challenge(challenge_payload)
    ratings_path = tmp_path / "ratings_missing_alias.csv"
    _write_name_ratings(ratings_path, synthetic_input_dir, replacement_name=original_name)

    raw_dir = tmp_path / "raw_lab_unresolved"
    refresh_bracket_lab_data(
        challenge="mock-challenge-2026",
        raw_dir=raw_dir,
        challenge_provider=_build_challenge_provider(challenge_payload=mutated_payload),
        rating_source_provider=LocalRatingSourceProvider(ratings_file=ratings_path),
        fetched_at=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="Missing ratings for tournament teams"):
        prepare_bracket_lab_data(raw_dir=raw_dir, out_dir=tmp_path / "prepared_bad")
