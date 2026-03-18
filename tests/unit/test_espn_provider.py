from __future__ import annotations

import copy
from pathlib import Path
from typing import cast

import httpx
import pytest
from tests.helpers.mock_espn_payloads import build_mock_payloads

from bracket_sim.infrastructure.providers.espn_api import (
    EspnApiProvider,
    parse_espn_challenge_reference,
    parse_espn_group_url,
)


def _build_provider(
    *,
    challenge_payload: dict[str, object],
    group_payloads: list[dict[str, object]],
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


def test_parse_espn_group_url_success() -> None:
    ref = parse_espn_group_url(
        "https://fantasy.espn.com/games/tournament-challenge-bracket-2025/"
        "group?id=17ec8225-d281-4abd-be6c-6f0299c4f596"
    )
    assert ref.challenge_key == "tournament-challenge-bracket-2025"
    assert ref.group_id == "17ec8225-d281-4abd-be6c-6f0299c4f596"


def test_parse_espn_group_url_requires_group_id() -> None:
    with pytest.raises(ValueError, match="group id"):
        parse_espn_group_url("https://fantasy.espn.com/games/tournament-challenge-bracket-2025/group")


@pytest.mark.parametrize(
    ("challenge", "expected_key"),
    [
        (
            "https://fantasy.espn.com/games/tournament-challenge-bracket-2026/bracket",
            "tournament-challenge-bracket-2026",
        ),
        (
            "https://fantasy.espn.com/games/tournament-challenge-bracket-2026/"
            "group?id=test-group",
            "tournament-challenge-bracket-2026",
        ),
        ("tournament-challenge-bracket-2026", "tournament-challenge-bracket-2026"),
    ],
)
def test_parse_espn_challenge_reference_accepts_urls_and_keys(
    challenge: str,
    expected_key: str,
) -> None:
    ref = parse_espn_challenge_reference(challenge)
    assert ref.challenge_key == expected_key
    assert ref.challenge_url == (
        f"https://fantasy.espn.com/games/{expected_key}/bracket"
    )


@pytest.mark.parametrize(
    ("completed_round_cutoff", "expected_constraints"),
    [
        (0, 0),
        (1, 32),
        (6, 63),
    ],
)
def test_fetch_results_extracts_constraints_by_correct_outcomes(
    synthetic_input_dir: Path,
    completed_round_cutoff: int,
    expected_constraints: int,
) -> None:
    games = _load_games(synthetic_input_dir)
    completed_game_ids: set[str] = set()
    for game in games:
        if int(str(game["round"])) <= completed_round_cutoff:
            completed_game_ids.add(str(game["game_id"]))

    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=completed_game_ids,
    )

    provider = _build_provider(
        challenge_payload=challenge_payload,
        group_payloads=[group_payload],
    )

    results = provider.fetch_results()
    assert len(results.constraints) == expected_constraints
    assert len(results.games) == 63
    assert len(results.teams) == 64


def test_fetch_national_picks_extracts_all_round_rows(synthetic_input_dir: Path) -> None:
    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )

    provider = _build_provider(
        challenge_payload=challenge_payload,
        group_payloads=[group_payload],
    )

    national_picks = provider.fetch_national_picks()

    assert len(national_picks.rows) == 384
    assert national_picks.total_brackets == 1_000
    assert national_picks.round_counts == {1: 32, 2: 16, 3: 8, 4: 4, 5: 2, 6: 1}
    assert national_picks.challenge_key == "mock-challenge-2026"
    assert national_picks.challenge_name == "Mock Challenge 2026"
    assert national_picks.source_url == "https://fantasy.espn.com/games/mock-challenge-2026/bracket"

    championship_rows = [row for row in national_picks.rows if row.round == 6]
    assert len(championship_rows) == 64
    assert sum(row.pick_count for row in championship_rows) == 1_000


def test_fetch_challenge_snapshot_parses_results_and_public_picks_from_one_payload(
    synthetic_input_dir: Path,
) -> None:
    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )

    provider = _build_provider(
        challenge_payload=challenge_payload,
        group_payloads=[group_payload],
    )

    snapshot = provider.fetch_challenge_snapshot()

    assert len(snapshot.results.games) == 63
    assert len(snapshot.results.teams) == 64
    assert len(snapshot.national_picks.rows) == 384
    assert snapshot.results.raw_snapshot == snapshot.national_picks.raw_snapshot


def test_fetch_national_picks_falls_back_for_placeholder_team_ids(
    synthetic_input_dir: Path,
) -> None:
    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )

    broken_payload = copy.deepcopy(challenge_payload)
    assert isinstance(broken_payload["propositions"], list)
    assert isinstance(broken_payload["propositions"][0], dict)
    assert isinstance(broken_payload["propositions"][0]["possibleOutcomes"], list)
    assert isinstance(broken_payload["propositions"][0]["possibleOutcomes"][0], dict)
    broken_payload["propositions"][0]["possibleOutcomes"][0]["name"] = "M-OH/SMU"
    broken_payload["propositions"][0]["possibleOutcomes"][0]["description"] = "M-OH/SMU"
    broken_payload["propositions"][0]["possibleOutcomes"][0]["abbrev"] = "M-OH/SMU"
    broken_payload["propositions"][0]["possibleOutcomes"][0]["mappings"] = [
        mapping
        for mapping in broken_payload["propositions"][0]["possibleOutcomes"][0]["mappings"]
        if not (
            isinstance(mapping, dict)
            and mapping.get("type") == "COMPETITOR_ID"
        )
    ]

    provider = _build_provider(
        challenge_payload=broken_payload,
        group_payloads=[group_payload],
    )

    national_picks = provider.fetch_national_picks()
    placeholder_rows = [row for row in national_picks.rows if row.team_name == "M-OH/SMU"]
    assert placeholder_rows
    assert all(row.team_id == "placeholder-m-oh-smu" for row in placeholder_rows)


def test_fetch_entries_retry_recovers_from_transient_entry_issue(synthetic_input_dir: Path) -> None:
    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )

    first_payload = copy.deepcopy(group_payload)
    assert isinstance(first_payload["entries"], list)
    first_payload["entries"][0]["picks"] = None

    provider = _build_provider(
        challenge_payload=challenge_payload,
        group_payloads=[first_payload, group_payload],
    )

    results = provider.fetch_results()
    entries = provider.fetch_entries(
        proposition_ids={game.game_id for game in results.games},
        outcome_team_id_by_outcome_id=results.outcome_team_id_by_outcome_id,
    )

    assert entries.retry_attempted is True
    assert len(entries.skipped_entries) == 0
    assert len(entries.entries) == len(group_payload["entries"])


def test_fetch_entries_retry_then_skip(synthetic_input_dir: Path) -> None:
    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )

    broken_payload = copy.deepcopy(group_payload)
    assert isinstance(broken_payload["entries"], list)
    broken_payload["entries"][0]["picks"] = None

    provider = _build_provider(
        challenge_payload=challenge_payload,
        group_payloads=[broken_payload, broken_payload],
    )

    results = provider.fetch_results()
    entries = provider.fetch_entries(
        proposition_ids={game.game_id for game in results.games},
        outcome_team_id_by_outcome_id=results.outcome_team_id_by_outcome_id,
    )

    assert entries.retry_attempted is True
    assert len(entries.entries) == len(group_payload["entries"]) - 1
    assert len(entries.skipped_entries) == 1
    assert entries.skipped_entries[0].entry_id == group_payload["entries"][0]["id"]


def test_fetch_national_picks_requires_choice_counter(synthetic_input_dir: Path) -> None:
    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )

    broken_payload = copy.deepcopy(challenge_payload)
    assert isinstance(broken_payload["propositions"], list)
    assert isinstance(broken_payload["propositions"][0], dict)
    assert isinstance(broken_payload["propositions"][0]["possibleOutcomes"], list)
    assert isinstance(broken_payload["propositions"][0]["possibleOutcomes"][0], dict)
    broken_payload["propositions"][0]["possibleOutcomes"][0]["choiceCounters"] = []

    provider = _build_provider(
        challenge_payload=broken_payload,
        group_payloads=[group_payload],
    )

    with pytest.raises(ValueError, match="choiceCounter"):
        provider.fetch_national_picks()


def test_fetch_national_picks_requires_bracket_scoring_format(synthetic_input_dir: Path) -> None:
    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )

    broken_payload = copy.deepcopy(challenge_payload)
    assert isinstance(broken_payload["propositions"], list)
    assert isinstance(broken_payload["propositions"][0], dict)
    assert isinstance(broken_payload["propositions"][0]["possibleOutcomes"], list)
    assert isinstance(broken_payload["propositions"][0]["possibleOutcomes"][0], dict)
    assert isinstance(
        broken_payload["propositions"][0]["possibleOutcomes"][0]["choiceCounters"],
        list,
    )
    assert isinstance(
        broken_payload["propositions"][0]["possibleOutcomes"][0]["choiceCounters"][0],
        dict,
    )
    broken_payload["propositions"][0]["possibleOutcomes"][0]["choiceCounters"][0][
        "scoringFormatId"
    ] = 6

    provider = _build_provider(
        challenge_payload=broken_payload,
        group_payloads=[group_payload],
    )

    with pytest.raises(ValueError, match="scoringFormatId"):
        provider.fetch_national_picks()


def test_fetch_national_picks_warns_on_inconsistent_totals(
    synthetic_input_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )

    broken_payload = copy.deepcopy(challenge_payload)
    assert isinstance(broken_payload["propositions"], list)
    assert isinstance(broken_payload["propositions"][0], dict)
    assert isinstance(broken_payload["propositions"][0]["possibleOutcomes"], list)
    assert isinstance(broken_payload["propositions"][0]["possibleOutcomes"][0], dict)
    assert isinstance(
        broken_payload["propositions"][0]["possibleOutcomes"][0]["choiceCounters"],
        list,
    )
    assert isinstance(
        broken_payload["propositions"][0]["possibleOutcomes"][0]["choiceCounters"][0],
        dict,
    )
    broken_payload["propositions"][0]["possibleOutcomes"][0]["choiceCounters"][0]["count"] = 9999

    provider = _build_provider(
        challenge_payload=broken_payload,
        group_payloads=[group_payload],
    )

    with caplog.at_level("WARNING", logger="bracket_sim.infrastructure.providers.espn_api"):
        national_picks = provider.fetch_national_picks()

    assert len(national_picks.rows) == 384
    first_game_id = national_picks.rows[0].game_id
    assert national_picks.total_brackets == sum(
        row.pick_count for row in national_picks.rows if row.game_id == first_game_id
    )
    assert "National pick totals mismatch" in caplog.text


def test_fetch_challenge_snapshot_normalizes_noncanonical_proposition_display_order(
    synthetic_input_dir: Path,
) -> None:
    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )

    broken_payload = copy.deepcopy(challenge_payload)
    assert isinstance(broken_payload["propositions"], list)
    for proposition in broken_payload["propositions"]:
        assert isinstance(proposition, dict)
        round_number = int(cast(int, proposition["scoringPeriodId"]))
        display_order = int(cast(int, proposition["displayOrder"]))
        if round_number in {1, 3, 5, 6} or (round_number == 2 and display_order % 3 == 0):
            proposition["displayOrder"] = display_order - 1
        elif round_number == 4 and display_order == 4:
            proposition["displayOrder"] = 3

    provider = _build_provider(
        challenge_payload=broken_payload,
        group_payloads=[group_payload],
    )

    snapshot = provider.fetch_challenge_snapshot()

    assert len(snapshot.results.games) == 63

    observed_orders_by_round: dict[int, set[int]] = {}
    for row in snapshot.national_picks.rows:
        observed_orders_by_round.setdefault(row.round, set()).add(row.display_order)

    assert observed_orders_by_round == {
        1: set(range(1, 33)),
        2: set(range(1, 17)),
        3: set(range(1, 9)),
        4: set(range(1, 5)),
        5: set(range(1, 3)),
        6: {1},
    }


def test_fetch_national_picks_rejects_unexpected_proposition_structure(
    synthetic_input_dir: Path,
) -> None:
    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )

    broken_payload = copy.deepcopy(challenge_payload)
    assert isinstance(broken_payload["propositions"], list)
    broken_payload["propositions"].pop()

    provider = _build_provider(
        challenge_payload=broken_payload,
        group_payloads=[group_payload],
    )

    with pytest.raises(ValueError, match="Expected 63 propositions"):
        provider.fetch_national_picks()


def _load_games(fixture_dir: Path) -> list[dict[str, object]]:
    import json

    payload = json.loads((fixture_dir / "games.json").read_text(encoding="utf-8"))
    return cast(list[dict[str, object]], payload)
