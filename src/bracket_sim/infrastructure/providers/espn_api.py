"""ESPN Tournament Challenge provider adapter for refresh-data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from bracket_sim.infrastructure.providers.contracts import (
    EntriesData,
    EntriesProvider,
    RawConstraintRow,
    RawEntryRow,
    RawGameRow,
    RawTeamRow,
    ResultsData,
    ResultsProvider,
    SkippedEntry,
)

_EXPECTED_ROUND_COUNTS: dict[int, int] = {1: 32, 2: 16, 3: 8, 4: 4, 5: 2, 6: 1}


@dataclass(frozen=True)
class EspnGroupReference:
    """Parsed ESPN group URL components needed for API requests."""

    challenge_key: str
    group_id: str
    group_url: str


@dataclass(frozen=True)
class _ParsedOutcome:
    outcome_id: str
    team_id: str
    team_name: str
    seed: int
    region: str
    matchup_position: int
    display_order: int


@dataclass(frozen=True)
class _ParsedProposition:
    proposition_id: str
    round_number: int
    display_order: int
    possible_team_ids: set[str]
    round_one_ordered_team_ids: list[str]
    correct_outcome_ids: list[str]
    status: str


class EspnApiProvider(ResultsProvider, EntriesProvider):
    """HTTP adapter for ESPN challenge and group APIs."""

    def __init__(
        self,
        *,
        group_url: str,
        api_base_url: str = "https://gambit-api.fantasy.espn.com",
        timeout_seconds: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._ref = parse_espn_group_url(group_url)
        self._api_base_url = api_base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_seconds, follow_redirects=True)
        self._owns_client = client is None

    def close(self) -> None:
        """Close owned HTTP resources."""

        if self._owns_client:
            self._client.close()

    def fetch_results(self) -> ResultsData:
        """Fetch challenge payload and build canonical topology + constraints."""

        payload = self._get_json(
            f"{self._api_base_url}/apis/v1/challenges/{self._ref.challenge_key}?includeAllProps=true"
        )
        return _parse_results_payload(payload)

    def fetch_entries(
        self,
        *,
        proposition_ids: set[str],
        outcome_team_id_by_outcome_id: dict[str, str],
    ) -> EntriesData:
        """Fetch group entries and parse picks with retry-once recovery."""

        group_url = (
            f"{self._api_base_url}/apis/v1/challenges/{self._ref.challenge_key}"
            f"/groups/{self._ref.group_id}?view=pagetype_group_picks"
        )
        payload = self._get_json(group_url)

        parsed_entries, failures, total_entries = _parse_entries_payload(
            payload=payload,
            proposition_ids=proposition_ids,
            outcome_team_id_by_outcome_id=outcome_team_id_by_outcome_id,
        )

        retry_attempted = False
        retry_payload: dict[str, Any] | None = None
        final_failures = failures

        if failures:
            retry_attempted = True
            retry_payload = self._get_json(group_url)

            retry_target_ids = {
                failure.entry_id
                for failure in failures
                if failure.entry_id is not None and failure.entry_id != ""
            }
            retry_entries, retry_failures, _ = _parse_entries_payload(
                payload=retry_payload,
                proposition_ids=proposition_ids,
                outcome_team_id_by_outcome_id=outcome_team_id_by_outcome_id,
                target_entry_ids=retry_target_ids,
            )

            retry_entry_by_id = {entry.entry_id: entry for entry in retry_entries}
            retry_failure_by_id = {
                failure.entry_id: failure
                for failure in retry_failures
                if failure.entry_id is not None and failure.entry_id != ""
            }

            recovered_entries: list[RawEntryRow] = []
            final_failures = []
            for failure in failures:
                if failure.entry_id and failure.entry_id in retry_entry_by_id:
                    recovered_entries.append(retry_entry_by_id[failure.entry_id])
                    continue

                if failure.entry_id and failure.entry_id in retry_failure_by_id:
                    final_failures.append(retry_failure_by_id[failure.entry_id])
                    continue

                if failure.entry_id and failure.entry_id not in retry_target_ids:
                    final_failures.append(failure)
                    continue

                if failure.entry_id:
                    final_failures.append(
                        SkippedEntry(
                            entry_id=failure.entry_id,
                            entry_name=failure.entry_name,
                            error="Entry missing from retry payload",
                        )
                    )
                    continue

                final_failures.append(failure)

            parsed_entries.extend(recovered_entries)

        parsed_entries.sort(key=lambda entry: entry.entry_id)

        entries_payload = payload.get("entries")
        if isinstance(entries_payload, list) and entries_payload:
            first_entry_keys = _sorted_dict_keys(entries_payload[0])
        else:
            first_entry_keys = []

        return EntriesData(
            entries=parsed_entries,
            total_entries=total_entries,
            skipped_entries=sorted(
                final_failures,
                key=lambda failure: (
                    failure.entry_id or "",
                    failure.entry_name or "",
                    failure.error,
                ),
            ),
            retry_attempted=retry_attempted,
            api_shape_hints={
                "group_keys": _sorted_dict_keys(payload),
                "entry_keys": first_entry_keys,
            },
            raw_snapshot=payload,
            raw_retry_snapshot=retry_payload,
        )

    def _get_json(self, url: str) -> dict[str, Any]:
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            msg = f"Failed ESPN API request: {url} ({exc})"
            raise ValueError(msg) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            msg = f"ESPN API returned invalid JSON for {url}"
            raise ValueError(msg) from exc

        if not isinstance(payload, dict):
            msg = f"ESPN API returned non-object payload for {url}"
            raise ValueError(msg)

        return payload


def parse_espn_group_url(group_url: str) -> EspnGroupReference:
    """Parse challenge key and group id from an ESPN group URL."""

    parsed = urlparse(group_url.strip())
    challenge_key = _parse_challenge_key(parsed.path)
    query = parse_qs(parsed.query)

    group_values = query.get("id")
    group_id = group_values[0].strip() if group_values else ""

    if challenge_key == "":
        msg = f"Could not parse challenge key from group URL: {group_url}"
        raise ValueError(msg)
    if group_id == "":
        msg = f"Could not parse group id from group URL: {group_url}"
        raise ValueError(msg)

    return EspnGroupReference(challenge_key=challenge_key, group_id=group_id, group_url=group_url)


def _parse_challenge_key(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    if len(parts) < 3:
        return ""

    # Expected shape: /games/{challenge_key}/group
    if parts[0] != "games":
        return ""
    if parts[2] != "group":
        return ""
    return parts[1].strip()


def _parse_results_payload(payload: dict[str, Any]) -> ResultsData:
    proposition_payloads = payload.get("propositions")
    if not isinstance(proposition_payloads, list):
        msg = "Challenge payload missing required list field 'propositions'"
        raise ValueError(msg)

    propositions: list[_ParsedProposition] = []
    teams_by_id: dict[str, RawTeamRow] = {}
    outcome_team_id_by_outcome_id: dict[str, str] = {}

    for proposition_payload in proposition_payloads:
        if not isinstance(proposition_payload, dict):
            msg = "Challenge payload contains non-object proposition"
            raise ValueError(msg)

        parsed = _parse_proposition(
            proposition_payload=proposition_payload,
            teams_by_id=teams_by_id,
            outcome_team_id_by_outcome_id=outcome_team_id_by_outcome_id,
        )
        propositions.append(parsed)

    if len(propositions) != 63:
        msg = f"Expected 63 propositions, got {len(propositions)}"
        raise ValueError(msg)

    round_counts = Counter(proposition.round_number for proposition in propositions)
    if dict(round_counts) != _EXPECTED_ROUND_COUNTS:
        msg = (
            "Unexpected proposition round distribution: "
            f"{dict(round_counts)}, expected {_EXPECTED_ROUND_COUNTS}"
        )
        raise ValueError(msg)

    games = _build_games_from_propositions(propositions)
    constraints = _build_constraints(
        propositions=propositions,
        outcome_team_id_by_outcome_id=outcome_team_id_by_outcome_id,
    )

    status_counter = Counter(
        proposition.status or "UNKNOWN"
        for proposition in propositions
    )

    correct_outcome_counts = {
        "zero": sum(
            1 for proposition in propositions if len(proposition.correct_outcome_ids) == 0
        ),
        "single": sum(
            1 for proposition in propositions if len(proposition.correct_outcome_ids) == 1
        ),
        "multi": sum(
            1 for proposition in propositions if len(proposition.correct_outcome_ids) > 1
        ),
    }

    teams = sorted(teams_by_id.values(), key=lambda team: team.team_id)

    first_prop = proposition_payloads[0] if proposition_payloads else {}
    prop_keys = _sorted_dict_keys(first_prop)

    challenge_state_raw = payload.get("state")
    challenge_state = challenge_state_raw.strip() if isinstance(challenge_state_raw, str) else None

    scoring_status_raw = payload.get("scoringStatus")
    challenge_scoring_status = (
        scoring_status_raw.strip() if isinstance(scoring_status_raw, str) else None
    )

    challenge_key_raw = payload.get("key")
    challenge_key = challenge_key_raw.strip() if isinstance(challenge_key_raw, str) else None

    return ResultsData(
        teams=teams,
        games=games,
        constraints=constraints,
        outcome_team_id_by_outcome_id=outcome_team_id_by_outcome_id,
        proposition_status_counts=dict(sorted(status_counter.items())),
        correct_outcome_counts=correct_outcome_counts,
        challenge_state=challenge_state,
        challenge_scoring_status=challenge_scoring_status,
        challenge_key=challenge_key,
        api_shape_hints={
            "challenge_keys": _sorted_dict_keys(payload),
            "proposition_keys": prop_keys,
        },
        raw_snapshot=payload,
    )


def _build_games_from_propositions(propositions: list[_ParsedProposition]) -> list[RawGameRow]:
    propositions_by_round: dict[int, list[_ParsedProposition]] = {
        round_number: sorted(
            [
                proposition
                for proposition in propositions
                if proposition.round_number == round_number
            ],
            key=lambda proposition: (proposition.display_order, proposition.proposition_id),
        )
        for round_number in _EXPECTED_ROUND_COUNTS
    }

    games_by_id: dict[str, RawGameRow] = {}

    for proposition in propositions_by_round[1]:
        if len(proposition.round_one_ordered_team_ids) != 2:
            msg = (
                f"Round 1 proposition {proposition.proposition_id} must contain exactly "
                "two ordered teams"
            )
            raise ValueError(msg)

        left_team_id, right_team_id = proposition.round_one_ordered_team_ids
        games_by_id[proposition.proposition_id] = RawGameRow(
            game_id=proposition.proposition_id,
            round=1,
            left_team_id=left_team_id,
            right_team_id=right_team_id,
            left_game_id=None,
            right_game_id=None,
        )

    for round_number in range(2, 7):
        previous_round = propositions_by_round[round_number - 1]
        current_round = propositions_by_round[round_number]

        for proposition in current_round:
            left_child, right_child = _find_children_for_proposition(
                proposition=proposition,
                candidate_children=previous_round,
            )

            games_by_id[proposition.proposition_id] = RawGameRow(
                game_id=proposition.proposition_id,
                round=round_number,
                left_team_id=None,
                right_team_id=None,
                left_game_id=left_child.proposition_id,
                right_game_id=right_child.proposition_id,
            )

    parent_count_by_game_id: Counter[str] = Counter()
    for game in games_by_id.values():
        if game.round == 1:
            continue
        assert game.left_game_id is not None
        assert game.right_game_id is not None
        parent_count_by_game_id[game.left_game_id] += 1
        parent_count_by_game_id[game.right_game_id] += 1

    for proposition in propositions:
        if proposition.round_number == 6:
            continue
        count = parent_count_by_game_id[proposition.proposition_id]
        if count != 1:
            msg = (
                f"Proposition {proposition.proposition_id} has invalid parent count {count}; "
                "cannot construct safe bracket topology"
            )
            raise ValueError(msg)

    return sorted(games_by_id.values(), key=lambda game: (game.round, game.game_id))


def _find_children_for_proposition(
    *,
    proposition: _ParsedProposition,
    candidate_children: list[_ParsedProposition],
) -> tuple[_ParsedProposition, _ParsedProposition]:
    valid_pairs: list[tuple[_ParsedProposition, _ParsedProposition]] = []

    for left_candidate, right_candidate in combinations(candidate_children, 2):
        left_set = left_candidate.possible_team_ids
        right_set = right_candidate.possible_team_ids

        if left_set & right_set:
            continue

        if left_set | right_set != proposition.possible_team_ids:
            continue

        ordered_pair = tuple(
            sorted(
                (left_candidate, right_candidate),
                key=lambda node: (node.display_order, node.proposition_id),
            )
        )
        valid_pairs.append((ordered_pair[0], ordered_pair[1]))

    unique_pairs = {
        (left.proposition_id, right.proposition_id): (left, right)
        for left, right in valid_pairs
    }

    if len(unique_pairs) != 1:
        msg = (
            f"Could not infer unique children for proposition {proposition.proposition_id} "
            f"in round {proposition.round_number} (found {len(unique_pairs)} candidates)"
        )
        raise ValueError(msg)

    return next(iter(unique_pairs.values()))


def _build_constraints(
    *,
    propositions: list[_ParsedProposition],
    outcome_team_id_by_outcome_id: dict[str, str],
) -> list[RawConstraintRow]:
    constraints: list[RawConstraintRow] = []

    for proposition in sorted(
        propositions,
        key=lambda node: (node.round_number, node.display_order, node.proposition_id),
    ):
        if len(proposition.correct_outcome_ids) != 1:
            continue

        outcome_id = proposition.correct_outcome_ids[0]
        if outcome_id not in outcome_team_id_by_outcome_id:
            msg = (
                f"Proposition {proposition.proposition_id} references unknown correct outcome "
                f"{outcome_id}"
            )
            raise ValueError(msg)

        constraints.append(
            RawConstraintRow(
                game_id=proposition.proposition_id,
                winner=outcome_team_id_by_outcome_id[outcome_id],
            )
        )

    return constraints


def _parse_proposition(
    *,
    proposition_payload: dict[str, Any],
    teams_by_id: dict[str, RawTeamRow],
    outcome_team_id_by_outcome_id: dict[str, str],
) -> _ParsedProposition:
    proposition_id = _required_text(proposition_payload, "id", context="proposition")
    round_number = _required_int(
        proposition_payload,
        "scoringPeriodId",
        context=f"proposition {proposition_id}",
    )
    display_order = _required_int(
        proposition_payload,
        "displayOrder",
        context=f"proposition {proposition_id}",
    )

    possible_outcomes_payload = proposition_payload.get("possibleOutcomes")
    if not isinstance(possible_outcomes_payload, list) or len(possible_outcomes_payload) == 0:
        msg = f"Proposition {proposition_id} missing required non-empty 'possibleOutcomes' list"
        raise ValueError(msg)

    parsed_outcomes: list[_ParsedOutcome] = []
    for outcome_payload in possible_outcomes_payload:
        if not isinstance(outcome_payload, dict):
            msg = f"Proposition {proposition_id} contains non-object outcome"
            raise ValueError(msg)

        parsed_outcome = _parse_outcome(
            proposition_id=proposition_id,
            outcome_payload=outcome_payload,
        )
        parsed_outcomes.append(parsed_outcome)

        existing_team = teams_by_id.get(parsed_outcome.team_id)
        candidate_team = RawTeamRow(
            team_id=parsed_outcome.team_id,
            name=parsed_outcome.team_name,
            seed=parsed_outcome.seed,
            region=parsed_outcome.region,
        )

        if existing_team is None:
            teams_by_id[parsed_outcome.team_id] = candidate_team
        elif existing_team != candidate_team:
            msg = (
                f"Team {parsed_outcome.team_id} has inconsistent metadata across propositions: "
                f"{existing_team} vs {candidate_team}"
            )
            raise ValueError(msg)

        outcome_team_id_by_outcome_id[parsed_outcome.outcome_id] = parsed_outcome.team_id

    possible_team_ids = {outcome.team_id for outcome in parsed_outcomes}

    ordered_outcomes = sorted(
        parsed_outcomes,
        key=lambda outcome: (outcome.matchup_position, outcome.display_order, outcome.outcome_id),
    )

    round_one_ordered_team_ids: list[str] = []
    seen_round_one_teams: set[str] = set()
    for outcome in ordered_outcomes:
        if outcome.team_id in seen_round_one_teams:
            continue
        seen_round_one_teams.add(outcome.team_id)
        round_one_ordered_team_ids.append(outcome.team_id)

    correct_outcomes_payload = proposition_payload.get("correctOutcomes")
    if correct_outcomes_payload is None:
        correct_outcome_ids = []
    elif isinstance(correct_outcomes_payload, list):
        correct_outcome_ids = [
            _clean_text(outcome_id)
            for outcome_id in correct_outcomes_payload
            if _clean_text(outcome_id) != ""
        ]
    else:
        msg = f"Proposition {proposition_id} has invalid 'correctOutcomes' field type"
        raise ValueError(msg)

    status_raw = proposition_payload.get("status")
    status = _clean_text(status_raw) or "UNKNOWN"

    return _ParsedProposition(
        proposition_id=proposition_id,
        round_number=round_number,
        display_order=display_order,
        possible_team_ids=possible_team_ids,
        round_one_ordered_team_ids=round_one_ordered_team_ids,
        correct_outcome_ids=correct_outcome_ids,
        status=status,
    )


def _parse_outcome(*, proposition_id: str, outcome_payload: dict[str, Any]) -> _ParsedOutcome:
    outcome_id = _required_text(
        outcome_payload,
        "id",
        context=f"proposition {proposition_id} outcome",
    )

    mapping_by_type = _mapping_by_type(outcome_payload)

    team_id = _clean_text(mapping_by_type.get("COMPETITOR_ID"))
    if team_id == "":
        msg = (
            f"Proposition {proposition_id} outcome {outcome_id} missing required "
            "COMPETITOR_ID mapping"
        )
        raise ValueError(msg)

    team_name = _clean_text(outcome_payload.get("name")) or _clean_text(
        outcome_payload.get("description")
    )
    if team_name == "":
        msg = f"Proposition {proposition_id} outcome {outcome_id} missing team name"
        raise ValueError(msg)

    seed = _parse_seed(
        proposition_id=proposition_id,
        outcome_id=outcome_id,
        outcome_payload=outcome_payload,
        mapping_by_type=mapping_by_type,
    )
    region = _parse_region(
        proposition_id=proposition_id,
        outcome_id=outcome_id,
        outcome_payload=outcome_payload,
    )

    matchup_position = _optional_int(outcome_payload.get("matchupPosition"))
    if matchup_position is None:
        matchup_position = 999

    display_order = _optional_int(outcome_payload.get("displayOrder"))
    if display_order is None:
        display_order = 999

    return _ParsedOutcome(
        outcome_id=outcome_id,
        team_id=team_id,
        team_name=team_name,
        seed=seed,
        region=region,
        matchup_position=matchup_position,
        display_order=display_order,
    )


def _parse_seed(
    *,
    proposition_id: str,
    outcome_id: str,
    outcome_payload: dict[str, Any],
    mapping_by_type: dict[str, str],
) -> int:
    seed_candidates = [
        mapping_by_type.get("SEED"),
        _clean_text(outcome_payload.get("regionSeed")),
    ]
    for seed_candidate in seed_candidates:
        if seed_candidate is None or seed_candidate == "":
            continue
        try:
            return int(seed_candidate)
        except ValueError:
            continue

    msg = f"Proposition {proposition_id} outcome {outcome_id} missing valid seed"
    raise ValueError(msg)


def _parse_region(*, proposition_id: str, outcome_id: str, outcome_payload: dict[str, Any]) -> str:
    region_id = _clean_text(outcome_payload.get("regionId"))
    if region_id != "":
        return region_id

    region_competitor_id = _clean_text(outcome_payload.get("regionCompetitorId"))
    if "." in region_competitor_id:
        candidate = region_competitor_id.split(".", 1)[0].strip()
        if candidate != "":
            return candidate

    msg = f"Proposition {proposition_id} outcome {outcome_id} missing region information"
    raise ValueError(msg)


def _parse_entries_payload(
    *,
    payload: dict[str, Any],
    proposition_ids: set[str],
    outcome_team_id_by_outcome_id: dict[str, str],
    target_entry_ids: set[str] | None = None,
) -> tuple[list[RawEntryRow], list[SkippedEntry], int]:
    entries_payload = payload.get("entries")
    if not isinstance(entries_payload, list):
        msg = "Group payload missing required list field 'entries'"
        raise ValueError(msg)

    parsed_entries: list[RawEntryRow] = []
    failures: list[SkippedEntry] = []

    for idx, entry_payload in enumerate(entries_payload, start=1):
        if not isinstance(entry_payload, dict):
            failures.append(
                SkippedEntry(
                    entry_id=None,
                    entry_name=None,
                    error=f"Entry row {idx} is not an object",
                )
            )
            continue

        entry_id = _clean_text(entry_payload.get("id")) or None
        entry_name = _clean_text(entry_payload.get("name")) or None

        if target_entry_ids is not None:
            if entry_id is None:
                continue
            if entry_id not in target_entry_ids:
                continue

        try:
            parsed_entries.append(
                _parse_single_entry(
                    entry_payload=entry_payload,
                    proposition_ids=proposition_ids,
                    outcome_team_id_by_outcome_id=outcome_team_id_by_outcome_id,
                )
            )
        except ValueError as exc:
            failures.append(
                SkippedEntry(entry_id=entry_id, entry_name=entry_name, error=str(exc))
            )

    return parsed_entries, failures, len(entries_payload)


def _parse_single_entry(
    *,
    entry_payload: dict[str, Any],
    proposition_ids: set[str],
    outcome_team_id_by_outcome_id: dict[str, str],
) -> RawEntryRow:
    entry_id = _required_text(entry_payload, "id", context="group entry")
    entry_name = _clean_text(entry_payload.get("name")) or entry_id

    picks_payload = entry_payload.get("picks")
    if not isinstance(picks_payload, list):
        msg = f"Entry {entry_id} missing required picks list"
        raise ValueError(msg)

    picks: dict[str, str] = {}
    for pick_payload in picks_payload:
        if not isinstance(pick_payload, dict):
            msg = f"Entry {entry_id} has non-object pick row"
            raise ValueError(msg)

        proposition_id = _required_text(
            pick_payload,
            "propositionId",
            context=f"entry {entry_id} pick",
        )
        if proposition_id not in proposition_ids:
            msg = f"Entry {entry_id} references unknown proposition {proposition_id}"
            raise ValueError(msg)

        outcome_id = _select_outcome_id(
            pick_payload,
            entry_id=entry_id,
            proposition_id=proposition_id,
        )
        if outcome_id not in outcome_team_id_by_outcome_id:
            msg = (
                f"Entry {entry_id} pick for proposition {proposition_id} references "
                f"unknown outcome {outcome_id}"
            )
            raise ValueError(msg)

        picks[proposition_id] = outcome_team_id_by_outcome_id[outcome_id]

    if set(picks) != proposition_ids:
        missing = sorted(proposition_ids - set(picks))
        extra = sorted(set(picks) - proposition_ids)
        msg = (
            f"Entry {entry_id} must contain picks for all propositions. "
            f"Missing={missing[:5]} Extra={extra[:5]}"
        )
        raise ValueError(msg)

    ordered_picks = {game_id: picks[game_id] for game_id in sorted(picks)}
    return RawEntryRow(entry_id=entry_id, entry_name=entry_name, picks=ordered_picks)


def _select_outcome_id(
    pick_payload: dict[str, Any],
    *,
    entry_id: str,
    proposition_id: str,
) -> str:
    outcomes_picked_payload = pick_payload.get("outcomesPicked")
    if not isinstance(outcomes_picked_payload, list) or len(outcomes_picked_payload) == 0:
        msg = f"Entry {entry_id} pick for proposition {proposition_id} is missing outcomesPicked"
        raise ValueError(msg)

    selected_outcome_id = ""
    selected_order = 1_000_000

    for outcome_payload in outcomes_picked_payload:
        if not isinstance(outcome_payload, dict):
            continue

        outcome_id = _clean_text(outcome_payload.get("outcomeId"))
        if outcome_id == "":
            continue

        pick_order = _optional_int(outcome_payload.get("pickOrder"))
        if pick_order is None:
            pick_order = 0

        if selected_outcome_id == "" or pick_order < selected_order:
            selected_outcome_id = outcome_id
            selected_order = pick_order

    if selected_outcome_id == "":
        msg = (
            f"Entry {entry_id} pick for proposition {proposition_id} does not contain "
            "a usable outcomeId"
        )
        raise ValueError(msg)

    return selected_outcome_id


def _mapping_by_type(outcome_payload: dict[str, Any]) -> dict[str, str]:
    mapping_payload = outcome_payload.get("mappings")
    if not isinstance(mapping_payload, list):
        return {}

    mapping_by_type: dict[str, str] = {}
    for mapping in mapping_payload:
        if not isinstance(mapping, dict):
            continue

        mapping_type = _clean_text(mapping.get("type"))
        mapping_value = _clean_text(mapping.get("value"))
        if mapping_type == "" or mapping_value == "":
            continue

        mapping_by_type[mapping_type] = mapping_value

    return mapping_by_type


def _required_text(payload: dict[str, Any], key: str, *, context: str) -> str:
    value = _clean_text(payload.get(key))
    if value == "":
        msg = f"{context} missing required field '{key}'"
        raise ValueError(msg)
    return value


def _required_int(payload: dict[str, Any], key: str, *, context: str) -> int:
    value = _optional_int(payload.get(key))
    if value is None:
        msg = f"{context} missing required integer field '{key}'"
        raise ValueError(msg)
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None

    if isinstance(value, int):
        return value

    text = _clean_text(value)
    if text == "":
        return None

    try:
        return int(text)
    except ValueError:
        return None


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _sorted_dict_keys(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(str(key) for key in value)
