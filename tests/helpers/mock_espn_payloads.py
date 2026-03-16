from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_NATIONAL_PICK_TOTAL = 1_000


def build_mock_payloads(
    *,
    fixture_dir: Path,
    completed_game_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[str, str], str]]:
    """Build ESPN-like challenge/group payloads from synthetic fixture data."""

    teams = json.loads((fixture_dir / "teams.json").read_text(encoding="utf-8"))
    games = json.loads((fixture_dir / "games.json").read_text(encoding="utf-8"))
    entries = json.loads((fixture_dir / "entries.json").read_text(encoding="utf-8"))
    constraints = json.loads((fixture_dir / "constraints.json").read_text(encoding="utf-8"))

    teams_by_id = {team["team_id"]: team for team in teams}
    games_by_id = {game["game_id"]: game for game in games}
    winner_by_game = {row["game_id"]: row["winner_team_id"] for row in constraints}
    first_entry_picks = entries[0]["picks"] if entries else {}
    for game in games:
        game_id = game["game_id"]
        if game_id not in winner_by_game and game_id in first_entry_picks:
            winner_by_game[game_id] = first_entry_picks[game_id]

    possible_teams_by_game_id: dict[str, set[str]] = {}

    def possible_teams(game_id: str) -> set[str]:
        if game_id in possible_teams_by_game_id:
            return possible_teams_by_game_id[game_id]

        game = games_by_id[game_id]
        if game["round"] == 1:
            possible_teams_by_game_id[game_id] = {game["left_team_id"], game["right_team_id"]}
            return possible_teams_by_game_id[game_id]

        left_set = possible_teams(game["left_game_id"])
        right_set = possible_teams(game["right_game_id"])
        possible_teams_by_game_id[game_id] = set(left_set) | set(right_set)
        return possible_teams_by_game_id[game_id]

    for game_id in games_by_id:
        possible_teams(game_id)

    games_by_round: dict[int, list[dict[str, Any]]] = {}
    for game in games:
        games_by_round.setdefault(int(game["round"]), []).append(game)

    for round_number in games_by_round:
        games_by_round[round_number].sort(key=lambda game: game["game_id"])

    outcome_id_by_game_team: dict[tuple[str, str], str] = {}
    propositions: list[dict[str, Any]] = []

    for round_number in sorted(games_by_round):
        for display_order, game in enumerate(games_by_round[round_number], start=1):
            game_id = game["game_id"]
            possible_team_ids = sorted(possible_teams_by_game_id[game_id])

            if round_number == 1:
                possible_team_ids = [game["left_team_id"], game["right_team_id"]]

            possible_outcomes: list[dict[str, Any]] = []
            national_pick_counts = _build_choice_counts(
                possible_team_ids=possible_team_ids,
                teams_by_id=teams_by_id,
            )
            for matchup_position, team_id in enumerate(possible_team_ids, start=1):
                team = teams_by_id[team_id]
                outcome_id = f"{game_id}:{team_id}"
                outcome_id_by_game_team[(game_id, team_id)] = outcome_id
                pick_count = national_pick_counts[team_id]
                possible_outcomes.append(
                    {
                        "id": outcome_id,
                        "name": team["name"],
                        "displayOrder": matchup_position,
                        "matchupPosition": matchup_position,
                        "regionId": team["region"],
                        "regionSeed": team["seed"],
                        "choiceCounters": [
                            {
                                "count": pick_count,
                                "outcomeId": outcome_id,
                                "percentage": pick_count / _NATIONAL_PICK_TOTAL,
                                "scoringFormatId": 5,
                            }
                        ],
                        "mappings": [
                            {"type": "COMPETITOR_ID", "value": team_id},
                            {"type": "SEED", "value": str(team["seed"])},
                        ],
                    }
                )

            correct_outcomes: list[str] = []
            if game_id in completed_game_ids:
                winner_team_id = winner_by_game[game_id]
                correct_outcomes = [outcome_id_by_game_team[(game_id, winner_team_id)]]

            propositions.append(
                {
                    "id": game_id,
                    "scoringPeriodId": round_number,
                    "displayOrder": display_order,
                    "status": "COMPLETE" if len(correct_outcomes) == 1 else "OPEN",
                    "possibleOutcomes": possible_outcomes,
                    "correctOutcomes": correct_outcomes,
                }
            )

    challenge_payload = {
        "id": 277,
        "key": "mock-challenge-2026",
        "name": "Mock Challenge 2026",
        "state": "IN_PROGRESS",
        "propositionLockDate": 1773936900000,
        "propositionLockDatePassed": False,
        "scoringStatus": "LIVE",
        "propositions": propositions,
    }

    group_entries: list[dict[str, Any]] = []
    for entry in entries:
        picks_payload = []
        for game_id, winner_team_id in sorted(entry["picks"].items()):
            picks_payload.append(
                {
                    "propositionId": game_id,
                    "outcomesPicked": [
                        {
                            "outcomeId": outcome_id_by_game_team[(game_id, winner_team_id)],
                            "pickOrder": 0,
                        }
                    ],
                }
            )

        group_entries.append(
            {
                "id": entry["entry_id"],
                "name": entry["entry_name"],
                "picks": picks_payload,
            }
        )

    group_payload = {
        "groupId": "mock-group-2026",
        "entries": group_entries,
        "entryStats": {},
    }

    return challenge_payload, group_payload, outcome_id_by_game_team


def _build_choice_counts(
    *,
    possible_team_ids: list[str],
    teams_by_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    weights: list[int] = []
    for idx, team_id in enumerate(possible_team_ids, start=1):
        team = teams_by_id[team_id]
        seed = int(team["seed"])
        weights.append((20 - seed) * 100 + (len(possible_team_ids) - idx + 1))

    total_weight = sum(weights)
    raw_counts = [_NATIONAL_PICK_TOTAL * weight / total_weight for weight in weights]
    floor_counts = [int(value) for value in raw_counts]
    remainder = _NATIONAL_PICK_TOTAL - sum(floor_counts)
    order = sorted(
        range(len(possible_team_ids)),
        key=lambda idx: (raw_counts[idx] - floor_counts[idx], weights[idx], -idx),
        reverse=True,
    )
    for idx in order[:remainder]:
        floor_counts[idx] += 1

    return {
        team_id: floor_counts[idx]
        for idx, team_id in enumerate(possible_team_ids)
    }
