# mypy: ignore-errors
"""Generate a synthetic full 64-team normalized fixture for Phase 1 testing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

REGIONS = ["east", "west", "south", "midwest"]
ROUND1_PAIRINGS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]


def main() -> None:
    root = Path(__file__).resolve().parent

    teams: list[dict[str, object]] = []
    games: list[dict[str, object]] = []

    next_game_number = 1

    def new_game_id() -> str:
        nonlocal next_game_number
        game_id = f"g{next_game_number:03d}"
        next_game_number += 1
        return game_id

    region_team_ids: dict[str, dict[int, str]] = {}
    region_champion_game_ids: dict[str, str] = {}

    for region in REGIONS:
        team_ids: dict[int, str] = {}
        for seed in range(1, 17):
            team_id = f"{region}-{seed:02d}"
            team_ids[seed] = team_id
            teams.append(
                {
                    "team_id": team_id,
                    "name": f"{region.title()} Team {seed}",
                    "seed": seed,
                    "region": region,
                }
            )

        region_team_ids[region] = team_ids

        round1_game_ids: list[str] = []
        for left_seed, right_seed in ROUND1_PAIRINGS:
            game_id = new_game_id()
            round1_game_ids.append(game_id)
            games.append(
                {
                    "game_id": game_id,
                    "round": 1,
                    "left_team_id": team_ids[left_seed],
                    "right_team_id": team_ids[right_seed],
                    "left_game_id": None,
                    "right_game_id": None,
                }
            )

        previous_round_game_ids = round1_game_ids
        for round_number in (2, 3, 4):
            current_round_game_ids: list[str] = []
            for idx in range(0, len(previous_round_game_ids), 2):
                game_id = new_game_id()
                current_round_game_ids.append(game_id)
                games.append(
                    {
                        "game_id": game_id,
                        "round": round_number,
                        "left_team_id": None,
                        "right_team_id": None,
                        "left_game_id": previous_round_game_ids[idx],
                        "right_game_id": previous_round_game_ids[idx + 1],
                    }
                )
            previous_round_game_ids = current_round_game_ids

        assert len(previous_round_game_ids) == 1
        region_champion_game_ids[region] = previous_round_game_ids[0]

    semifinal_a_id = new_game_id()
    semifinal_b_id = new_game_id()
    championship_id = new_game_id()

    games.extend(
        [
            {
                "game_id": semifinal_a_id,
                "round": 5,
                "left_team_id": None,
                "right_team_id": None,
                "left_game_id": region_champion_game_ids["east"],
                "right_game_id": region_champion_game_ids["west"],
            },
            {
                "game_id": semifinal_b_id,
                "round": 5,
                "left_team_id": None,
                "right_team_id": None,
                "left_game_id": region_champion_game_ids["south"],
                "right_game_id": region_champion_game_ids["midwest"],
            },
            {
                "game_id": championship_id,
                "round": 6,
                "left_team_id": None,
                "right_team_id": None,
                "left_game_id": semifinal_a_id,
                "right_game_id": semifinal_b_id,
            },
        ]
    )

    assert len(teams) == 64
    assert len(games) == 63

    team_seed = {team["team_id"]: int(team["seed"]) for team in teams}
    ordered_games = sorted(games, key=lambda game: (int(game["round"]), str(game["game_id"])))

    def pick_winner(
        left_team_id: str,
        right_team_id: str,
        upset_probability: float,
        state: int,
    ) -> tuple[str, int]:
        left_seed = team_seed[left_team_id]
        right_seed = team_seed[right_team_id]

        if left_seed < right_seed:
            favorite, underdog = left_team_id, right_team_id
        elif right_seed < left_seed:
            favorite, underdog = right_team_id, left_team_id
        else:
            favorite, underdog = sorted([left_team_id, right_team_id])

        state = (1103515245 * state + 12345) % (2**31)
        random_unit = state / float(2**31)
        if random_unit < upset_probability:
            return underdog, state
        return favorite, state

    entries: list[dict[str, object]] = []
    strategies: list[tuple[str, str, float, int]] = [
        ("entry_chalk", "Chalk Strategy", 0.05, 101),
        ("entry_balanced", "Balanced Strategy", 0.20, 202),
        ("entry_upsets", "Upset Hunter", 0.35, 303),
        ("entry_chaos", "Chaos Theory", 0.50, 404),
        ("entry_east_bias", "East Region Bias", 0.15, 505),
        ("entry_west_bias", "West Region Bias", 0.15, 606),
        ("entry_south_bias", "South Region Bias", 0.15, 707),
        ("entry_midwest_bias", "Midwest Region Bias", 0.15, 808),
    ]

    for entry_id, entry_name, upset_probability, random_state in strategies:
        picks: dict[str, str] = {}

        for game in ordered_games:
            game_id = str(game["game_id"])
            if int(game["round"]) == 1:
                left_team_id = str(game["left_team_id"])
                right_team_id = str(game["right_team_id"])
            else:
                left_team_id = picks[str(game["left_game_id"])]
                right_team_id = picks[str(game["right_game_id"])]

            winner, random_state = pick_winner(
                left_team_id=left_team_id,
                right_team_id=right_team_id,
                upset_probability=upset_probability,
                state=random_state,
            )

            if "east_bias" in entry_id and winner.startswith("east-"):
                winner = left_team_id if left_team_id.startswith("east-") else right_team_id
            if "west_bias" in entry_id and winner.startswith("west-"):
                winner = left_team_id if left_team_id.startswith("west-") else right_team_id
            if "south_bias" in entry_id and winner.startswith("south-"):
                winner = left_team_id if left_team_id.startswith("south-") else right_team_id
            if "midwest_bias" in entry_id and winner.startswith("midwest-"):
                winner = left_team_id if left_team_id.startswith("midwest-") else right_team_id

            picks[game_id] = winner

        entries.append({"entry_id": entry_id, "entry_name": entry_name, "picks": picks})

    round1_games = [game for game in ordered_games if int(game["round"]) == 1]
    constraints: list[dict[str, str]] = []
    for game in round1_games[:4]:
        left_team_id = str(game["left_team_id"])
        right_team_id = str(game["right_team_id"])
        winner = (
            left_team_id
            if team_seed[left_team_id] < team_seed[right_team_id]
            else right_team_id
        )
        constraints.append({"game_id": str(game["game_id"]), "winner_team_id": winner})

    region_offsets = {"east": 1.2, "west": 0.8, "south": 0.4, "midwest": 0.0}
    with (root / "ratings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["team_id", "rating", "tempo"])
        for team in sorted(teams, key=lambda row: str(row["team_id"])):
            seed = int(team["seed"])
            region = str(team["region"])
            rating = 40.0 - (1.4 * seed) + region_offsets[region]
            tempo = 64.0 + ((seed + REGIONS.index(region) * 3) % 11) * 0.9
            writer.writerow([str(team["team_id"]), f"{rating:+.2f}", f"{tempo:.1f}"])

    for name, payload in (
        ("teams.json", teams),
        ("games.json", games),
        ("entries.json", entries),
        ("constraints.json", constraints),
    ):
        with (root / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
