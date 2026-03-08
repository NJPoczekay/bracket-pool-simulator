"""Bracket graph loading and validation utilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from bracket_sim.domain.models import Game, Team

_EXPECTED_ROUND_COUNTS: dict[int, int] = {1: 32, 2: 16, 3: 8, 4: 4, 5: 2, 6: 1}


@dataclass(frozen=True)
class BracketGraph:
    """Validated game graph plus derived topology helpers."""

    teams_by_id: dict[str, Team]
    games_by_id: dict[str, Game]
    topological_game_ids: list[str]
    championship_game_id: str
    parents_by_game_id: dict[str, list[str]]
    children_by_game_id: dict[str, list[str]]
    possible_teams_by_game_id: dict[str, set[str]]


def build_bracket_graph(teams: list[Team], games: list[Game]) -> BracketGraph:
    """Validate and build a complete 64-team / 63-game bracket graph."""

    if len(teams) != 64:
        msg = f"Expected exactly 64 teams, got {len(teams)}"
        raise ValueError(msg)

    teams_by_id: dict[str, Team] = {}
    for team in teams:
        if team.team_id in teams_by_id:
            msg = f"Duplicate team id: {team.team_id}"
            raise ValueError(msg)
        teams_by_id[team.team_id] = team

    if len(games) != 63:
        msg = f"Expected exactly 63 games, got {len(games)}"
        raise ValueError(msg)

    games_by_id: dict[str, Game] = {}
    round_counts: dict[int, int] = dict.fromkeys(_EXPECTED_ROUND_COUNTS, 0)
    for game in games:
        if game.game_id in games_by_id:
            msg = f"Duplicate game id: {game.game_id}"
            raise ValueError(msg)
        games_by_id[game.game_id] = game
        round_counts[game.round] = round_counts.get(game.round, 0) + 1

    if round_counts != _EXPECTED_ROUND_COUNTS:
        msg = f"Invalid round distribution: {round_counts}, expected {_EXPECTED_ROUND_COUNTS}"
        raise ValueError(msg)

    parents_by_game_id: dict[str, list[str]] = {game_id: [] for game_id in games_by_id}
    children_by_game_id: dict[str, list[str]] = {game_id: [] for game_id in games_by_id}

    for game in games_by_id.values():
        if game.round == 1:
            assert game.left_team_id is not None
            assert game.right_team_id is not None
            if game.left_team_id not in teams_by_id or game.right_team_id not in teams_by_id:
                msg = f"Round 1 game {game.game_id} references unknown team ids"
                raise ValueError(msg)
            continue

        assert game.left_game_id is not None
        assert game.right_game_id is not None
        if game.left_game_id not in games_by_id or game.right_game_id not in games_by_id:
            msg = f"Game {game.game_id} references unknown upstream games"
            raise ValueError(msg)

        left_upstream = games_by_id[game.left_game_id]
        right_upstream = games_by_id[game.right_game_id]
        if left_upstream.round != game.round - 1 or right_upstream.round != game.round - 1:
            msg = f"Game {game.game_id} must reference only round {game.round - 1} games"
            raise ValueError(msg)

        parents_by_game_id[game.left_game_id].append(game.game_id)
        parents_by_game_id[game.right_game_id].append(game.game_id)
        children_by_game_id[game.game_id] = [game.left_game_id, game.right_game_id]

    championship_games = [
        game_id for game_id, parent_ids in parents_by_game_id.items() if not parent_ids
    ]
    if len(championship_games) != 1:
        msg = f"Expected exactly one championship game, found {len(championship_games)}"
        raise ValueError(msg)

    championship_game_id = championship_games[0]
    if games_by_id[championship_game_id].round != 6:
        msg = "Championship game must be in round 6"
        raise ValueError(msg)

    topological_game_ids = _topological_sort(children_by_game_id, parents_by_game_id)
    possible_teams_by_game_id = _build_possible_teams(
        topological_game_ids=topological_game_ids,
        games_by_id=games_by_id,
        children_by_game_id=children_by_game_id,
    )

    return BracketGraph(
        teams_by_id=teams_by_id,
        games_by_id=games_by_id,
        topological_game_ids=topological_game_ids,
        championship_game_id=championship_game_id,
        parents_by_game_id=parents_by_game_id,
        children_by_game_id=children_by_game_id,
        possible_teams_by_game_id=possible_teams_by_game_id,
    )


def _topological_sort(
    children_by_game_id: dict[str, list[str]],
    parents_by_game_id: dict[str, list[str]],
) -> list[str]:
    """Return deterministic leaf-to-root topological order or raise on cycles."""

    in_degree: dict[str, int] = {
        game_id: len(children) for game_id, children in children_by_game_id.items()
    }
    ready = deque(sorted(game_id for game_id, degree in in_degree.items() if degree == 0))

    ordered: list[str] = []
    while ready:
        game_id = ready.popleft()
        ordered.append(game_id)

        for parent_id in sorted(parents_by_game_id[game_id]):
            in_degree[parent_id] -= 1
            if in_degree[parent_id] == 0:
                ready.append(parent_id)

    if len(ordered) != len(children_by_game_id):
        msg = "Bracket graph contains a cycle or unresolved dependencies"
        raise ValueError(msg)

    return ordered


def _build_possible_teams(
    topological_game_ids: list[str],
    games_by_id: dict[str, Game],
    children_by_game_id: dict[str, list[str]],
) -> dict[str, set[str]]:
    """Compute all teams that can possibly appear in each game."""

    possible: dict[str, set[str]] = {}
    for game_id in topological_game_ids:
        game = games_by_id[game_id]
        if game.round == 1:
            assert game.left_team_id is not None
            assert game.right_team_id is not None
            possible[game_id] = {game.left_team_id, game.right_team_id}
            continue

        left_child_id, right_child_id = children_by_game_id[game_id]
        possible[game_id] = set(possible[left_child_id]) | set(possible[right_child_id])

    return possible
