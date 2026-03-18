"""Entry validation, scoring, and tie-split winner aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt

from bracket_sim.domain.bracket_graph import BracketGraph
from bracket_sim.domain.models import PoolEntry
from bracket_sim.domain.simulator import canonical_team_order

_MAX_WINS = 6
_DEFAULT_ROUND_VALUES = (1, 2, 4, 8, 16, 32)


def validate_entries(entries: list[PoolEntry], graph: BracketGraph) -> None:
    """Validate entry picks against bracket topology and participant flow."""

    seen_entry_ids: set[str] = set()
    expected_game_ids = set(graph.games_by_id)

    for entry in entries:
        if entry.entry_id in seen_entry_ids:
            msg = f"Duplicate entry id: {entry.entry_id}"
            raise ValueError(msg)
        seen_entry_ids.add(entry.entry_id)

        pick_map = {pick.game_id: pick.winner_team_id for pick in entry.picks}
        if set(pick_map) != expected_game_ids:
            missing = sorted(expected_game_ids - set(pick_map))
            extra = sorted(set(pick_map) - expected_game_ids)
            msg = (
                f"Entry {entry.entry_id} must pick all 63 games. "
                f"Missing={missing[:5]} Extra={extra[:5]}"
            )
            raise ValueError(msg)

        for game_id in graph.topological_game_ids:
            game = graph.games_by_id[game_id]
            chosen_winner = pick_map[game_id]

            if chosen_winner not in graph.possible_teams_by_game_id[game_id]:
                msg = (
                    f"Entry {entry.entry_id} pick {chosen_winner} is not possible in game {game_id}"
                )
                raise ValueError(msg)

            if game.round == 1:
                assert game.left_team_id is not None
                assert game.right_team_id is not None
                allowed = {game.left_team_id, game.right_team_id}
            else:
                left_child_id, right_child_id = graph.children_by_game_id[game_id]
                allowed = {pick_map[left_child_id], pick_map[right_child_id]}

            if chosen_winner not in allowed:
                msg = (
                    f"Entry {entry.entry_id} pick for game {game_id} is inconsistent with "
                    "its own upstream picks"
                )
                raise ValueError(msg)


def build_predicted_wins_matrix(
    entries: list[PoolEntry],
    graph: BracketGraph,
) -> tuple[list[str], list[str], npt.NDArray[np.int16]]:
    """Return predicted wins per entry/team from complete game picks."""

    team_ids = canonical_team_order(graph)
    team_index = {team_id: idx for idx, team_id in enumerate(team_ids)}

    predicted = np.zeros((len(entries), len(team_ids)), dtype=np.int16)
    entry_ids: list[str] = []
    for entry_idx, entry in enumerate(entries):
        entry_ids.append(entry.entry_id)
        for pick in entry.picks:
            predicted[entry_idx, team_index[pick.winner_team_id]] += 1

    return entry_ids, team_ids, predicted


def score_entries(
    predicted_wins: npt.NDArray[np.int16],
    actual_wins: npt.NDArray[np.int16],
    *,
    round_values: Sequence[int] = _DEFAULT_ROUND_VALUES,
    team_seeds: npt.NDArray[np.int16] | None = None,
    seed_bonus: bool = False,
) -> npt.NDArray[np.int32]:
    """Score each entry against each simulation using cumulative round values."""

    if predicted_wins.ndim != 2 or actual_wins.ndim != 2:
        msg = "Predicted and actual wins must be 2D arrays"
        raise ValueError(msg)

    n_entries, n_teams = predicted_wins.shape
    n_sims, sim_teams = actual_wins.shape
    if n_teams != sim_teams:
        msg = "Predicted and actual wins arrays must have the same team dimension"
        raise ValueError(msg)

    cumulative_round_values = _build_cumulative_round_values(round_values)
    correct_wins_lookup = np.minimum(
        np.arange(_MAX_WINS + 1)[:, None],
        np.arange(_MAX_WINS + 1)[None, :],
    )
    lookup = cumulative_round_values[correct_wins_lookup].astype(np.int32, copy=False)
    if seed_bonus:
        if team_seeds is None:
            msg = "team_seeds are required when seed_bonus is enabled"
            raise ValueError(msg)
        if team_seeds.shape != (n_teams,):
            msg = "team_seeds must have one value per team"
            raise ValueError(msg)

    scores = np.zeros((n_entries, n_sims), dtype=np.int32)
    for team_idx in range(n_teams):
        predicted_team = predicted_wins[:, team_idx][:, None]
        actual_team = actual_wins[:, team_idx][None, :]
        correct_wins = np.minimum(predicted_team, actual_team).astype(np.int32, copy=False)
        scores += lookup[predicted_team, actual_team]
        if seed_bonus:
            assert team_seeds is not None
            scores += correct_wins * int(team_seeds[team_idx])

    return scores


def _build_cumulative_round_values(round_values: Sequence[int]) -> npt.NDArray[np.int32]:
    if len(round_values) != _MAX_WINS:
        msg = f"round_values must contain {_MAX_WINS} entries"
        raise ValueError(msg)
    if any(value <= 0 for value in round_values):
        msg = "round_values must all be positive"
        raise ValueError(msg)

    cumulative = np.zeros(_MAX_WINS + 1, dtype=np.int32)
    running_total = 0
    for win_count, value in enumerate(round_values, start=1):
        running_total += int(value)
        cumulative[win_count] = running_total
    return cumulative


def aggregate_win_share_totals(scores: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
    """Compute raw tie-split first-place share totals for each entry."""

    if scores.ndim != 2:
        msg = "Scores must be a 2D array"
        raise ValueError(msg)

    max_scores = np.max(scores, axis=0)
    winners = scores == max_scores
    tie_counts = np.sum(winners, axis=0)

    shares = winners / tie_counts
    return cast(npt.NDArray[np.float64], np.sum(shares, axis=1))


def aggregate_win_shares(scores: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
    """Compute tie-split first-place shares for each entry."""

    share_totals = aggregate_win_share_totals(scores)
    return cast(npt.NDArray[np.float64], share_totals / scores.shape[1])
