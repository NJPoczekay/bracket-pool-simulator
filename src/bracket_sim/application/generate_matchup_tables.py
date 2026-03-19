"""Generate Bracket Lab matchup win-probability and value tables."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import numpy as np

from bracket_sim.application.bracket_lab_eval import (
    BRACKET_LAB_POINT_SPREAD_STD_DEV,
    build_bracket_lab_runtime,
)
from bracket_sim.domain.bracket_graph import BracketGraph
from bracket_sim.domain.bracket_lab_models import PublicPickRecord
from bracket_sim.domain.models import RatingRecord
from bracket_sim.domain.probability_model import kenpom_win_probability


@dataclass(frozen=True)
class MatchupTableRow:
    """One game/team row in the matchup and value tables."""

    game_id: str
    round: int
    display_order: int
    matchup_position: int
    game_label: str
    team_id: str
    team_name: str
    seed: int
    region: str
    win_probability: float
    public_pick_rate: float
    value: float | None


@dataclass(frozen=True)
class MatchupTablesResult:
    """Structured matchup-table output returned to the CLI layer."""

    input_dir: Path
    round_filter: int | None
    matchup_rows: list[MatchupTableRow]
    value_rows: list[MatchupTableRow]

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the result."""

        return {
            "input_dir": str(self.input_dir),
            "round": self.round_filter,
            "matchup_rows": [asdict(row) for row in self.matchup_rows],
            "value_rows": [asdict(row) for row in self.value_rows],
        }


def generate_matchup_tables(
    *,
    input_dir: Path,
    round_filter: int | None = 1,
) -> MatchupTablesResult:
    """Return deterministic matchup and value tables from prepared Bracket Lab inputs."""

    runtime = build_bracket_lab_runtime(input_dir)
    win_probabilities_by_game = _compute_game_win_probabilities(
        graph=runtime.graph,
        rating_records_by_team_id=runtime.rating_records_by_team_id,
        constraints_by_game_id=runtime.constraints_by_game_id,
        point_spread_std_dev=BRACKET_LAB_POINT_SPREAD_STD_DEV,
    )

    public_rows_by_game: dict[str, list[PublicPickRecord]] = defaultdict(list)
    for row in runtime.prepared.public_picks:
        if round_filter is not None and row.round != round_filter:
            continue
        public_rows_by_game[row.game_id].append(row)

    matchup_rows: list[MatchupTableRow] = []
    for game_id, game_rows in sorted(
        public_rows_by_game.items(),
        key=lambda item: (item[1][0].round, item[1][0].display_order, item[0]),
    ):
        game_label = _build_game_label(
            game_id=game_id,
            game_rows=game_rows,
            graph=runtime.graph,
        )
        for row in sorted(
            game_rows,
            key=lambda item: (item.matchup_position, item.team_name, item.team_id),
        ):
            win_probability = _require_game_probability(
                game_id=row.game_id,
                team_id=row.team_id,
                win_probabilities_by_game=win_probabilities_by_game,
            )
            matchup_rows.append(
                MatchupTableRow(
                    game_id=row.game_id,
                    round=row.round,
                    display_order=row.display_order,
                    matchup_position=row.matchup_position,
                    game_label=game_label,
                    team_id=row.team_id,
                    team_name=row.team_name,
                    seed=row.seed,
                    region=row.region,
                    win_probability=win_probability,
                    public_pick_rate=row.pick_percentage,
                    value=_build_value(
                        win_probability=win_probability,
                        public_pick_rate=row.pick_percentage,
                    ),
                )
            )

    value_rows = sorted(
        matchup_rows,
        key=lambda row: (
            row.value is None,
            -(row.value or 0.0),
            -row.win_probability,
            row.public_pick_rate,
            row.round,
            row.display_order,
            row.matchup_position,
            row.team_name,
            row.team_id,
        ),
    )

    return MatchupTablesResult(
        input_dir=input_dir,
        round_filter=round_filter,
        matchup_rows=matchup_rows,
        value_rows=value_rows,
    )


def _compute_game_win_probabilities(
    *,
    graph: BracketGraph,
    rating_records_by_team_id: dict[str, RatingRecord],
    constraints_by_game_id: dict[str, str],
    point_spread_std_dev: float,
) -> dict[str, dict[str, float]]:
    pairwise_cache: dict[tuple[str, str], float] = {}

    @cache
    def winner_distribution(game_id: str) -> tuple[tuple[str, float], ...]:
        constrained_winner = constraints_by_game_id.get(game_id)
        if constrained_winner is not None:
            return ((constrained_winner, 1.0),)

        game = graph.games_by_id[game_id]
        if game.round == 1:
            assert game.left_team_id is not None
            assert game.right_team_id is not None
            probability = _head_to_head_probability(
                left_team_id=game.left_team_id,
                right_team_id=game.right_team_id,
                rating_records_by_team_id=rating_records_by_team_id,
                point_spread_std_dev=point_spread_std_dev,
                pairwise_cache=pairwise_cache,
            )
            return (
                (game.left_team_id, probability),
                (game.right_team_id, 1.0 - probability),
            )

        left_game_id, right_game_id = graph.children_by_game_id[game_id]
        left_distribution = winner_distribution(left_game_id)
        right_distribution = winner_distribution(right_game_id)
        probabilities: dict[str, float] = defaultdict(float)

        for left_team_id, left_team_probability in left_distribution:
            for right_team_id, right_team_probability in right_distribution:
                matchup_probability = left_team_probability * right_team_probability
                if matchup_probability <= 0.0:
                    continue
                left_win_probability = _head_to_head_probability(
                    left_team_id=left_team_id,
                    right_team_id=right_team_id,
                    rating_records_by_team_id=rating_records_by_team_id,
                    point_spread_std_dev=point_spread_std_dev,
                    pairwise_cache=pairwise_cache,
                )
                probabilities[left_team_id] += matchup_probability * left_win_probability
                probabilities[right_team_id] += matchup_probability * (1.0 - left_win_probability)

        return tuple(sorted(probabilities.items()))

    return {
        game_id: dict(winner_distribution(game_id))
        for game_id in graph.topological_game_ids
    }


def _head_to_head_probability(
    *,
    left_team_id: str,
    right_team_id: str,
    rating_records_by_team_id: dict[str, RatingRecord],
    point_spread_std_dev: float,
    pairwise_cache: dict[tuple[str, str], float],
) -> float:
    cached = pairwise_cache.get((left_team_id, right_team_id))
    if cached is not None:
        return cached

    left = rating_records_by_team_id[left_team_id]
    right = rating_records_by_team_id[right_team_id]
    probability = float(
        kenpom_win_probability(
            left_ratings=np.asarray([left.rating], dtype=np.float64),
            right_ratings=np.asarray([right.rating], dtype=np.float64),
            left_tempos=np.asarray([left.tempo], dtype=np.float64),
            right_tempos=np.asarray([right.tempo], dtype=np.float64),
            point_spread_std_dev=point_spread_std_dev,
        )[0]
    )
    pairwise_cache[(left_team_id, right_team_id)] = probability
    pairwise_cache[(right_team_id, left_team_id)] = 1.0 - probability
    return probability


def _require_game_probability(
    *,
    game_id: str,
    team_id: str,
    win_probabilities_by_game: dict[str, dict[str, float]],
) -> float:
    probability = win_probabilities_by_game.get(game_id, {}).get(team_id)
    if probability is None:
        msg = f"Missing model win probability for team {team_id!r} in game {game_id!r}"
        raise ValueError(msg)
    return probability


def _build_value(*, win_probability: float, public_pick_rate: float) -> float | None:
    if public_pick_rate <= 0.0:
        return None
    return win_probability / public_pick_rate


def _build_game_label(
    *,
    game_id: str,
    game_rows: list[PublicPickRecord],
    graph: BracketGraph,
) -> str:
    game = graph.games_by_id[game_id]
    if game.round == 1 and game.left_team_id is not None and game.right_team_id is not None:
        left_name = graph.teams_by_id[game.left_team_id].name
        right_name = graph.teams_by_id[game.right_team_id].name
        return f"{left_name} vs {right_name}"

    if game.round == 6:
        return "Championship"
    if game.round == 5:
        return f"Final Four Game {game_rows[0].display_order}"

    regions = {row.region for row in game_rows}
    region_prefix = ""
    if len(regions) == 1:
        region_prefix = f"{next(iter(regions)).title()} "
    return f"{region_prefix}Round {game.round} Game {game_rows[0].display_order}"
