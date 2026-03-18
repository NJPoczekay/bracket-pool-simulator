"""Internal layout helpers for the Bracket Lab browser editor."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from bracket_sim.domain.bracket_graph import BracketGraph, build_bracket_graph
from bracket_sim.domain.models import Game, Team

BracketFeedSlot = Literal["left", "right"]


@dataclass(frozen=True)
class RegionRoundLayout:
    """Ordered games for one region and one round in the editor."""

    round: int
    game_ids: list[str]


@dataclass(frozen=True)
class RegionLayout:
    """All editor groups required to render one regional mini-bracket."""

    region: str
    title: str
    champion_game_id: str
    rounds: list[RegionRoundLayout]


@dataclass(frozen=True)
class SemifinalLayout:
    """One Final Four semifinal and the regions feeding into it."""

    game_id: str
    left_region: str
    right_region: str
    left_source_game_id: str
    right_source_game_id: str


@dataclass(frozen=True)
class ParentSlotLayout:
    """Parent-game routing for one child game in the editor graph."""

    parent_game_id: str
    slot: BracketFeedSlot


@dataclass(frozen=True)
class BracketLabEditorLayout:
    """Template-only topology payload for the clickable bracket editor."""

    regions: list[RegionLayout]
    left_region_ids: list[str]
    right_region_ids: list[str]
    semifinals: list[SemifinalLayout]
    championship_game_id: str
    parent_slots: dict[str, ParentSlotLayout]


def build_bracket_lab_editor_layout(
    *,
    teams: list[Team],
    games: list[Game],
) -> BracketLabEditorLayout:
    """Derive region and Final Four layout metadata from the prepared graph."""

    graph = build_bracket_graph(teams=teams, games=games)
    regions_by_game_id = _regions_by_game_id(graph=graph)
    championship_game = graph.games_by_id[graph.championship_game_id]
    if championship_game.left_game_id is None or championship_game.right_game_id is None:
        msg = "Championship game must define left and right semifinal sources"
        raise ValueError(msg)

    semifinal_ids = [championship_game.left_game_id, championship_game.right_game_id]
    semifinals: list[SemifinalLayout] = []
    region_order: list[str] = []
    champion_game_by_region: dict[str, str] = {}
    for semifinal_id in semifinal_ids:
        semifinal_game = graph.games_by_id[semifinal_id]
        if semifinal_game.left_game_id is None or semifinal_game.right_game_id is None:
            msg = f"Semifinal game {semifinal_id} must define child games"
            raise ValueError(msg)

        left_region = _single_region(
            regions_by_game_id[semifinal_game.left_game_id],
            game_id=semifinal_game.left_game_id,
        )
        right_region = _single_region(
            regions_by_game_id[semifinal_game.right_game_id],
            game_id=semifinal_game.right_game_id,
        )
        region_order.extend([left_region, right_region])
        champion_game_by_region[left_region] = semifinal_game.left_game_id
        champion_game_by_region[right_region] = semifinal_game.right_game_id
        semifinals.append(
            SemifinalLayout(
                game_id=semifinal_id,
                left_region=left_region,
                right_region=right_region,
                left_source_game_id=semifinal_game.left_game_id,
                right_source_game_id=semifinal_game.right_game_id,
            )
        )

    if len(region_order) != 4 or len(set(region_order)) != 4:
        msg = f"Expected exactly four unique regions, found {region_order}"
        raise ValueError(msg)

    return BracketLabEditorLayout(
        regions=[
            _build_region_layout(
                graph=graph,
                region=region,
                champion_game_id=champion_game_by_region[region],
            )
            for region in region_order
        ],
        left_region_ids=region_order[:2],
        right_region_ids=region_order[2:],
        semifinals=semifinals,
        championship_game_id=graph.championship_game_id,
        parent_slots=_build_parent_slots(graph=graph),
    )


def _build_region_layout(
    *,
    graph: BracketGraph,
    region: str,
    champion_game_id: str,
) -> RegionLayout:
    buckets: dict[int, list[str]] = defaultdict(list)
    _collect_region_rounds(
        game_id=champion_game_id,
        graph=graph,
        buckets=buckets,
    )
    rounds = [
        RegionRoundLayout(round=round_number, game_ids=buckets[round_number])
        for round_number in range(1, 5)
    ]
    return RegionLayout(
        region=region,
        title=region.replace("_", " ").title(),
        champion_game_id=champion_game_id,
        rounds=rounds,
    )


def _collect_region_rounds(
    *,
    game_id: str,
    graph: BracketGraph,
    buckets: dict[int, list[str]],
) -> None:
    game = graph.games_by_id[game_id]
    if game.round > 4:
        msg = f"Region layout only supports rounds 1-4, got round {game.round} for {game_id}"
        raise ValueError(msg)

    for child_game_id in graph.children_by_game_id[game_id]:
        _collect_region_rounds(
            game_id=child_game_id,
            graph=graph,
            buckets=buckets,
        )

    buckets[game.round].append(game_id)


def _build_parent_slots(*, graph: BracketGraph) -> dict[str, ParentSlotLayout]:
    parent_slots: dict[str, ParentSlotLayout] = {}
    for game_id, parent_ids in graph.parents_by_game_id.items():
        if not parent_ids:
            continue

        parent_game_id = parent_ids[0]
        parent_game = graph.games_by_id[parent_game_id]
        slot: BracketFeedSlot
        if parent_game.left_game_id == game_id:
            slot = "left"
        elif parent_game.right_game_id == game_id:
            slot = "right"
        else:  # pragma: no cover - defensive validation against graph drift
            msg = f"Game {game_id} is not wired into declared parent {parent_game_id}"
            raise ValueError(msg)

        parent_slots[game_id] = ParentSlotLayout(
            parent_game_id=parent_game_id,
            slot=slot,
        )
    return parent_slots


def _regions_by_game_id(*, graph: BracketGraph) -> dict[str, list[str]]:
    regions_by_game_id: dict[str, list[str]] = {}
    for game_id, team_ids in graph.possible_teams_by_game_id.items():
        regions_by_game_id[game_id] = sorted(
            {
                graph.teams_by_id[team_id].region
                for team_id in team_ids
            }
        )
    return regions_by_game_id


def _single_region(regions: list[str], *, game_id: str) -> str:
    if len(regions) != 1:
        msg = f"Expected one region for game {game_id}, found {regions}"
        raise ValueError(msg)
    return regions[0]
