from __future__ import annotations

from pathlib import Path

from bracket_sim.infrastructure.storage.bracket_lab_prepared_loader import (
    load_bracket_lab_prepared_input,
)
from bracket_sim.infrastructure.web.layout import build_bracket_lab_editor_layout


def test_build_bracket_lab_editor_layout_derives_regions_and_final_four(
    prepared_bracket_lab_dir: Path,
) -> None:
    prepared = load_bracket_lab_prepared_input(prepared_bracket_lab_dir)

    layout = build_bracket_lab_editor_layout(
        teams=prepared.teams,
        games=prepared.games,
    )

    assert [region.region for region in layout.regions] == [
        "east",
        "west",
        "south",
        "midwest",
    ]
    assert layout.left_region_ids == ["east", "west"]
    assert layout.right_region_ids == ["south", "midwest"]

    east_region = layout.regions[0]
    assert [round_layout.round for round_layout in east_region.rounds] == [1, 2, 3, 4]
    assert east_region.rounds[0].game_ids == [
        "g001",
        "g002",
        "g003",
        "g004",
        "g005",
        "g006",
        "g007",
        "g008",
    ]
    assert east_region.rounds[1].game_ids == ["g009", "g010", "g011", "g012"]
    assert east_region.rounds[2].game_ids == ["g013", "g014"]
    assert east_region.rounds[3].game_ids == ["g015"]

    assert [semifinal.game_id for semifinal in layout.semifinals] == ["g061", "g062"]
    assert layout.semifinals[0].left_region == "east"
    assert layout.semifinals[0].right_region == "west"
    assert layout.semifinals[1].left_region == "south"
    assert layout.semifinals[1].right_region == "midwest"
    assert layout.championship_game_id == "g063"


def test_build_bracket_lab_editor_layout_tracks_parent_slots(
    prepared_bracket_lab_dir: Path,
) -> None:
    prepared = load_bracket_lab_prepared_input(prepared_bracket_lab_dir)

    layout = build_bracket_lab_editor_layout(
        teams=prepared.teams,
        games=prepared.games,
    )

    assert layout.parent_slots["g001"].parent_game_id == "g009"
    assert layout.parent_slots["g001"].slot == "left"
    assert layout.parent_slots["g002"].parent_game_id == "g009"
    assert layout.parent_slots["g002"].slot == "right"
    assert layout.parent_slots["g015"].parent_game_id == "g061"
    assert layout.parent_slots["g015"].slot == "left"
    assert layout.parent_slots["g030"].parent_game_id == "g061"
    assert layout.parent_slots["g030"].slot == "right"
    assert layout.parent_slots["g061"].parent_game_id == "g063"
    assert layout.parent_slots["g061"].slot == "left"
