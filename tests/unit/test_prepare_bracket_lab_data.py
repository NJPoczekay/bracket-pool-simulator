from __future__ import annotations

from bracket_sim.application.prepare_bracket_lab_data import (
    _candidate_team_id,
    _materialize_play_in_slots,
    _PlayInCandidateSpec,
    _PlayInSlotSpec,
    _split_play_in_candidates,
    _validate_explicit_play_in_candidate_sources,
)
from bracket_sim.domain.models import Team
from bracket_sim.infrastructure.providers.contracts import RawRatingRow


def test_split_play_in_candidates_requires_two_slash_delimited_teams() -> None:
    assert _split_play_in_candidates("M-OH/SMU") == ["M-OH", "SMU"]


def test_candidate_team_id_is_stable_and_slugged() -> None:
    assert _candidate_team_id("M-OH") == "playin-m-oh"


def test_materialize_play_in_slots_computes_advancement_probabilities() -> None:
    slots = _materialize_play_in_slots(
        slot_specs=[
            _PlayInSlotSpec(
                game_id="g001",
                placeholder_team=Team(
                    team_id="placeholder-m-oh-smu",
                    name="M-OH/SMU",
                    seed=11,
                    region="south",
                ),
                candidates=[
                    _PlayInCandidateSpec(
                        team_id="playin-m-oh",
                        team_name="M-OH",
                        seed=11,
                        region="south",
                    ),
                    _PlayInCandidateSpec(
                        team_id="playin-smu",
                        team_name="SMU",
                        seed=11,
                        region="south",
                    ),
                ],
            )
        ],
        normalized_ratings=[
            RawRatingRow(team="playin-m-oh", rating=18.4, tempo=67.0),
            RawRatingRow(team="playin-smu", rating=20.1, tempo=69.2),
        ],
        rank_by_team_id={
            "playin-smu": 10,
            "playin-m-oh": 25,
        },
        point_spread_std_dev=11.0,
    )

    assert len(slots) == 1
    assert slots[0].placeholder_team_id == "placeholder-m-oh-smu"
    assert (
        abs(sum(candidate.advancement_probability for candidate in slots[0].candidates) - 1.0)
        < 1e-9
    )
    assert slots[0].candidates[1].advancement_probability > 0.5


def test_validate_explicit_play_in_candidate_sources_supports_curated_aliases() -> None:
    _validate_explicit_play_in_candidate_sources(
        slot_specs=[
            _PlayInSlotSpec(
                game_id="g001",
                placeholder_team=Team(
                    team_id="placeholder-pv-leh",
                    name="PV/LEH",
                    seed=16,
                    region="west",
                ),
                candidates=[
                    _PlayInCandidateSpec(
                        team_id="playin-pv",
                        team_name="PV",
                        seed=16,
                        region="west",
                    ),
                    _PlayInCandidateSpec(
                        team_id="playin-leh",
                        team_name="LEH",
                        seed=16,
                        region="west",
                    ),
                ],
            ),
            _PlayInSlotSpec(
                game_id="g002",
                placeholder_team=Team(
                    team_id="placeholder-m-oh-smu",
                    name="M-OH/SMU",
                    seed=11,
                    region="south",
                ),
                candidates=[
                    _PlayInCandidateSpec(
                        team_id="playin-m-oh",
                        team_name="M-OH",
                        seed=11,
                        region="south",
                    ),
                    _PlayInCandidateSpec(
                        team_id="playin-smu",
                        team_name="SMU",
                        seed=11,
                        region="south",
                    ),
                ],
            ),
        ],
        input_rows=[
            RawRatingRow(team="Prairie View A&M", rating=-10.69, tempo=71.0),
            RawRatingRow(team="Lehigh", rating=-10.41, tempo=66.9),
            RawRatingRow(team="Miami OH", rating=8.27, tempo=70.0),
            RawRatingRow(team="SMU", rating=20.1, tempo=69.2),
        ],
        manual_aliases=[],
    )
