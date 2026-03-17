"""Application orchestration for Bracket Lab data preparation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bracket_sim.domain.bracket_graph import build_bracket_graph
from bracket_sim.domain.bracket_lab_models import (
    CompletionInputs,
    CompletionModeAlias,
    PlayInCandidate,
    PlayInSlot,
    PublicPickRecord,
    RankedTeamInput,
    TournamentSeedInput,
)
from bracket_sim.domain.constraints import validate_constraints
from bracket_sim.domain.models import CompletedGameConstraint, RatingRecord, Team
from bracket_sim.domain.probability_model import logistic_win_probability
from bracket_sim.domain.product_models import CompletionMode
from bracket_sim.infrastructure.providers.contracts import RawAliasRow, RawRatingRow, RawTeamRow
from bracket_sim.infrastructure.providers.ratings import normalize_rating_rows
from bracket_sim.infrastructure.storage.bracket_lab_prepared_writer import (
    BracketLabPreparedDataset,
    write_bracket_lab_prepared_dataset,
)
from bracket_sim.infrastructure.storage.bracket_lab_raw_loader import load_bracket_lab_raw_input

_DEFAULT_RATING_SCALE = 10.0


@dataclass(frozen=True)
class PrepareBracketLabDataSummary:
    """Summary information returned after successful Bracket Lab preparation."""

    output_dir: Path
    teams: int
    games: int
    constraints: int
    public_picks: int
    ratings: int
    play_in_slots: int


@dataclass(frozen=True)
class _PlayInCandidateSpec:
    team_id: str
    team_name: str
    seed: int
    region: str


@dataclass(frozen=True)
class _PlayInSlotSpec:
    game_id: str
    placeholder_team: Team
    candidates: list[_PlayInCandidateSpec]


def prepare_bracket_lab_data(*, raw_dir: Path, out_dir: Path) -> PrepareBracketLabDataSummary:
    """Prepare a self-contained Bracket Lab dataset from raw challenge + KenPom inputs."""

    if raw_dir.resolve() == out_dir.resolve():
        msg = "--raw and --out must point to different directories"
        raise ValueError(msg)

    raw = load_bracket_lab_raw_input(raw_dir)
    teams = sorted(raw.teams, key=lambda team: team.team_id)
    games = sorted(raw.games, key=lambda game: (game.round, game.game_id))
    graph = build_bracket_graph(teams=teams, games=games)

    constraints = [
        CompletedGameConstraint(game_id=row.game_id, winner_team_id=row.winner)
        for row in sorted(raw.constraints, key=lambda item: item.game_id)
    ]
    validate_constraints(constraints=constraints, graph=graph)

    public_picks = sorted(
        raw.public_picks,
        key=lambda row: (row.round, row.display_order, row.matchup_position, row.outcome_id),
    )
    _validate_public_pick_rows(public_picks=public_picks, teams=teams)

    play_in_slots = _build_play_in_slot_specs(teams=teams, games=games)
    concrete_teams = [team for team in teams if not _is_placeholder_team_id(team.team_id)]
    synthetic_candidate_teams = [
        RawTeamRow(
            team_id=candidate.team_id,
            name=candidate.team_name,
            seed=candidate.seed,
            region=candidate.region,
        )
        for slot in play_in_slots
        for candidate in slot.candidates
    ]
    rating_target_teams = [
        RawTeamRow(team_id=team.team_id, name=team.name, seed=team.seed, region=team.region)
        for team in concrete_teams
    ] + synthetic_candidate_teams
    normalized_ratings, _ = normalize_rating_rows(
        input_rows=raw.kenpom_rows,
        teams=rating_target_teams,
        aliases=[RawAliasRow(alias=row.alias, team_id=row.team_id) for row in raw.aliases],
    )
    rating_rows = [
        RatingRecord(team_id=row.team, rating=row.rating, tempo=row.tempo)
        for row in sorted(normalized_ratings, key=lambda row: row.team)
    ]
    team_name_by_id = {team.team_id: team.name for team in rating_target_teams}
    rank_by_team_id = _compute_rankings(
        normalized_ratings=normalized_ratings,
        team_name_by_id=team_name_by_id,
    )

    play_in_slot_models = _materialize_play_in_slots(
        slot_specs=play_in_slots,
        normalized_ratings=normalized_ratings,
        rank_by_team_id=rank_by_team_id,
        rating_scale=_DEFAULT_RATING_SCALE,
    )
    completion_inputs = _build_completion_inputs(
        teams=teams,
        rating_rows=rating_rows,
        team_name_by_id=team_name_by_id,
        rank_by_team_id=rank_by_team_id,
    )
    metadata = _build_metadata(
        raw_metadata=raw.metadata,
        teams=teams,
        games=games,
        constraints=constraints,
        public_picks=public_picks,
        rating_rows=rating_rows,
        completion_inputs=completion_inputs,
        play_in_slots=play_in_slot_models,
    )
    prepared = BracketLabPreparedDataset(
        teams=teams,
        games=games,
        constraints=constraints,
        public_picks=public_picks,
        ratings=rating_rows,
        completion_inputs=completion_inputs,
        play_in_slots=play_in_slot_models,
        metadata=metadata,
    )
    write_bracket_lab_prepared_dataset(out_dir=out_dir, dataset=prepared)

    return PrepareBracketLabDataSummary(
        output_dir=out_dir,
        teams=len(teams),
        games=len(games),
        constraints=len(constraints),
        public_picks=len(public_picks),
        ratings=len(rating_rows),
        play_in_slots=len(play_in_slot_models),
    )


def _validate_public_pick_rows(*, public_picks: list[PublicPickRecord], teams: list[Team]) -> None:
    known_team_ids = {team.team_id for team in teams}
    for row in public_picks:
        if row.team_id not in known_team_ids:
            msg = f"Public pick row references unknown team_id '{row.team_id}'"
            raise ValueError(msg)


def _build_play_in_slot_specs(*, teams: list[Team], games: list[Any]) -> list[_PlayInSlotSpec]:
    team_by_id = {team.team_id: team for team in teams}
    slot_specs: list[_PlayInSlotSpec] = []

    for game in games:
        if game.round != 1:
            continue
        for team_id in (game.left_team_id, game.right_team_id):
            if team_id is None or not _is_placeholder_team_id(team_id):
                continue
            placeholder_team = team_by_id[team_id]
            candidate_names = _split_play_in_candidates(placeholder_team.name)
            candidates = [
                _PlayInCandidateSpec(
                    team_id=_candidate_team_id(candidate_name),
                    team_name=candidate_name,
                    seed=placeholder_team.seed,
                    region=placeholder_team.region,
                )
                for candidate_name in candidate_names
            ]
            slot_specs.append(
                _PlayInSlotSpec(
                    game_id=game.game_id,
                    placeholder_team=placeholder_team,
                    candidates=candidates,
                )
            )

    return sorted(
        slot_specs,
        key=lambda slot: (slot.game_id, slot.placeholder_team.team_id),
    )


def _split_play_in_candidates(placeholder_name: str) -> list[str]:
    parts = [part.strip() for part in placeholder_name.split("/") if part.strip()]
    if len(parts) != 2:
        msg = (
            "Unresolved play-in placeholder must contain exactly two slash-delimited teams: "
            f"{placeholder_name!r}"
        )
        raise ValueError(msg)
    return parts


def _candidate_team_id(candidate_name: str) -> str:
    collapsed = re.sub(r"[^a-z0-9]+", "-", candidate_name.casefold()).strip("-")
    if collapsed == "":
        msg = f"Could not synthesize play-in team id from {candidate_name!r}"
        raise ValueError(msg)
    return f"playin-{collapsed}"


def _compute_rankings(
    *,
    normalized_ratings: list[RawRatingRow],
    team_name_by_id: dict[str, str],
) -> dict[str, int]:
    ordered = sorted(
        normalized_ratings,
        key=lambda row: (
            -row.rating,
            _normalize_rank_key(team_name_by_id[row.team]),
            row.team,
        ),
    )
    return {row.team: idx for idx, row in enumerate(ordered, start=1)}


def _normalize_rank_key(value: str) -> str:
    lowered = value.casefold().replace("&", " and ")
    lowered = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _materialize_play_in_slots(
    *,
    slot_specs: list[_PlayInSlotSpec],
    normalized_ratings: list[RawRatingRow],
    rank_by_team_id: dict[str, int],
    rating_scale: float,
) -> list[PlayInSlot]:
    ratings_by_team_id = {row.team: row for row in normalized_ratings}
    slots: list[PlayInSlot] = []

    for slot in slot_specs:
        candidate_ratings = []
        for candidate in slot.candidates:
            if candidate.team_id not in ratings_by_team_id:
                msg = f"Missing rating for play-in candidate '{candidate.team_name}'"
                raise ValueError(msg)
            candidate_ratings.append(ratings_by_team_id[candidate.team_id])

        left_probability = float(
            logistic_win_probability(
                np.array([candidate_ratings[0].rating], dtype=np.float64),
                np.array([candidate_ratings[1].rating], dtype=np.float64),
                rating_scale=rating_scale,
            )[0]
        )
        probabilities = [left_probability, 1.0 - left_probability]
        candidates = [
            PlayInCandidate(
                team_id=slot.candidates[idx].team_id,
                team_name=slot.candidates[idx].team_name,
                rank=rank_by_team_id[slot.candidates[idx].team_id],
                rating=candidate_ratings[idx].rating,
                tempo=candidate_ratings[idx].tempo,
                advancement_probability=probabilities[idx],
            )
            for idx in range(2)
        ]
        slots.append(
            PlayInSlot(
                game_id=slot.game_id,
                placeholder_team_id=slot.placeholder_team.team_id,
                placeholder_team_name=slot.placeholder_team.name,
                seed=slot.placeholder_team.seed,
                region=slot.placeholder_team.region,
                candidates=candidates,
            )
        )

    return slots


def _build_completion_inputs(
    *,
    teams: list[Team],
    rating_rows: list[RatingRecord],
    team_name_by_id: dict[str, str],
    rank_by_team_id: dict[str, int],
) -> CompletionInputs:
    sorted_seed_inputs = sorted(teams, key=lambda team: (team.region, team.seed, team.team_id))
    sorted_kenpom_rows = sorted(
        rating_rows,
        key=lambda row: (rank_by_team_id[row.team_id], row.team_id),
    )
    return CompletionInputs(
        available_modes=[
            CompletionMode.TOURNAMENT_SEEDS,
            CompletionMode.POPULAR_PICKS,
            CompletionMode.KENPOM,
            CompletionMode.INTERNAL_MODEL_RANK,
        ],
        mode_aliases=[
            CompletionModeAlias(
                mode=CompletionMode.INTERNAL_MODEL_RANK,
                alias_of=CompletionMode.KENPOM,
            )
        ],
        tournament_seeds=[
            TournamentSeedInput(
                team_id=team.team_id,
                team_name=team.name,
                seed=team.seed,
                region=team.region,
            )
            for team in sorted_seed_inputs
        ],
        popular_pick_source="public_picks.csv",
        kenpom_rankings=[
            RankedTeamInput(
                team_id=row.team_id,
                team_name=team_name_by_id[row.team_id],
                rank=rank_by_team_id[row.team_id],
                rating=row.rating,
                tempo=row.tempo,
            )
            for row in sorted_kenpom_rows
        ],
    )


def _build_metadata(
    *,
    raw_metadata: dict[str, Any],
    teams: list[Team],
    games: list[Any],
    constraints: list[CompletedGameConstraint],
    public_picks: list[PublicPickRecord],
    rating_rows: list[RatingRecord],
    completion_inputs: CompletionInputs,
    play_in_slots: list[PlayInSlot],
) -> dict[str, Any]:
    canonical_hash = _compute_canonical_hash(
        teams=teams,
        games=games,
        constraints=constraints,
        public_picks=public_picks,
        rating_rows=rating_rows,
        completion_inputs=completion_inputs,
        play_in_slots=play_in_slots,
    )
    return {
        "schema_version": "prepare-bracket-lab-data.v1",
        "source": raw_metadata.get("source", {}),
        "counts": {
            "teams": len(teams),
            "games": len(games),
            "constraints": len(constraints),
            "public_picks": len(public_picks),
            "ratings": len(rating_rows),
            "play_in_slots": len(play_in_slots),
        },
        "national_picks": raw_metadata.get("national_picks", {}),
        "ratings_source": raw_metadata.get("ratings_source"),
        "completion_modes": [mode.value for mode in completion_inputs.available_modes],
        "mode_aliases": {
            alias.mode.value: alias.alias_of.value
            for alias in completion_inputs.mode_aliases
        },
        "canonical_sha256": canonical_hash,
    }


def _compute_canonical_hash(
    *,
    teams: list[Any],
    games: list[Any],
    constraints: list[Any],
    public_picks: list[Any],
    rating_rows: list[Any],
    completion_inputs: CompletionInputs,
    play_in_slots: list[Any],
) -> str:
    payload = {
        "teams": [row.model_dump(mode="json") for row in teams],
        "games": [row.model_dump(mode="json") for row in games],
        "constraints": [row.model_dump(mode="json") for row in constraints],
        "public_picks": [row.model_dump(mode="json") for row in public_picks],
        "ratings": [row.model_dump(mode="json") for row in rating_rows],
        "completion_inputs": completion_inputs.model_dump(mode="json"),
        "play_in_slots": [row.model_dump(mode="json") for row in play_in_slots],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_placeholder_team_id(team_id: str) -> bool:
    return team_id.startswith("placeholder-")
