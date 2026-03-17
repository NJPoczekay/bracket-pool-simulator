"""Atomic writer for prepared Bracket Lab datasets."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from bracket_sim.domain.bracket_lab_models import CompletionInputs, PlayInSlot, PublicPickRecord
from bracket_sim.domain.models import CompletedGameConstraint, Game, RatingRecord, Team


@dataclass(frozen=True)
class BracketLabPreparedDataset:
    """Complete prepared Bracket Lab dataset."""

    teams: list[Team]
    games: list[Game]
    constraints: list[CompletedGameConstraint]
    public_picks: list[PublicPickRecord]
    ratings: list[RatingRecord]
    completion_inputs: CompletionInputs
    play_in_slots: list[PlayInSlot]
    metadata: dict[str, Any]


def write_bracket_lab_prepared_dataset(
    *,
    out_dir: Path,
    dataset: BracketLabPreparedDataset,
) -> None:
    """Write prepared Bracket Lab artifacts atomically."""

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = out_dir.parent / f".{out_dir.name}.tmp-{uuid4().hex}"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
        _write_dataset(staging_dir=staging_dir, dataset=dataset)

        if out_dir.exists():
            shutil.rmtree(out_dir)
        staging_dir.rename(out_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def _write_dataset(*, staging_dir: Path, dataset: BracketLabPreparedDataset) -> None:
    _write_json(staging_dir / "teams.json", [team.model_dump() for team in dataset.teams])
    _write_json(staging_dir / "games.json", [game.model_dump() for game in dataset.games])
    _write_public_picks_csv(staging_dir / "public_picks.csv", dataset.public_picks)
    _write_ratings_csv(staging_dir / "ratings.csv", dataset.ratings)
    _write_json(
        staging_dir / "completion_inputs.json",
        dataset.completion_inputs.model_dump(mode="json"),
    )
    if dataset.constraints:
        _write_json(
            staging_dir / "constraints.json",
            [constraint.model_dump() for constraint in dataset.constraints],
        )
    if dataset.play_in_slots:
        _write_json(
            staging_dir / "play_in_slots.json",
            [slot.model_dump(mode="json") for slot in dataset.play_in_slots],
        )
    _write_json(staging_dir / "metadata.json", dataset.metadata)


def _write_public_picks_csv(path: Path, rows: list[PublicPickRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "game_id",
                "round",
                "display_order",
                "outcome_id",
                "team_id",
                "team_name",
                "seed",
                "region",
                "matchup_position",
                "pick_count",
                "pick_percentage",
            ]
        )
        for row in sorted(
            rows,
            key=lambda item: (
                item.round,
                item.display_order,
                item.matchup_position,
                item.outcome_id,
            ),
        ):
            writer.writerow(
                [
                    row.game_id,
                    row.round,
                    row.display_order,
                    row.outcome_id,
                    row.team_id,
                    row.team_name,
                    row.seed,
                    row.region,
                    row.matchup_position,
                    row.pick_count,
                    f"{row.pick_percentage:.12g}",
                ]
            )


def _write_ratings_csv(path: Path, ratings: list[RatingRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["team_id", "rating", "tempo"])
        for rating in sorted(ratings, key=lambda item: item.team_id):
            writer.writerow([rating.team_id, f"{rating.rating:.12g}", f"{rating.tempo:.12g}"])


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
