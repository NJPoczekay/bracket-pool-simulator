"""Write normalized artifacts for prepared datasets."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from bracket_sim.domain.models import CompletedGameConstraint, Game, PoolEntry, RatingRecord, Team


@dataclass(frozen=True)
class PreparedDataset:
    """Fully normalized datasets ready for simulation."""

    teams: list[Team]
    games: list[Game]
    entries: list[PoolEntry]
    constraints: list[CompletedGameConstraint]
    ratings: list[RatingRecord]
    metadata: dict[str, Any] | None = None


def write_prepared_dataset(
    *,
    out_dir: Path,
    dataset: PreparedDataset,
) -> None:
    """Write prepared artifacts atomically to output directory."""

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = out_dir.parent / f".{out_dir.name}.tmp-{uuid4().hex}"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
        _write_normalized_json_csv(staging_dir=staging_dir, dataset=dataset)

        if out_dir.exists():
            shutil.rmtree(out_dir)
        staging_dir.rename(out_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def _write_normalized_json_csv(*, staging_dir: Path, dataset: PreparedDataset) -> None:
    teams_payload = [team.model_dump() for team in dataset.teams]
    games_payload = [game.model_dump() for game in dataset.games]
    entries_payload = []
    for entry in dataset.entries:
        sorted_picks = sorted(entry.picks, key=lambda pick: pick.game_id)
        entries_payload.append(
            {
                "entry_id": entry.entry_id,
                "entry_name": entry.entry_name,
                "picks": {pick.game_id: pick.winner_team_id for pick in sorted_picks},
            }
        )
    constraints_payload = [constraint.model_dump() for constraint in dataset.constraints]

    _write_json(path=staging_dir / "teams.json", payload=teams_payload)
    _write_json(path=staging_dir / "games.json", payload=games_payload)
    _write_json(path=staging_dir / "entries.json", payload=entries_payload)
    _write_json(path=staging_dir / "constraints.json", payload=constraints_payload)
    _write_ratings_csv(path=staging_dir / "ratings.csv", ratings=dataset.ratings)
    if dataset.metadata is not None:
        _write_json(path=staging_dir / "metadata.json", payload=dataset.metadata)


def _write_ratings_csv(path: Path, ratings: list[RatingRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["team_id", "rating", "tempo"])
        for rating in ratings:
            writer.writerow([rating.team_id, f"{rating.rating:.12g}", f"{rating.tempo:.12g}"])


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
