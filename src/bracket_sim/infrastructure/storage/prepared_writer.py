"""Write normalized and cache artifacts for prepared datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import polars as pl

from bracket_sim.domain.models import CompletedGameConstraint, Game, PoolEntry, RatingRecord, Team


@dataclass(frozen=True)
class PreparedDataset:
    """Fully normalized datasets ready for simulation."""

    teams: list[Team]
    games: list[Game]
    entries: list[PoolEntry]
    constraints: list[CompletedGameConstraint]
    ratings: list[RatingRecord]


def write_prepared_dataset(
    *,
    raw_dir: Path,
    out_dir: Path,
    dataset: PreparedDataset,
) -> None:
    """Write prepared artifacts atomically to output directory."""

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = out_dir.parent / f".{out_dir.name}.tmp-{uuid4().hex}"
    cache_dir = staging_dir / "cache"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
        cache_dir.mkdir(parents=True, exist_ok=False)

        _write_normalized_json_csv(staging_dir=staging_dir, dataset=dataset)
        _write_cache_parquet(cache_dir=cache_dir, dataset=dataset)
        _write_manifest(cache_dir=cache_dir, raw_dir=raw_dir, dataset=dataset)

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


def _write_cache_parquet(*, cache_dir: Path, dataset: PreparedDataset) -> None:
    teams_df = pl.DataFrame(
        {
            "team_id": [team.team_id for team in dataset.teams],
            "name": [team.name for team in dataset.teams],
            "seed": [team.seed for team in dataset.teams],
            "region": [team.region for team in dataset.teams],
        }
    )
    teams_df.write_parquet(cache_dir / "teams.parquet")

    games_df = pl.DataFrame(
        {
            "game_id": [game.game_id for game in dataset.games],
            "round": [game.round for game in dataset.games],
            "left_team_id": [game.left_team_id for game in dataset.games],
            "right_team_id": [game.right_team_id for game in dataset.games],
            "left_game_id": [game.left_game_id for game in dataset.games],
            "right_game_id": [game.right_game_id for game in dataset.games],
        }
    )
    games_df.write_parquet(cache_dir / "games.parquet")

    entries_df = pl.DataFrame(
        {
            "entry_id": [entry.entry_id for entry in dataset.entries],
            "entry_name": [entry.entry_name for entry in dataset.entries],
        }
    )
    entries_df.write_parquet(cache_dir / "entries.parquet")

    picks_rows: list[dict[str, str]] = []
    for entry in dataset.entries:
        for pick in sorted(entry.picks, key=lambda row: row.game_id):
            picks_rows.append(
                {
                    "entry_id": entry.entry_id,
                    "game_id": pick.game_id,
                    "winner_team_id": pick.winner_team_id,
                }
            )
    entry_picks_df = pl.DataFrame(picks_rows)
    entry_picks_df.write_parquet(cache_dir / "entry_picks.parquet")

    constraints_rows = [
        {"game_id": constraint.game_id, "winner_team_id": constraint.winner_team_id}
        for constraint in dataset.constraints
    ]
    constraints_df = (
        pl.DataFrame(constraints_rows)
        if constraints_rows
        else pl.DataFrame(
            schema={
                "game_id": pl.String,
                "winner_team_id": pl.String,
            }
        )
    )
    constraints_df.write_parquet(cache_dir / "constraints.parquet")

    ratings_df = pl.DataFrame(
        {
            "team": [rating.team for rating in dataset.ratings],
            "rating": [rating.rating for rating in dataset.ratings],
            "tempo": [rating.tempo for rating in dataset.ratings],
        }
    )
    ratings_df.write_parquet(cache_dir / "ratings.parquet")


def _write_manifest(*, cache_dir: Path, raw_dir: Path, dataset: PreparedDataset) -> None:
    manifest = {
        "schema_version": 1,
        "counts": {
            "teams": len(dataset.teams),
            "games": len(dataset.games),
            "entries": len(dataset.entries),
            "constraints": len(dataset.constraints),
            "ratings": len(dataset.ratings),
        },
        "raw_input_hashes": _raw_input_hashes(raw_dir),
    }
    _write_json(path=cache_dir / "manifest.json", payload=manifest)


def _write_ratings_csv(path: Path, ratings: list[RatingRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["team", "rating", "tempo"])
        for rating in ratings:
            writer.writerow([rating.team, f"{rating.rating:.12g}", f"{rating.tempo:.12g}"])


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _raw_input_hashes(raw_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in sorted(
        ["teams.csv", "games.csv", "entries.json", "constraints.json", "ratings.csv", "aliases.csv"]
    ):
        file_path = raw_dir / filename
        if not file_path.exists():
            continue
        hashes[filename] = _sha256_for_file(file_path)
    return hashes


def _sha256_for_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
