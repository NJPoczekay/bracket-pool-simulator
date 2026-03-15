"""Application orchestration for refresh-data provider workflow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bracket_sim.infrastructure.providers.contracts import (
    EntriesProvider,
    RatingsProvider,
    ResultsProvider,
)
from bracket_sim.infrastructure.providers.espn_api import EspnApiProvider, parse_espn_group_url
from bracket_sim.infrastructure.providers.ratings import KenPomRatingsProvider, LocalRatingsProvider
from bracket_sim.infrastructure.storage.raw_refresh_writer import (
    RefreshedRawDataset,
    write_refreshed_raw_dataset,
)


@dataclass(frozen=True)
class RefreshDataSummary:
    """Summary information returned after successful refresh-data run."""

    output_dir: Path
    teams: int
    games: int
    entries: int
    skipped_entries: int
    constraints: int
    ratings: int
    aliases: int
    retry_attempted: bool


def refresh_data(
    *,
    group_url: str,
    raw_dir: Path,
    ratings_file: Path | None = None,
    use_kenpom: bool = False,
    min_usable_entries: int = 1,
    results_provider: ResultsProvider | None = None,
    entries_provider: EntriesProvider | None = None,
    ratings_provider: RatingsProvider | None = None,
    fetched_at: datetime | None = None,
) -> RefreshDataSummary:
    """Refresh canonical raw inputs from external providers."""

    if min_usable_entries < 1:
        msg = "min_usable_entries must be >= 1"
        raise ValueError(msg)

    ref = parse_espn_group_url(group_url)

    owned_espn_provider: EspnApiProvider | None = None
    if results_provider is None or entries_provider is None:
        owned_espn_provider = EspnApiProvider(group_url=group_url)
        if results_provider is None:
            results_provider = owned_espn_provider
        if entries_provider is None:
            entries_provider = owned_espn_provider

    owned_kenpom_provider: KenPomRatingsProvider | None = None
    if ratings_provider is None:
        if use_kenpom:
            owned_kenpom_provider = KenPomRatingsProvider()
            ratings_provider = owned_kenpom_provider
        else:
            ratings_provider = LocalRatingsProvider(ratings_file=ratings_file, fallback_dir=raw_dir)

    try:
        assert results_provider is not None
        results_data = results_provider.fetch_results()

        assert entries_provider is not None
        entries_data = entries_provider.fetch_entries(
            proposition_ids={game.game_id for game in results_data.games},
            outcome_team_id_by_outcome_id=results_data.outcome_team_id_by_outcome_id,
        )

        if len(entries_data.entries) < min_usable_entries:
            msg = (
                f"Usable entries below threshold: {len(entries_data.entries)} < "
                f"{min_usable_entries}. Skipped entries: {len(entries_data.skipped_entries)}"
            )
            raise ValueError(msg)

        assert ratings_provider is not None
        ratings_data = ratings_provider.fetch_ratings(teams=results_data.teams)
    finally:
        if owned_espn_provider is not None:
            owned_espn_provider.close()
        if owned_kenpom_provider is not None:
            owned_kenpom_provider.close()

    canonical_hash = _compute_canonical_hash(
        teams=results_data.teams,
        games=results_data.games,
        entries=entries_data.entries,
        constraints=results_data.constraints,
        ratings=ratings_data.ratings,
        aliases=ratings_data.aliases,
    )

    fetched_at_value = fetched_at or datetime.now(UTC)
    metadata = _build_metadata(
        fetched_at=fetched_at_value,
        group_url=group_url,
        challenge_key=results_data.challenge_key or ref.challenge_key,
        group_id=ref.group_id,
        counts={
            "teams": len(results_data.teams),
            "games": len(results_data.games),
            "entries": len(entries_data.entries),
            "constraints": len(results_data.constraints),
            "ratings": len(ratings_data.ratings),
            "aliases": len(ratings_data.aliases),
        },
        proposition_status_counts=results_data.proposition_status_counts,
        correct_outcome_counts=results_data.correct_outcome_counts,
        challenge_state=results_data.challenge_state,
        challenge_scoring_status=results_data.challenge_scoring_status,
        entries_total=entries_data.total_entries,
        entries_retry_attempted=entries_data.retry_attempted,
        skipped_entries=[
            {
                "entry_id": skipped.entry_id,
                "entry_name": skipped.entry_name,
                "error": skipped.error,
            }
            for skipped in entries_data.skipped_entries
        ],
        ratings_source=ratings_data.source,
        api_shape_hints={
            "challenge": results_data.api_shape_hints,
            "group": entries_data.api_shape_hints,
        },
        canonical_hash=canonical_hash,
    )

    snapshots: dict[str, dict[str, Any]] = {
        "challenge": results_data.raw_snapshot,
        "group": entries_data.raw_snapshot,
    }
    if entries_data.raw_retry_snapshot is not None:
        snapshots["group_retry"] = entries_data.raw_retry_snapshot

    refreshed_dataset = RefreshedRawDataset(
        teams=results_data.teams,
        games=results_data.games,
        entries=entries_data.entries,
        constraints=results_data.constraints,
        ratings=ratings_data.ratings,
        aliases=ratings_data.aliases,
        metadata=metadata,
        snapshots=snapshots,
    )
    write_refreshed_raw_dataset(out_dir=raw_dir, dataset=refreshed_dataset)

    return RefreshDataSummary(
        output_dir=raw_dir,
        teams=len(results_data.teams),
        games=len(results_data.games),
        entries=len(entries_data.entries),
        skipped_entries=len(entries_data.skipped_entries),
        constraints=len(results_data.constraints),
        ratings=len(ratings_data.ratings),
        aliases=len(ratings_data.aliases),
        retry_attempted=entries_data.retry_attempted,
    )


def _compute_canonical_hash(
    *,
    teams: list[Any],
    games: list[Any],
    entries: list[Any],
    constraints: list[Any],
    ratings: list[Any],
    aliases: list[Any],
) -> str:
    canonical_payload = {
        "teams": [team.__dict__ for team in teams],
        "games": [game.__dict__ for game in games],
        "entries": [entry.__dict__ for entry in entries],
        "constraints": [constraint.__dict__ for constraint in constraints],
        "ratings": [rating.__dict__ for rating in ratings],
        "aliases": [alias.__dict__ for alias in aliases],
    }
    serialized = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(serialized).hexdigest()


def _build_metadata(
    *,
    fetched_at: datetime,
    group_url: str,
    challenge_key: str,
    group_id: str,
    counts: dict[str, int],
    proposition_status_counts: dict[str, int],
    correct_outcome_counts: dict[str, int],
    challenge_state: str | None,
    challenge_scoring_status: str | None,
    entries_total: int,
    entries_retry_attempted: bool,
    skipped_entries: list[dict[str, Any]],
    ratings_source: str,
    api_shape_hints: dict[str, Any],
    canonical_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": "refresh-data.v1",
        "fetched_at_utc": fetched_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "source": {
            "group_url": group_url,
            "challenge_key": challenge_key,
            "group_id": group_id,
        },
        "counts": counts,
        "challenge": {
            "state": challenge_state,
            "scoring_status": challenge_scoring_status,
            "proposition_status_counts": proposition_status_counts,
            "correct_outcome_counts": correct_outcome_counts,
        },
        "entries": {
            "payload_entries": entries_total,
            "usable_entries": counts["entries"],
            "skipped_entries": len(skipped_entries),
            "retry_attempted": entries_retry_attempted,
        },
        "skipped_entries": skipped_entries,
        "ratings_source": ratings_source,
        "api_shape_hints": api_shape_hints,
        "snapshots": {
            "challenge": "snapshots/challenge.json",
            "group": "snapshots/group.json",
            "group_retry": "snapshots/group_retry.json" if entries_retry_attempted else None,
        },
        "canonical_sha256": canonical_hash,
    }
