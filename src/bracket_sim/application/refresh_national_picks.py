"""Application orchestration for public ESPN national-picks acquisition."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bracket_sim.infrastructure.providers.contracts import NationalPicksData, NationalPicksProvider
from bracket_sim.infrastructure.providers.espn_api import EspnApiProvider
from bracket_sim.infrastructure.storage.national_picks_writer import (
    NationalPicksDataset,
    write_national_picks_dataset,
)


@dataclass(frozen=True)
class RefreshNationalPicksSummary:
    """Summary information returned after successful national-picks refresh."""

    output_dir: Path
    games: int
    rows: int
    total_brackets: int


def refresh_national_picks(
    *,
    challenge: str,
    out_dir: Path,
    provider: NationalPicksProvider | None = None,
    fetched_at: datetime | None = None,
) -> RefreshNationalPicksSummary:
    """Refresh national pick-count data from ESPN and persist it locally."""

    owned_provider: EspnApiProvider | None = None
    if provider is None:
        owned_provider = EspnApiProvider(challenge=challenge)
        provider = owned_provider

    try:
        assert provider is not None
        national_picks = provider.fetch_national_picks()
    finally:
        if owned_provider is not None:
            owned_provider.close()

    fetched_at_value = fetched_at or datetime.now(UTC)
    canonical_hash = _compute_canonical_hash(
        rows=[row.__dict__ for row in national_picks.rows],
        metadata={
            "challenge_id": national_picks.challenge_id,
            "challenge_key": national_picks.challenge_key,
            "challenge_name": national_picks.challenge_name,
            "challenge_state": national_picks.challenge_state,
            "proposition_lock_date": national_picks.proposition_lock_date,
            "proposition_lock_date_passed": national_picks.proposition_lock_date_passed,
            "round_counts": national_picks.round_counts,
            "source_url": national_picks.source_url,
            "total_brackets": national_picks.total_brackets,
        },
    )

    metadata = _build_metadata(
        fetched_at=fetched_at_value,
        national_picks=national_picks,
        canonical_hash=canonical_hash,
    )
    dataset = NationalPicksDataset(
        rows=national_picks.rows,
        metadata=metadata,
        snapshots={"challenge": national_picks.raw_snapshot},
    )
    write_national_picks_dataset(out_dir=out_dir, dataset=dataset)

    return RefreshNationalPicksSummary(
        output_dir=out_dir,
        games=len({row.game_id for row in national_picks.rows}),
        rows=len(national_picks.rows),
        total_brackets=national_picks.total_brackets,
    )


def _compute_canonical_hash(*, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    serialized = json.dumps(
        {"rows": rows, "metadata": metadata},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _build_metadata(
    *,
    fetched_at: datetime,
    national_picks: NationalPicksData,
    canonical_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": "national-picks.v1",
        "challenge_id": national_picks.challenge_id,
        "challenge_key": national_picks.challenge_key,
        "challenge_name": national_picks.challenge_name,
        "challenge_state": national_picks.challenge_state,
        "fetched_at": fetched_at.astimezone(UTC).isoformat(),
        "source_url": national_picks.source_url,
        "proposition_lock_date": national_picks.proposition_lock_date,
        "proposition_lock_date_passed": national_picks.proposition_lock_date_passed,
        "games": len({row.game_id for row in national_picks.rows}),
        "rows": len(national_picks.rows),
        "total_brackets": national_picks.total_brackets,
        "round_counts": dict(sorted(national_picks.round_counts.items())),
        "api_shape_hints": national_picks.api_shape_hints,
        "canonical_hash": canonical_hash,
    }
