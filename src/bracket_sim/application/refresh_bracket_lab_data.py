"""Application orchestration for Bracket Lab data refresh."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bracket_sim.domain.product_models import CompletionMode
from bracket_sim.infrastructure.providers.contracts import (
    ChallengeSnapshotData,
    ChallengeSnapshotProvider,
    RatingSourceProvider,
    RawAliasRow,
    RawRatingRow,
)
from bracket_sim.infrastructure.providers.espn_api import (
    EspnApiProvider,
    EspnChallengeReference,
    parse_espn_challenge_reference,
)
from bracket_sim.infrastructure.providers.ratings import (
    KenPomRatingSourceProvider,
    LocalRatingSourceProvider,
)
from bracket_sim.infrastructure.storage._file_io import load_required_csv_rows
from bracket_sim.infrastructure.storage.bracket_lab_raw_writer import (
    BracketLabRawDataset,
    write_bracket_lab_raw_dataset,
)


@dataclass(frozen=True)
class RefreshBracketLabDataSummary:
    """Summary information returned after successful Bracket Lab refresh."""

    output_dir: Path
    teams: int
    games: int
    constraints: int
    public_pick_rows: int
    kenpom_rows: int
    aliases: int


def refresh_bracket_lab_data(
    *,
    challenge: str,
    raw_dir: Path,
    ratings_file: Path | None = None,
    use_kenpom: bool = False,
    challenge_provider: ChallengeSnapshotProvider | None = None,
    rating_source_provider: RatingSourceProvider | None = None,
    fetched_at: datetime | None = None,
) -> RefreshBracketLabDataSummary:
    """Refresh raw Bracket Lab data from ESPN challenge data plus KenPom source rows."""

    if rating_source_provider is None and not use_kenpom and ratings_file is None:
        msg = "Bracket Lab refresh requires either --ratings-file or --kenpom"
        raise ValueError(msg)

    challenge_ref = parse_espn_challenge_reference(challenge)

    owned_challenge_provider: EspnApiProvider | None = None
    if challenge_provider is None:
        owned_challenge_provider = EspnApiProvider(challenge=challenge)
        challenge_provider = owned_challenge_provider

    owned_kenpom_provider: KenPomRatingSourceProvider | None = None
    if rating_source_provider is None:
        if use_kenpom:
            owned_kenpom_provider = KenPomRatingSourceProvider()
            rating_source_provider = owned_kenpom_provider
        else:
            rating_source_provider = LocalRatingSourceProvider(ratings_file=ratings_file)

    try:
        assert challenge_provider is not None
        challenge_snapshot = challenge_provider.fetch_challenge_snapshot()

        assert rating_source_provider is not None
        rating_source = rating_source_provider.fetch_rating_source()
    finally:
        if owned_challenge_provider is not None:
            owned_challenge_provider.close()
        if owned_kenpom_provider is not None:
            owned_kenpom_provider.close()

    aliases = _load_existing_aliases(raw_dir / "aliases.csv")
    fetched_at_value = fetched_at or datetime.now(UTC)
    metadata = _build_metadata(
        fetched_at=fetched_at_value,
        challenge_ref=challenge_ref,
        challenge_snapshot=challenge_snapshot,
        ratings_source=rating_source.source,
        kenpom_rows=rating_source.ratings,
        aliases=aliases,
    )
    dataset = BracketLabRawDataset(
        teams=challenge_snapshot.results.teams,
        games=challenge_snapshot.results.games,
        constraints=challenge_snapshot.results.constraints,
        national_picks=challenge_snapshot.national_picks.rows,
        kenpom_rows=rating_source.ratings,
        aliases=aliases,
        metadata=metadata,
        snapshots={"challenge": challenge_snapshot.results.raw_snapshot},
    )
    write_bracket_lab_raw_dataset(out_dir=raw_dir, dataset=dataset)

    return RefreshBracketLabDataSummary(
        output_dir=raw_dir,
        teams=len(challenge_snapshot.results.teams),
        games=len(challenge_snapshot.results.games),
        constraints=len(challenge_snapshot.results.constraints),
        public_pick_rows=len(challenge_snapshot.national_picks.rows),
        kenpom_rows=len(rating_source.ratings),
        aliases=len(aliases),
    )


def _load_existing_aliases(path: Path) -> list[RawAliasRow]:
    if not path.exists():
        return []

    rows, fieldnames = load_required_csv_rows(path, missing_prefix="Required raw file is missing")
    expected = {"alias", "team_id"}
    if set(fieldnames) != expected:
        msg = f"{path.name} must have columns {sorted(expected)}, got {fieldnames}"
        raise ValueError(msg)

    aliases: list[RawAliasRow] = []
    for row in rows:
        alias = str(row.get("alias") or "").strip()
        team_id = str(row.get("team_id") or "").strip()
        if alias == "" or team_id == "":
            msg = f"{path.name} contains blank alias/team_id values"
            raise ValueError(msg)
        aliases.append(RawAliasRow(alias=alias, team_id=team_id))
    return aliases


def _build_metadata(
    *,
    fetched_at: datetime,
    challenge_ref: EspnChallengeReference,
    challenge_snapshot: ChallengeSnapshotData,
    ratings_source: str,
    kenpom_rows: list[RawRatingRow],
    aliases: list[RawAliasRow],
) -> dict[str, Any]:
    canonical_hash = _compute_canonical_hash(
        teams=challenge_snapshot.results.teams,
        games=challenge_snapshot.results.games,
        constraints=challenge_snapshot.results.constraints,
        national_picks=challenge_snapshot.national_picks.rows,
        kenpom_rows=kenpom_rows,
        aliases=aliases,
    )
    return {
        "schema_version": "refresh-bracket-lab-data.v1",
        "fetched_at_utc": fetched_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "source": {
            "challenge": challenge_ref.challenge_key,
            "challenge_key": (
                challenge_snapshot.results.challenge_key or challenge_ref.challenge_key
            ),
            "challenge_url": challenge_ref.challenge_url,
        },
        "counts": {
            "teams": len(challenge_snapshot.results.teams),
            "games": len(challenge_snapshot.results.games),
            "constraints": len(challenge_snapshot.results.constraints),
            "national_pick_rows": len(challenge_snapshot.national_picks.rows),
            "kenpom_rows": len(kenpom_rows),
            "aliases": len(aliases),
        },
        "challenge": {
            "state": challenge_snapshot.results.challenge_state,
            "scoring_status": challenge_snapshot.results.challenge_scoring_status,
            "proposition_status_counts": challenge_snapshot.results.proposition_status_counts,
            "correct_outcome_counts": challenge_snapshot.results.correct_outcome_counts,
        },
        "national_picks": {
            "total_brackets": challenge_snapshot.national_picks.total_brackets,
            "round_counts": dict(sorted(challenge_snapshot.national_picks.round_counts.items())),
            "challenge_state": challenge_snapshot.national_picks.challenge_state,
            "source_url": challenge_snapshot.national_picks.source_url,
        },
        "ratings_source": ratings_source,
        "completion_modes": [
            CompletionMode.TOURNAMENT_SEEDS.value,
            CompletionMode.POPULAR_PICKS.value,
            CompletionMode.KENPOM.value,
            CompletionMode.INTERNAL_MODEL_RANK.value,
        ],
        "mode_aliases": {
            CompletionMode.INTERNAL_MODEL_RANK.value: CompletionMode.KENPOM.value,
        },
        "snapshots": {
            "challenge": "snapshots/challenge.json",
        },
        "canonical_sha256": canonical_hash,
    }


def _compute_canonical_hash(
    *,
    teams: list[Any],
    games: list[Any],
    constraints: list[Any],
    national_picks: list[Any],
    kenpom_rows: list[Any],
    aliases: list[Any],
) -> str:
    payload = {
        "teams": [row.__dict__ for row in teams],
        "games": [row.__dict__ for row in games],
        "constraints": [row.__dict__ for row in constraints],
        "national_picks": [row.__dict__ for row in national_picks],
        "kenpom_rows": [row.__dict__ for row in kenpom_rows],
        "aliases": [row.__dict__ for row in aliases],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
