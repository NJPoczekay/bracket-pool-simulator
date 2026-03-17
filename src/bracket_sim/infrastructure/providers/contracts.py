"""Provider contracts and DTOs for refresh-data orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RawTeamRow:
    """Canonical raw team row written to teams.csv."""

    team_id: str
    name: str
    seed: int
    region: str


@dataclass(frozen=True)
class RawGameRow:
    """Canonical raw game row written to games.csv."""

    game_id: str
    round: int
    left_team_id: str | None
    right_team_id: str | None
    left_game_id: str | None
    right_game_id: str | None


@dataclass(frozen=True)
class RawEntryRow:
    """Canonical raw entry row written to entries.json."""

    entry_id: str
    entry_name: str
    picks: dict[str, str]


@dataclass(frozen=True)
class RawConstraintRow:
    """Canonical raw constraint row written to constraints.json."""

    game_id: str
    winner: str


@dataclass(frozen=True)
class RawRatingRow:
    """Canonical raw rating row written to ratings.csv."""

    team: str
    rating: float
    tempo: float


@dataclass(frozen=True)
class RawAliasRow:
    """Canonical raw alias row written to aliases.csv."""

    alias: str
    team_id: str


@dataclass(frozen=True)
class RawNationalPickRow:
    """Canonical national pick row written to national_picks.csv."""

    game_id: str
    round: int
    display_order: int
    outcome_id: str
    team_id: str
    team_name: str
    seed: int
    region: str
    matchup_position: int
    pick_count: int
    pick_percentage: float


@dataclass(frozen=True)
class SkippedEntry:
    """Skipped entry diagnostics captured in metadata."""

    entry_id: str | None
    entry_name: str | None
    error: str


@dataclass(frozen=True)
class ResultsData:
    """Normalized topology/results payload sourced from provider APIs."""

    teams: list[RawTeamRow]
    games: list[RawGameRow]
    constraints: list[RawConstraintRow]
    outcome_team_id_by_outcome_id: dict[str, str]
    proposition_status_counts: dict[str, int]
    correct_outcome_counts: dict[str, int]
    challenge_state: str | None
    challenge_scoring_status: str | None
    challenge_key: str | None
    api_shape_hints: dict[str, Any]
    raw_snapshot: dict[str, Any]


@dataclass(frozen=True)
class EntriesData:
    """Normalized entries payload plus retry/skip diagnostics."""

    entries: list[RawEntryRow]
    total_entries: int
    skipped_entries: list[SkippedEntry]
    retry_attempted: bool
    api_shape_hints: dict[str, Any]
    raw_snapshot: dict[str, Any]
    raw_retry_snapshot: dict[str, Any] | None


@dataclass(frozen=True)
class RatingsData:
    """Normalized ratings payload and optional alias additions."""

    ratings: list[RawRatingRow]
    aliases: list[RawAliasRow]
    source: str


@dataclass(frozen=True)
class NationalPicksData:
    """Normalized national pick payload sourced from challenge APIs."""

    rows: list[RawNationalPickRow]
    total_brackets: int
    round_counts: dict[int, int]
    challenge_id: int | None
    challenge_key: str | None
    challenge_name: str | None
    challenge_state: str | None
    proposition_lock_date: int | None
    proposition_lock_date_passed: bool | None
    source_url: str
    api_shape_hints: dict[str, Any]
    raw_snapshot: dict[str, Any]


@dataclass(frozen=True)
class ChallengeSnapshotData:
    """Combined challenge snapshot parsed once for multiple downstream consumers."""

    results: ResultsData
    national_picks: NationalPicksData


@dataclass(frozen=True)
class RatingSourceData:
    """Raw rating rows captured before team-id normalization."""

    ratings: list[RawRatingRow]
    source: str


class ResultsProvider(Protocol):
    """Provider interface for bracket topology and definitive winners."""

    def fetch_results(self) -> ResultsData:
        """Fetch and parse topology/results data."""


class EntriesProvider(Protocol):
    """Provider interface for group entry picks."""

    def fetch_entries(
        self,
        *,
        proposition_ids: set[str],
        outcome_team_id_by_outcome_id: dict[str, str],
    ) -> EntriesData:
        """Fetch and parse entry picks for the target challenge group."""


class RatingsProvider(Protocol):
    """Provider interface for team ratings snapshots."""

    def fetch_ratings(self, *, teams: list[RawTeamRow]) -> RatingsData:
        """Fetch and normalize team ratings for the current field."""


class NationalPicksProvider(Protocol):
    """Provider interface for national pick count snapshots."""

    def fetch_national_picks(self) -> NationalPicksData:
        """Fetch and normalize public national pick counts."""


class ChallengeSnapshotProvider(Protocol):
    """Provider interface for shared challenge snapshots."""

    def fetch_challenge_snapshot(self) -> ChallengeSnapshotData:
        """Fetch one challenge payload and parse results plus public picks."""


class RatingSourceProvider(Protocol):
    """Provider interface for raw ranking/rating source rows."""

    def fetch_rating_source(self) -> RatingSourceData:
        """Fetch raw rating rows without resolving them to tournament team ids."""
