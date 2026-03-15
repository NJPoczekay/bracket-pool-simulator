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
