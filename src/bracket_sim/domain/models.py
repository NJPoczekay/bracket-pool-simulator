"""Typed domain models for bracket simulation and scoring."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Team(BaseModel):
    """Canonical team representation used by simulation and scoring."""

    model_config = ConfigDict(frozen=True)

    team_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    seed: int = Field(ge=1, le=16)
    region: str = Field(min_length=1)


class Game(BaseModel):
    """Single elimination game node with explicit upstream references."""

    model_config = ConfigDict(frozen=True)

    game_id: str = Field(min_length=1)
    round: int = Field(ge=1, le=6)
    left_team_id: str | None = None
    right_team_id: str | None = None
    left_game_id: str | None = None
    right_game_id: str | None = None

    @model_validator(mode="after")
    def validate_sources(self) -> Game:
        """Enforce team sources in round 1 and game sources in later rounds."""

        if self.round == 1:
            if self.left_team_id is None or self.right_team_id is None:
                msg = f"Round 1 game {self.game_id} must define left_team_id and right_team_id"
                raise ValueError(msg)
            if self.left_game_id is not None or self.right_game_id is not None:
                msg = f"Round 1 game {self.game_id} cannot define upstream game ids"
                raise ValueError(msg)
        else:
            if self.left_game_id is None or self.right_game_id is None:
                msg = f"Round {self.round} game {self.game_id} must define upstream game ids"
                raise ValueError(msg)
            if self.left_team_id is not None or self.right_team_id is not None:
                msg = f"Round {self.round} game {self.game_id} cannot define direct team ids"
                raise ValueError(msg)
            if self.left_game_id == self.right_game_id:
                msg = f"Game {self.game_id} cannot reference the same upstream game twice"
                raise ValueError(msg)

        return self


class Bracket(BaseModel):
    """Collection wrapper for teams and games in a tournament bracket."""

    model_config = ConfigDict(frozen=True)

    teams: list[Team]
    games: list[Game]


class EntryPick(BaseModel):
    """Predicted winner for a given game by a pool entry."""

    model_config = ConfigDict(frozen=True)

    game_id: str = Field(min_length=1)
    winner_team_id: str = Field(min_length=1)


class PoolEntry(BaseModel):
    """A pool participant and their complete game-by-game picks."""

    model_config = ConfigDict(frozen=True)

    entry_id: str = Field(min_length=1)
    entry_name: str = Field(min_length=1)
    picks: list[EntryPick]

    @field_validator("picks")
    @classmethod
    def picks_must_be_unique_by_game(cls, picks: list[EntryPick]) -> list[EntryPick]:
        """Reject duplicate picks for the same game within an entry."""

        seen: set[str] = set()
        for pick in picks:
            if pick.game_id in seen:
                msg = f"Duplicate pick for game {pick.game_id}"
                raise ValueError(msg)
            seen.add(pick.game_id)
        return picks


class CompletedGameConstraint(BaseModel):
    """Known completed game outcome that must be locked in simulation."""

    model_config = ConfigDict(frozen=True)

    game_id: str = Field(min_length=1)
    winner_team_id: str = Field(min_length=1)


class RatingRecord(BaseModel):
    """Single team rating used by the win-probability model."""

    model_config = ConfigDict(frozen=True)

    team: str = Field(min_length=1)
    rating: float
    tempo: float


class RatingSnapshot(BaseModel):
    """Complete ratings set for a simulation run."""

    model_config = ConfigDict(frozen=True)

    records: list[RatingRecord]


class SimulationConfig(BaseModel):
    """Config for deterministic local simulations."""

    model_config = ConfigDict(frozen=True)

    input_dir: Path
    n_sims: int = Field(gt=0)
    seed: int
    rating_scale: float = Field(default=10.0, gt=0)


class SimulationEntryResult(BaseModel):
    """Aggregated simulation metrics for one pool entry."""

    model_config = ConfigDict(frozen=True)

    entry_id: str
    entry_name: str
    win_share: float
    average_score: float


class SimulationResult(BaseModel):
    """Serializable output for CLI table and JSON rendering."""

    model_config = ConfigDict(frozen=True)

    n_sims: int
    seed: int
    entry_results: list[SimulationEntryResult]
    champion_counts: dict[str, int]
