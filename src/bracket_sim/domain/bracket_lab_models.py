"""Typed models for Bracket Lab data preparation artifacts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bracket_sim.domain.product_models import CompletionMode


class PublicPickRecord(BaseModel):
    """One public-pick outcome row prepared for Bracket Lab."""

    model_config = ConfigDict(frozen=True)

    game_id: str = Field(min_length=1)
    round: int = Field(ge=1, le=6)
    display_order: int = Field(ge=1)
    outcome_id: str = Field(min_length=1)
    team_id: str = Field(min_length=1)
    team_name: str = Field(min_length=1)
    seed: int = Field(ge=1, le=16)
    region: str = Field(min_length=1)
    matchup_position: int = Field(ge=1)
    pick_count: int = Field(ge=0)
    pick_percentage: float = Field(ge=0.0, le=1.0)


class TournamentSeedInput(BaseModel):
    """Seed-based completion input for one team or placeholder slot."""

    model_config = ConfigDict(frozen=True)

    team_id: str = Field(min_length=1)
    team_name: str = Field(min_length=1)
    seed: int = Field(ge=1, le=16)
    region: str = Field(min_length=1)


class RankedTeamInput(BaseModel):
    """One ranked team row used by ranking-driven completion modes."""

    model_config = ConfigDict(frozen=True)

    team_id: str = Field(min_length=1)
    team_name: str = Field(min_length=1)
    rank: int = Field(ge=1)
    rating: float
    tempo: float


class CompletionModeAlias(BaseModel):
    """Document one completion-mode alias relationship."""

    model_config = ConfigDict(frozen=True)

    mode: CompletionMode
    alias_of: CompletionMode


class CompletionInputs(BaseModel):
    """All completion inputs prepared for Bracket Lab."""

    model_config = ConfigDict(frozen=True)

    available_modes: list[CompletionMode]
    mode_aliases: list[CompletionModeAlias] = Field(default_factory=list)
    tournament_seeds: list[TournamentSeedInput]
    popular_pick_source: str = Field(min_length=1)
    kenpom_rankings: list[RankedTeamInput]


class PlayInCandidate(BaseModel):
    """One synthetic play-in candidate resolved to a KenPom source row."""

    model_config = ConfigDict(frozen=True)

    team_id: str = Field(min_length=1)
    team_name: str = Field(min_length=1)
    rank: int = Field(ge=1)
    rating: float
    tempo: float
    advancement_probability: float = Field(ge=0.0, le=1.0)


class PlayInSlot(BaseModel):
    """Prepared metadata for one unresolved First Four placeholder slot."""

    model_config = ConfigDict(frozen=True)

    game_id: str = Field(min_length=1)
    placeholder_team_id: str = Field(min_length=1)
    placeholder_team_name: str = Field(min_length=1)
    seed: int = Field(ge=1, le=16)
    region: str = Field(min_length=1)
    candidates: list[PlayInCandidate] = Field(min_length=2)


from bracket_sim.domain.product_models import BracketLabBootstrap  # noqa: E402

BracketLabBootstrap.model_rebuild(
    _types_namespace={
        "CompletionInputs": CompletionInputs,
        "PlayInSlot": PlayInSlot,
    }
)
