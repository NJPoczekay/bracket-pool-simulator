"""Typed domain models for the bracket simulator."""

from pydantic import BaseModel, Field


class Team(BaseModel):
    """Canonical team representation used by the simulation engine."""

    team_id: str
    name: str
    seed: int = Field(ge=1, le=16)


class SimulationConfig(BaseModel):
    """Core simulation parameters for deterministic runs."""

    n_sims: int = Field(gt=0)
    seed: int
