"""Typed domain models for bracket simulation, scoring, and run artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ALLOWED_ENGINES = {"numpy", "numba"}
_ALLOWED_LOG_LEVELS = {"debug", "info", "warning", "error"}


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

    team_id: str = Field(min_length=1)
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
    batch_size: int | None = Field(default=None, gt=0)
    run_dir: Path | None = None
    resume: bool = False
    engine: str = Field(default="numpy")
    log_level: str = Field(default="warning")

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, value: str) -> str:
        """Normalize and validate engine selection."""

        normalized = value.strip().lower()
        if normalized not in _ALLOWED_ENGINES:
            msg = f"Unsupported engine {value!r}; expected one of {sorted(_ALLOWED_ENGINES)}"
            raise ValueError(msg)
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Normalize and validate configured log level."""

        normalized = value.strip().lower()
        if normalized not in _ALLOWED_LOG_LEVELS:
            msg = (
                f"Unsupported log level {value!r}; expected one of "
                f"{sorted(_ALLOWED_LOG_LEVELS)}"
            )
            raise ValueError(msg)
        return normalized

    @model_validator(mode="after")
    def validate_runtime_options(self) -> SimulationConfig:
        """Reject invalid combinations of batch/resume options."""

        if self.resume and self.run_dir is None:
            msg = "--resume requires --run-dir so an existing checkpoint can be loaded"
            raise ValueError(msg)
        return self

    @property
    def effective_batch_size(self) -> int:
        """Return the concrete batch size used for execution."""

        return min(self.batch_size or self.n_sims, self.n_sims)


class ReportConfig(BaseModel):
    """Config for deterministic offline report generation."""

    model_config = ConfigDict(frozen=True)

    input_dir: Path
    output_dir: Path
    n_sims: int = Field(gt=0)
    seed: int
    rating_scale: float = Field(default=10.0, gt=0)
    batch_size: int | None = Field(default=None, gt=0)
    engine: str = Field(default="numpy")

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, value: str) -> str:
        """Normalize and validate engine selection."""

        normalized = value.strip().lower()
        if normalized not in _ALLOWED_ENGINES:
            msg = f"Unsupported engine {value!r}; expected one of {sorted(_ALLOWED_ENGINES)}"
            raise ValueError(msg)
        return normalized

    @property
    def effective_batch_size(self) -> int:
        """Return the concrete batch size used for execution."""

        return min(self.batch_size or self.n_sims, self.n_sims)


class RunManifest(BaseModel):
    """Reproducibility metadata persisted alongside run artifacts."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    run_id: str
    created_at: datetime
    code_version: str
    git_commit: str | None = None
    input_dir: Path
    dataset_hash: str = Field(min_length=64, max_length=64)
    input_hashes: dict[str, str]
    n_sims: int = Field(gt=0)
    seed: int
    rating_scale: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    engine: str
    log_level: str
    entry_ids: list[str]
    team_ids: list[str]


class RunCheckpoint(BaseModel):
    """Incremental aggregate state for resumable simulation runs."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    run_id: str
    completed_sims: int = Field(ge=0)
    completed_batches: int = Field(ge=0)
    win_share_totals: list[float]
    score_totals: list[int]
    champion_counts: dict[str, int]


class SimulationRunMetadata(BaseModel):
    """Operational metadata for one simulation execution."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    engine: str
    batch_size: int = Field(gt=0)
    batches_completed: int = Field(ge=0)
    resumed_from_checkpoint: bool
    run_dir: Path | None = None
    manifest_path: Path | None = None
    checkpoint_path: Path | None = None
    result_path: Path | None = None
    log_path: Path | None = None


class SimulationEntryResult(BaseModel):
    """Aggregated simulation metrics for one pool entry."""

    model_config = ConfigDict(frozen=True)

    entry_id: str
    entry_name: str
    win_share: float
    average_score: float


class SimulationResult(BaseModel):
    """Serializable output for CLI table, JSON, and persisted run artifacts."""

    model_config = ConfigDict(frozen=True)

    n_sims: int
    seed: int
    entry_results: list[SimulationEntryResult]
    champion_counts: dict[str, int]
    run_metadata: SimulationRunMetadata


class ReportArtifact(BaseModel):
    """One persisted report artifact inside an output bundle."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    path: Path
    kind: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    row_count: int | None = Field(default=None, ge=0)


class ReportBundleManifest(BaseModel):
    """Reproducibility metadata for a generated report bundle."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    report_id: str
    created_at: datetime
    code_version: str
    git_commit: str | None = None
    input_dir: Path
    dataset_hash: str = Field(min_length=64, max_length=64)
    input_hashes: dict[str, str]
    output_dir: Path
    n_sims: int = Field(gt=0)
    seed: int
    rating_scale: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    engine: str
    entry_ids: list[str]
    team_ids: list[str]
    artifacts: list[ReportArtifact]


class TeamAdvancementOddsRow(BaseModel):
    """Round-by-round advancement probabilities for one team."""

    model_config = ConfigDict(frozen=True)

    team_id: str = Field(min_length=1)
    team_name: str = Field(min_length=1)
    seed: int = Field(ge=1, le=16)
    region: str = Field(min_length=1)
    reach_round_of_32: float = Field(ge=0.0, le=1.0)
    reach_sweet_16: float = Field(ge=0.0, le=1.0)
    reach_elite_8: float = Field(ge=0.0, le=1.0)
    reach_final_four: float = Field(ge=0.0, le=1.0)
    reach_title_game: float = Field(ge=0.0, le=1.0)
    win_championship: float = Field(ge=0.0, le=1.0)


class EntryReportRow(BaseModel):
    """Summary metrics for one entry in a report bundle."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    entry_id: str = Field(min_length=1)
    entry_name: str = Field(min_length=1)
    win_share: float = Field(ge=0.0, le=1.0)
    average_score: float


class ChampionOddsRow(BaseModel):
    """Champion probability summary for one team."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    team_id: str = Field(min_length=1)
    team_name: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)


class ChampionSensitivityRow(BaseModel):
    """Entry performance conditioned on a specific champion outcome."""

    model_config = ConfigDict(frozen=True)

    champion_team_id: str = Field(min_length=1)
    champion_team_name: str = Field(min_length=1)
    champion_probability: float = Field(ge=0.0, le=1.0)
    champion_simulations: int = Field(ge=1)
    entry_rank: int = Field(ge=1)
    entry_id: str = Field(min_length=1)
    entry_name: str = Field(min_length=1)
    baseline_win_share: float = Field(ge=0.0, le=1.0)
    conditional_win_share: float = Field(ge=0.0, le=1.0)
    win_share_delta: float
    baseline_average_score: float
    conditional_average_score: float
    average_score_delta: float


class ReportSummary(BaseModel):
    """Compact JSON summary for a generated report bundle."""

    model_config = ConfigDict(frozen=True)

    report_id: str
    output_dir: Path
    n_sims: int = Field(gt=0)
    seed: int
    engine: str
    batch_size: int = Field(gt=0)
    entry_count: int = Field(ge=0)
    team_count: int = Field(ge=0)
    top_entries: list[EntryReportRow]
    top_champions: list[ChampionOddsRow]


class ReportBundleResult(BaseModel):
    """In-memory result returned after writing a report bundle."""

    model_config = ConfigDict(frozen=True)

    manifest: ReportBundleManifest
    summary: ReportSummary


class BenchmarkConfig(BaseModel):
    """Config for hotspot benchmark runs."""

    model_config = ConfigDict(frozen=True)

    input_dir: Path
    n_sims: int = Field(gt=0)
    repeats: int = Field(default=3, gt=0)
    engine: str = Field(default="numpy")
    simulation_budget_ms: float = Field(default=1_500.0, gt=0)
    scoring_budget_ms: float = Field(default=750.0, gt=0)
    rating_scale: float = Field(default=10.0, gt=0)

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, value: str) -> str:
        """Normalize and validate engine selection."""

        normalized = value.strip().lower()
        if normalized not in _ALLOWED_ENGINES:
            msg = f"Unsupported engine {value!r}; expected one of {sorted(_ALLOWED_ENGINES)}"
            raise ValueError(msg)
        return normalized


class BenchmarkMeasurement(BaseModel):
    """One measured benchmark category compared against a budget."""

    model_config = ConfigDict(frozen=True)

    mean_ms: float
    min_ms: float
    budget_ms: float
    within_budget: bool


class BenchmarkReport(BaseModel):
    """Hotspot benchmark summary for simulation and scoring paths."""

    model_config = ConfigDict(frozen=True)

    n_sims: int
    repeats: int
    engine: str
    simulation: BenchmarkMeasurement
    scoring: BenchmarkMeasurement
