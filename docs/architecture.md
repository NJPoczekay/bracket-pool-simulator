# Bracket Pool Simulator Rewrite: Recommended Architecture

## Objectives

- Deterministic, reproducible simulations.
- Clear separation between data ingestion, simulation logic, and scoring.
- Easy year-to-year updates with minimal code changes.
- Testable core logic independent of scraping and external websites.
- Fast enough to run large simulation counts locally and in CI.

## Core Architectural Decisions

1. Use a layered architecture with strict boundaries:
   - `domain`: pure business objects and rules (teams, games, entries, scoring).
   - `application`: orchestration use-cases (simulate tournament, score pool, report odds).
   - `infrastructure`: ESPN/KenPom/result providers, file/db storage, CLI/API adapters.
2. Keep the simulation engine pure and side-effect free:
   - Inputs: immutable tournament state + model parameters + RNG seed.
   - Outputs: typed result arrays/tables.
3. Store raw source data separately from parsed/normalized data:
   - Never mutate raw scraped HTML/JSON.
   - Build parse/normalize pipelines that are idempotent and rerunnable.
4. Model the bracket explicitly as a game graph:
   - 63 game nodes with parent-child links.
   - Avoid implicit pairing based on dict iteration order.
5. Treat completed games as constraints:
   - Hard-lock known outcomes before simulation.
   - Validate that constraints are bracket-consistent before running.

## Proposed Stack (Python)

- Runtime: Python 3.12+
- Data models/validation: `pydantic`
- Numerical engine: `numpy`
- Optional speed-up: `numba` for hot loops (behind a feature flag)
- Tabular/reporting: `polars`
- Scraping/API access:
  - Preferred: direct HTTP (`httpx`) + structured endpoints if available
  - Fallback: browser automation (`playwright`) only when necessary
- CLI: `typer`
- Config: `pydantic-settings` + environment profiles
- Logging: `structlog` (or standard `logging` with JSON formatter)
- Serialization/storage:
  - Local cache format: Parquet for tables, JSON for metadata
  - Optional DB: SQLite + `sqlmodel` for reproducible run history

## Repository Structure

```text
src/
  bracket_sim/
    domain/
      models.py
      bracket_graph.py
      scoring.py
      probability_model.py
    application/
      simulate_pool.py
      update_data.py
      generate_reports.py
    infrastructure/
      providers/
        espn_provider.py
        kenpom_provider.py
        results_provider.py
      parsing/
        espn_parser.py
        name_normalizer.py
      storage/
        cache_repo.py
        run_repo.py
      cli/
        main.py
tests/
  unit/
  integration/
  fixtures/
```

## Data Contracts

Define typed schemas for:

- `Team`, `Game`, `Bracket`, `EntryPick`, `PoolEntry`
- `CompletedGameConstraint`
- `RatingSnapshot` (source, effective date, values)
- `SimulationConfig` (n_sims, seed, scoring rules, lock policy)
- `SimulationResult` (team outcomes, entry scores, pool win shares)

These contracts should be versioned. Breaking schema changes should force migration scripts.

## Simulation and Scoring Design

1. Build canonical tournament state:
   - Team list, seedings, bracket graph, completed games, ratings snapshot.
2. Resolve name mapping once in ingestion.
3. Precompute all static arrays before the Monte Carlo loop.
4. Use seeded RNG and store seed/run metadata.
5. Run simulation in batches (for memory control and checkpointing).
6. Score entries vectorially from per-team round outcomes.
7. Compute tie-split winner shares per simulation.
8. Persist both summary metrics and optional detailed traces.

## Tooling to Add

- Package/dependency management: `uv`
- Lint/format:
  - `ruff` (lint + import sort)
- Type checking: `mypy` (strict on domain + application layers)
- Tests: `pytest` + `pytest-cov`
- Pre-commit hooks: `pre-commit` with lint/type/test gates
- CI: GitHub Actions
  - Matrix: Python versions
  - Steps: lint, type-check, unit tests, integration tests (with fixtures)
- Task runner: `just` or `make` for common commands
- Optional benchmarking: `pytest-benchmark` for sim/scoring hotspots

## Testing Strategy

- Unit tests:
  - Bracket graph validity and advancement logic
  - Constraint application
  - Probability function sanity and bounds
  - Scoring correctness for known toy brackets
- Property tests (optional `hypothesis`):
  - Exactly one champion per simulation
  - No team exceeds feasible wins
  - Score monotonicity properties
- Integration tests:
  - End-to-end with fixed fixtures and fixed seed
  - Deterministic golden outputs for regression protection

## Observability and Operations

- Structured logs with run IDs and source snapshot IDs.
- Persist run manifest:
  - code version
  - config
  - seed
  - input data hashes
- Add lightweight data quality checks:
  - missing teams
  - duplicate aliases
  - impossible completed-game constraints

## Migration Plan (Suggested)

1. Build new domain + scoring engine first using fixture data.
2. Add explicit bracket graph + completed-game constraint layer.
3. Add new data ingestion adapters and parsers.
4. Validate outputs against current system on historical tournaments.
5. Add CLI/reporting and deprecate legacy scripts.

## Practical Defaults

- Default to offline-first execution using cached normalized data.
- Require explicit `--refresh-data` to hit external sources.
- Require explicit seed in production runs (or auto-generate and persist).
- Keep scraping as a replaceable adapter, not core logic.
