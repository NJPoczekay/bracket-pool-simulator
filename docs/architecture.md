# Bracket Pool Simulator Architecture

## Objectives

- Deterministic, reproducible simulations.
- Clear separation between data ingestion, simulation logic, scoring, and interface adapters.
- Easy year-to-year updates with minimal code changes.
- Testable core logic independent of scraping and external websites.
- Fast enough to run large simulation counts locally and in CI.
- Stable product-facing contracts for the staged web analyzer/optimizer roadmap.

## Layered Design

1. Use a layered architecture with strict boundaries:
   - `domain`: pure business objects and rules (`models.py`, `product_models.py`, bracket graph, scoring, constraints).
   - `application`: orchestration use-cases and product bootstrap services (`simulate_pool.py`, `generate_reports.py`, `product_foundation.py`).
   - `infrastructure`: providers, file storage, observability, CLI presentation, and web/API adapters.
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
6. Keep presentation-specific formatting out of reusable services:
   - CLI text rendering lives under `infrastructure/cli/presenter.py`.
   - FastAPI routes return typed models from the application layer.

## Proposed Stack (Python)

- Runtime: Python 3.12+
- Data models/validation: `pydantic`
- Numerical engine: `numpy`
- Optional speed-up: `numba` for hot loops (behind a feature flag)
- Tabular/reporting: `polars`
- Web/API surface: `fastapi` + `uvicorn`
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
      product_models.py
      bracket_graph.py
      scoring.py
      probability_model.py
      constraints.py
    application/
      simulate_pool.py
      generate_reports.py
      prepare_data.py
      refresh_data.py
      refresh_national_picks.py
      product_foundation.py
    infrastructure/
      providers/
        espn_api.py
        ratings.py
        contracts.py
      storage/
        cache_keys.py
        run_artifacts.py
        report_bundle.py
        normalized_loader.py
        prepared_writer.py
        raw_loader.py
      cli/
        main.py
        presenter.py
      web/
        main.py
        app.py
tests/
  unit/
  integration/
  fixtures/
```

## Product Foundation And Integrated App Surface

Phase 0 established the contracts and adapters needed for later browser features. The integrated web app now uses that same foundation to host two adjacent workflows without merging their state:

- `Bracket Lab`: pre-tournament planning and future analyzer/optimizer work
- `Pool Tracker`: locked-entry odds tracking and report automation

- Shared product-facing models live in `domain/product_models.py`.
- The local FastAPI app lives in `infrastructure/web/main.py`.
- Scheduler helpers for tracker pools live in `infrastructure/web/app.py`.
- Product bootstrap metadata comes from `application/product_foundation.py`.
- CLI rendering is isolated in `infrastructure/cli/presenter.py` so service results stay reusable.

The product contracts cover:

- top-level workflow metadata for the integrated app shell
- bracket editing: `BracketEditPick`, `EditableBracket`
- pool/scoring setup: `PoolSettings`, `ScoringSystem`, `ScoringSystemKey`
- completion setup: `CompletionMode`, `CompletionModeOption`
- future analyzer payloads: `PickDiagnostic`, `BracketAnalysis`
- future optimizer payloads: `OptimizationAlternative`, `OptimizationResult`

These contracts should remain versioned and stable enough for future API endpoints. `PoolSettings` intentionally stays separate from tracker pool configuration because Bracket Lab is pre-lock and Pool Tracker is post-lock.

## Cache And Manifest Strategy

Reusable simulation-derived artifacts now share a common identity model:

1. `dataset_hash`
   - Hash every top-level `.json`, `.csv`, and `.parquet` file in a prepared dataset directory.
   - Sort by filename, hash file contents, then hash the resulting file-hash manifest.
2. `cache_key`
   - Hash the artifact kind plus the `dataset_hash` plus the JSON-serialized settings payload.
   - Planned first-class artifact kinds are `analysis` and `optimization`.
3. Manifests
   - Run and report manifests persist both `dataset_hash` and `input_hashes`.
   - This keeps coarse cache identity and per-file reproducibility metadata aligned.

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

## Testing Strategy

- Unit tests:
  - Bracket graph validity and advancement logic
  - Constraint application
  - Probability function sanity and bounds
  - Scoring correctness for known toy brackets
  - Product model validation and cache-key determinism
  - Web/API bootstrap contract coverage
- Property tests (optional `hypothesis`):
  - Exactly one champion per simulation
  - No team exceeds feasible wins
  - Score monotonicity properties
- Integration tests:
  - End-to-end with fixed fixtures and fixed seed
  - Deterministic golden outputs for regression protection

## Observability and Operations

- Structured logs with run IDs and source snapshot IDs.
- Persist run/report manifests with:
  - code version
  - git commit
  - config
  - seed
  - dataset hash
  - per-file input hashes
- Add lightweight data quality checks:
  - missing teams
  - duplicate aliases
  - impossible completed-game constraints

## Practical Defaults

- Default to offline-first execution using cached normalized data.
- Require explicit `--refresh-data` to hit external sources.
- Require explicit seed in production runs (or auto-generate and persist).
- Keep scraping as a replaceable adapter, not core logic.
- Keep the integrated web shell thin and route all product logic through typed application services.
