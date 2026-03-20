# Bracket Pool Simulator Rewrite Roadmap

## MVP Goal

Ship a deterministic, testable CLI that:
- Loads normalized local inputs (no live scraping required)
- Simulates tournament outcomes with a fixed seed
- Scores pool entries with ESPN-style scoring
- Outputs each entry's estimated chance to win the pool (with tie splits)

## Guiding Principles

- Build the pure simulation/scoring core before external integrations.
- Keep ingestion and scraping outside core logic.
- Require reproducibility (seed + input hashes + config).
- Prefer small, verifiable increments with regression tests.

## Integration Note

The local web product is now one app shell with two adjacent workflows:

- `Bracket Lab` for pre-tournament planning and future optimizer work
- `Pool Tracker` for post-lock odds tracking on real pool entries

Phase 7 covers the `Pool Tracker` half of that integrated app. It should not absorb Bracket Lab assumptions or become the source of truth for pre-lock optimizer inputs.

## Phase 0: Foundation (1-2 days)

### Deliverables
- Project skeleton (`domain`, `application`, `infrastructure`)
- Dependency/tool setup (`uv`)
- Lint/type/test pipeline:
  - `ruff`
  - `mypy`
  - `pytest`
  - `pre-commit`
- CI workflow running lint/type/tests

### Issues
- Create package layout under `src/`.
- Add `pyproject.toml` with dependencies and dev dependencies.
- Configure `ruff`, `mypy`, `pytest`.
- Add pre-commit hooks.
- Add GitHub Actions CI.
- Add `justfile` or `Makefile` commands (`lint`, `typecheck`, `test`, `ci`).

### Exit Criteria
- Clean repo bootstraps with one command.
- CI green on every push.

## Phase 1: Core Domain + Engine (Week 1)

### Deliverables
- Typed domain models:
  - `Team`, `Game`, `Bracket`, `EntryPick`, `PoolEntry`, `SimulationConfig`
- Explicit bracket graph (63 games)
- Constraint application for completed games
- Deterministic Monte Carlo simulator (seeded RNG)
- ESPN-style scoring and tie-split pool winner logic
- CLI command:
  - `simulate --input <normalized-dir> --n-sims 100000 --seed 42`

### Issues
- Implement bracket graph builder/validator.
- Implement constraints validator (detect impossible states).
- Implement simulation loop with vectorized data structures.
- Implement scoring function from team outcomes + picks.
- Implement winner-share aggregation with ties.
- Implement CLI output table + JSON output mode.

### Exit Criteria
- One end-to-end local run from fixture inputs.
- Same seed + same input => identical outputs.

## Phase 2: MVP Validation and Confidence (Week 1)

### Deliverables
- Golden fixture dataset (at least one historical pool/year)
- Unit + integration test suite for core behavior
- Parity analysis against legacy implementation on same fixtures

### Issues
- Add unit tests for bracket advancement rules.
- Add unit tests for scoring correctness on toy cases.
- Add invariant tests (exactly one champion; feasible win counts).
- Add integration test: fixture -> full simulation -> golden outputs.
- Add parity script comparing top entry win odds vs legacy.

### Exit Criteria
- Core tests stable and meaningful.
- Parity deltas documented and accepted.

### Implementation Notes (Current)
- Validation runbook: `docs/phase2_validation.md`
- Golden snapshot: `tests/expected/synthetic_64_seed99_n300.json`
- Golden test: `tests/integration/test_golden_output.py`

## Phase 3: Data Normalization Pipeline (Week 2)

### Deliverables
- Offline-first ingestion pipeline:
  - raw source files -> normalized canonical datasets
- Name/alias normalization rules and validators
- Cache repository using Parquet/JSON

### Issues
- Define schema for normalized team, bracket, entry, ratings, constraints.
- Implement parser modules for local raw files.
- Implement alias mapping resolver + duplicate detection.
- Implement data quality checks (missing teams, unknown aliases, bad constraints).
- Add command:
  - `prepare-data --raw <dir> --out <normalized-dir>`

### Exit Criteria
- Repeatable data prep without touching simulation code.
- Prepared dataset passes validation checks.

## Phase 4: External Data Providers (Week 2-3)

### Deliverables
- Replaceable provider adapters:
  - pool entries
  - completed game results
  - ratings snapshot
  - public national pick counts
- Refresh workflow:
  - `refresh-data` then `prepare-data`
  - `refresh-national-picks` for acquisition-only national snapshots

### Issues
- Implement provider interface contracts.
- Add `httpx`-based clients where possible.
- Add `playwright` fallback adapter only where needed.
- Add retry/backoff and clear failure modes.
- Persist raw fetch snapshots with metadata.

### Exit Criteria
- Live refresh works end-to-end.
- Offline rerun from cached normalized data is reproducible.

## Phase 5: Hardening + Performance (Week 3+)

### Deliverables
- Run manifest capture:
  - seed, config, code version, input hashes
- Structured logging with run IDs
- Batch simulation/checkpointing for large runs
- Optional accelerated path (`numba`) behind feature flag
- Benchmarks and performance budget

### Issues
- Add run manifest writer and verifier.
- Add JSON logs and log levels.
- Implement batch execution and resumable runs.
- Add benchmark tests for simulation/scoring hotspots.
- Add optional numba-accelerated engine implementation.

### Exit Criteria
- Predictable runtime at target simulation counts.
- Easy debugging and reproducibility for any run artifact.

## Phase 6: Productization (Optional)

### Deliverables
- Rich reports (team advancement odds, sensitivity analysis)
- Report bundle artifacts (CSV/JSON outputs, summary manifest)
- Deterministic offline report generation from normalized inputs

### Issues
- Add report generation module and templates.
- Define report schemas and output artifact layout.
- Add CLI/report command for deterministic report generation.
- Revisit sensitivity semantics for sparse scoring formats so we can distinguish directly scoring pivotal games from broader conditional-equity swings when needed.

### Exit Criteria
- Non-CLI-consumable report artifacts are viable without manual analysis.

## Phase 7: Self-Serve Access + Automation (Optional)

### Deliverables
- `Pool Tracker` section inside the integrated API/UI for non-CLI users
- Scheduled refresh + report generation

### Issues
- Add API endpoint(s) for run + fetch results.
- Add minimal Pool Tracker UI flow for non-technical users inside the shared app shell.
- Add scheduler/job automation.
- Keep live tracker pool config separate from pre-tournament optimizer/planning state.

### Exit Criteria
- Non-technical users can trigger live pool refreshes and consume results without manual file editing.

## MVP Scope Lock (What We Build First)

1. Foundation/tooling (Phase 0)
2. Pure core engine + CLI from local fixtures (Phase 1)
3. Validation suite + parity check (Phase 2)

Do not block MVP on live scraping, UI, or advanced optimization.

## Suggested Milestone Checklist

- Milestone A: Tooling and CI baseline complete
- Milestone B: Deterministic simulation/scoring CLI complete
- Milestone C: Golden tests and parity acceptance complete (MVP)
- Milestone D: Data pipeline complete
- Milestone E: Live provider integration complete
- Milestone F: Performance and observability complete
- Milestone G: Self-serve access and automation complete

## Immediate Next 5 Tickets

1. Create project skeleton + `pyproject.toml` + CI.
2. Implement typed domain models and bracket graph validator.
3. Implement deterministic simulator and scoring engine.
4. Implement CLI command for local fixture simulation.
5. Add golden integration test with fixed seed and expected output.
