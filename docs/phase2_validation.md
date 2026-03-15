# Phase 2 Validation

This document describes the validation artifacts and checks added for Phase 2.

## What Was Added

- Golden regression artifact:
  - `tests/expected/synthetic_64_seed99_n300.json`
  - Generated from:
    - input: `tests/fixtures/synthetic_64`
    - `n_sims=300`
    - `seed=99`
    - `rating_scale=10.0`
- Golden regression integration test:
  - `tests/integration/test_golden_output.py`
- Additional core validation tests:
  - scoring depth behavior and advancement consistency checks
  - champion and feasible-win invariants

## Golden Artifact Workflow

Run the golden regression test:

```bash
uv run --extra dev pytest tests/integration/test_golden_output.py
```

Regenerate the golden artifact intentionally:

```bash
uv run --extra dev python -m bracket_sim.infrastructure.cli.main simulate \
  --input tests/fixtures/synthetic_64 \
  --n-sims 300 \
  --seed 99 \
  --json > tests/expected/synthetic_64_seed99_n300.json
```

Notes:
- Regenerate only when changes are intentional.
- Review the JSON diff in PRs as part of behavior-change validation.

## Full Verification

Run all lint/type/test checks:

```bash
make ci
```
