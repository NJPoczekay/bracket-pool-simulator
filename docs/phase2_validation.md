# Phase 2 Validation and Parity

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
- Legacy parity script:
  - `scripts/check_legacy_parity.py`
- Parity script integration tests:
  - `tests/integration/test_legacy_parity_script.py`

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

## Legacy Parity Script

Run with defaults (`top_n=10`, `max_delta=0.02`, `n_sims=10000`, `seed=42`):

```bash
uv run --extra dev python scripts/check_legacy_parity.py \
  --legacy-json /path/to/legacy_results.json
```

Override parameters:

```bash
uv run --extra dev python scripts/check_legacy_parity.py \
  --legacy-json /path/to/legacy_results.json \
  --input-dir tests/fixtures/synthetic_64 \
  --n-sims 10000 \
  --seed 42 \
  --top-n 10 \
  --max-delta 0.02
```

Expected legacy JSON schema:

```json
{
  "entry_win_shares": {
    "entry_chalk": 0.20,
    "entry_balanced": 0.17
  }
}
```

Behavior:
- compares current output against legacy values for the top `N` entries by current ranking
- prints a table of current share, legacy share, and absolute delta
- exits with:
  - `0`: parity within threshold
  - `1`: one or more deltas exceed threshold
  - `2`: invalid input or missing/invalid legacy file

## Full Verification

Run all lint/type/test checks:

```bash
make ci
```
