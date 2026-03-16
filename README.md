# Bracket Pool Simulator CLI

`bracket-pool-simulator` is a command-line tool for refreshing NCAA bracket pool data, normalizing it into simulation-ready inputs, and running deterministic Monte Carlo simulations against an ESPN-style pool.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) for dependency management

## Setup

Install project dependencies:

```bash
uv sync --extra dev
```

If you want to use the optional Numba simulation engine, install that extra too:

```bash
uv sync --extra dev --extra numba
```

You can run the CLI in either of these ways:

```bash
uv run bracket-sim --help
```

```bash
uv run python -m bracket_sim.infrastructure.cli.main --help
```

## Commands

The CLI exposes five commands:

- `refresh-data`: fetch raw bracket data from ESPN plus ratings data
- `prepare-data`: normalize raw files into validated simulation inputs
- `simulate`: run deterministic pool simulations
- `benchmark`: measure simulation and scoring performance against budgets
- `refresh-national-picks`: download ESPN national pick-count snapshots

## Typical Workflow

### 1. Refresh raw data

Use `refresh-data` when you want to build a raw dataset from a live ESPN Tournament Challenge group.

```bash
uv run bracket-sim refresh-data \
  --group-url "https://fantasy.espn.com/games/tournament-challenge-bracket-2026/group?id=YOUR_GROUP_ID" \
  --raw data/raw/2026
```

By default the command expects ratings from either:

- `--ratings-file /path/to/ratings.csv`, or
- an existing cached `ratings.csv` already present inside `--raw`

You can also fetch ratings from KenPom instead:

```bash
export KENPOM_COOKIE="your authenticated cookie string"

uv run bracket-sim refresh-data \
  --group-url "https://fantasy.espn.com/games/tournament-challenge-bracket-2026/group?id=YOUR_GROUP_ID" \
  --raw data/raw/2026 \
  --kenpom
```

Useful options:

- `--min-usable-entries`: fail if too many ESPN entries are skipped during parsing
- `--ratings-file`: read a local CSV with columns `team,rating,tempo` or `team_id,rating,tempo`
- `--kenpom`: fetch ratings from KenPom using `KENPOM_COOKIE`

Expected raw directory contents after a successful refresh:

- `teams.csv`
- `games.csv`
- `entries.json`
- `constraints.json`
- `ratings.csv`
- `metadata.json`
- `snapshots/challenge.json`
- `snapshots/group.json`
- optionally `snapshots/group_retry.json`

### 2. Prepare normalized inputs

`prepare-data` converts the raw dataset into validated simulation inputs. `--raw` and `--out` must be different directories.

```bash
uv run bracket-sim prepare-data \
  --raw data/raw/2026 \
  --out data/prepared/2026
```

Expected prepared directory contents:

- `teams.json`
- `games.json`
- `entries.json`
- `constraints.json` when completed games exist
- `ratings.csv`

### 3. Run simulations

Use `simulate` on a prepared dataset:

```bash
uv run bracket-sim simulate \
  --input data/prepared/2026 \
  --n-sims 100000 \
  --seed 42
```

Structured JSON output:

```bash
uv run bracket-sim simulate \
  --input data/prepared/2026 \
  --n-sims 100000 \
  --seed 42 \
  --json
```

Useful options:

- `--engine`: `numpy` or `numba`
- `--batch-size`: split large runs into checkpointed batches
- `--run-dir`: write resumable artifacts
- `--resume`: continue a prior run from `--run-dir`
- `--log-level`: `debug`, `info`, `warning`, or `error`

When `--run-dir` is provided, the simulator writes:

- `manifest.json`
- `checkpoint.json`
- `result.json`
- `log.jsonl`

Resume example:

```bash
uv run bracket-sim simulate \
  --input data/prepared/2026 \
  --n-sims 1000000 \
  --seed 42 \
  --batch-size 50000 \
  --run-dir runs/2026-main \
  --resume
```

`--resume` requires the same input data and runtime settings that created the original manifest.

### 4. Benchmark performance

`benchmark` measures simulation and scoring runtime and exits non-zero if either budget is exceeded.

```bash
uv run bracket-sim benchmark \
  --input data/prepared/2026 \
  --n-sims 5000 \
  --repeats 3
```

Tune the budgets if needed:

```bash
uv run bracket-sim benchmark \
  --input data/prepared/2026 \
  --n-sims 5000 \
  --repeats 3 \
  --simulation-budget-ms 1500 \
  --scoring-budget-ms 750
```

### 5. Refresh national pick counts

This command downloads the public ESPN challenge payload and stores national pick-count artifacts for later analysis.

```bash
uv run bracket-sim refresh-national-picks \
  --challenge tournament-challenge-bracket-2026 \
  --out data/national-picks/2026
```

`--challenge` accepts an ESPN bracket URL, group URL, or challenge key.

Output includes:

- `national_picks.csv`
- `metadata.json`
- `snapshots/challenge.json`

## Local Example With Bundled Test Data

The repository includes a prepared synthetic dataset under `tests/fixtures/synthetic_64`, so you can try the simulator immediately:

```bash
uv run bracket-sim simulate \
  --input tests/fixtures/synthetic_64 \
  --n-sims 100 \
  --seed 7
```

You can also emit JSON from the same fixture:

```bash
uv run bracket-sim simulate \
  --input tests/fixtures/synthetic_64 \
  --n-sims 100 \
  --seed 7 \
  --json
```

## Notes

- `simulate` and `benchmark` require normalized input files, not raw refresh outputs.
- `prepare-data` is the bridge between acquisition (`refresh-data`) and simulation (`simulate`).
- `refresh-national-picks` is acquisition-only; it does not change simulation inputs by itself.
