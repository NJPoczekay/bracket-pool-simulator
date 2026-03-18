# Bracket Pool Simulator

`bracket-pool-simulator` supports the original CLI workflow plus one integrated local web/API app. The project refreshes NCAA bracket pool data, normalizes it into simulation-ready inputs, runs deterministic Monte Carlo simulations against an ESPN-style pool, and now presents two adjacent browser workflows in one shell:

- `Bracket Lab`: pre-tournament planning and future optimizer/analyzer flows
- `Pool Tracker`: live pool refresh/report/odds tracking once real entries are locked

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

You can run the local web/API surface in either of these ways:

```bash
uv run bracket-web --reload
```

```bash
uv run bracket-sim serve --reload
```

By default the server runs on [http://127.0.0.1:8000](http://127.0.0.1:8000) and exposes:

- `/`: the integrated Bracket Lab + Pool Tracker shell
- `/api/health`: a health/version probe
- `/api/foundation`: shared product metadata for workflows, scoring systems, completion modes, and cache rules
- `/api/cache-key`: deterministic cache-key preview endpoint for future analysis/optimization settings
- `/api/pools`: live tracker pool metadata, or an empty list when tracking is not configured

## Commands

The CLI exposes nine commands:

- `refresh-bracket-lab-data`: fetch Bracket Lab raw data from ESPN challenge APIs plus KenPom inputs
- `prepare-bracket-lab-data`: normalize raw Bracket Lab files into prepared analysis/optimization inputs
- `refresh-data`: fetch tracker raw bracket data from ESPN plus ratings data
- `prepare-data`: normalize tracker raw files into validated simulation inputs
- `simulate`: run deterministic pool simulations
- `benchmark`: measure simulation and scoring performance against budgets
- `report`: generate deterministic offline report bundles from normalized inputs
- `refresh-national-picks`: download ESPN national pick-count snapshots
- `serve`: run the integrated local web/API surface, or pass `--config` to enable live pool tracking data

## Typical Workflow

### Bracket Lab: Refresh raw data

Use `refresh-bracket-lab-data` when you want a pre-lock dataset for bracket completion, analyzer, and optimizer work.

```bash
uv run bracket-sim refresh-bracket-lab-data \
  --challenge tournament-challenge-bracket-2026 \
  --ratings-file /path/to/kenpom.csv
```

Default raw output: `data/2026/bracket-lab/tournament-challenge-bracket-2026/raw`

Or load the KenPom source rows from a saved HTML snapshot:

```bash
uv run bracket-sim refresh-bracket-lab-data \
  --challenge tournament-challenge-bracket-2026 \
  --kenpom
```

`--kenpom` looks for a file like
`data/kenpom_snapshots/2026 Pomeroy College Basketball Ratings.html`.

Expected raw Bracket Lab contents:

- `teams.csv`
- `games.csv`
- `constraints.json` when completed games exist
- `national_picks.csv`
- `kenpom.csv`
- `metadata.json`
- `snapshots/challenge.json`
- optionally `aliases.csv` when you maintain manual team-name overrides

### Bracket Lab: Prepare inputs

`prepare-bracket-lab-data` converts the raw Bracket Lab snapshot into a self-contained dataset for later analysis and optimization work.

```bash
uv run bracket-sim prepare-bracket-lab-data \
  --raw data/2026/bracket-lab/tournament-challenge-bracket-2026/raw
```

Default prepared output: `data/2026/bracket-lab/tournament-challenge-bracket-2026/prepared`

Expected prepared Bracket Lab contents:

- `teams.json`
- `games.json`
- `constraints.json` when completed games exist
- `public_picks.csv`
- `ratings.csv`
- `completion_inputs.json`
- `play_in_slots.json` when unresolved First Four slots exist
- `metadata.json`

### 1. Refresh raw data (Pool Tracker)

Use `refresh-data` when you want to build a raw dataset from a live ESPN Tournament Challenge group.

```bash
uv run bracket-sim refresh-data \
  --group-url "https://fantasy.espn.com/games/tournament-challenge-bracket-2026/group?id=YOUR_GROUP_ID"
```

Default raw output: `data/2026/tracker/<group-id>/raw`

By default the command expects ratings from either:

- `--ratings-file /path/to/ratings.csv`, or
- an existing cached `ratings.csv` already present inside `--raw`

You can also load ratings from a saved KenPom HTML snapshot instead:

```bash
uv run bracket-sim refresh-data \
  --group-url "https://fantasy.espn.com/games/tournament-challenge-bracket-2026/group?id=YOUR_GROUP_ID" \
  --kenpom
```

When `--kenpom` is set, the CLI looks for a file like
`data/kenpom_snapshots/2026 Pomeroy College Basketball Ratings.html`.

Useful options:

- `--min-usable-entries`: fail if too many ESPN entries are skipped during parsing
- `--ratings-file`: read a local CSV with columns `team,rating,tempo` or `team_id,rating,tempo`
- `--kenpom`: load a saved KenPom HTML snapshot from `data/kenpom_snapshots`

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
  --raw data/2026/tracker/YOUR_GROUP_ID/raw
```

Default prepared output: `data/2026/tracker/YOUR_GROUP_ID/prepared`

Expected prepared directory contents:

- `teams.json`
- `games.json`
- `entries.json`
- `constraints.json` when completed games exist
- `ratings.csv`

Prepared datasets now participate in a shared identity scheme for future browser features:

- `dataset_hash`: SHA-256 over the sorted top-level `.json`/`.csv`/`.parquet` file hashes
- `input_hashes`: per-file hashes persisted into run and report manifests
- future analysis/optimization cache keys: SHA-256 over artifact kind + dataset hash + settings payload

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

### 5. Generate offline report bundles

`report` turns a prepared dataset into machine-friendly artifacts for downstream analysis without requiring manual CLI inspection.

```bash
uv run bracket-sim report \
  --input data/prepared/2026 \
  --n-sims 100000 \
  --seed 42
```

Default report archive root: `reports/<season>/tracker/<dataset-slug>/`

When `--out` is omitted, the CLI writes a timestamped archive bundle under that root and refreshes `latest/` with the newest canonical artifacts.

Useful options:

- `--batch-size`: split deterministic report generation into batches
- `--engine`: `numpy` or `numba`
- `--json`: print the summary bundle metadata as JSON

Output includes:

- `manifest.json`
- `summary.json`
- `team_advancement_odds.csv`
- `entry_summary.csv`
- `champion_sensitivity.csv`

### 6. Refresh national pick counts

This command downloads the public ESPN challenge payload and stores national pick-count artifacts for later analysis.

```bash
uv run bracket-sim refresh-national-picks \
  --challenge tournament-challenge-bracket-2026
```

Default output: `data/2026/national-picks/tournament-challenge-bracket-2026`

`--challenge` accepts an ESPN bracket URL, group URL, or challenge key.

Output includes:

- `national_picks.csv`
- `metadata.json`
- `snapshots/challenge.json`

### 7. Run the local web/API surface

The web app now keeps both browser-facing workflows in one place:

- `Bracket Lab` comes first and stays focused on pre-lock planning and future optimizer flows
- `Pool Tracker` comes second and stays focused on locked-entry pool odds tracking

```bash
uv run bracket-sim serve --host 127.0.0.1 --port 8000 --reload
```

Without `--config`, the app still renders both sections, but `Pool Tracker` stays in a setup state and `/api/pools` returns an empty list.

Pass `--config` to enable live pool tracking data. This mode runs the existing `refresh-data -> prepare-data -> report` pipeline synchronously for each configured pool and keeps report bundles separated by pool.

Start from the example config:

```bash
cp config/pools.example.toml config/pools.toml
```

Paths inside the TOML are resolved relative to the config file, so each pool can stay self-contained inside one workspace. You can still set `raw_dir`, `prepared_dir`, and `reports_root` explicitly, but they now default automatically from the pool id and parsed season.

If those tracker path fields are omitted, they default to:

- `data/<season>/tracker/<pool-id>/raw`
- `data/<season>/tracker/<pool-id>/prepared`
- `reports/<season>/tracker/<pool-id>`

The tracker config is intentionally tracker-only. Keep pre-lock optimizer assumptions out of this file because Bracket Lab and Pool Tracker do not share mutable state.

Launch the integrated app with live tracker data:

```bash
uv run bracket-sim serve \
  --config config/pools.toml \
  --host 127.0.0.1 \
  --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. `--reload` is only supported without `--config`.

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
- `serve` always runs the same integrated app. Pass `--config` only to enable live Pool Tracker data.
