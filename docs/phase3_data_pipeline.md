# Phase 3 Data Preparation Pipeline

This document describes the canonical offline data-prep flow added in Phase 3.

## Command

```bash
uv run --extra dev python -m bracket_sim.infrastructure.cli.main prepare-data \
  --raw /path/to/raw \
  --out /path/to/normalized
```

The command:

1. loads canonical raw local files
2. resolves aliases to canonical team ids
3. validates bracket, entries, constraints, and rating coverage
4. writes normalized simulation inputs atomically

## Canonical Raw Contract

Required files under `--raw`:

- `teams.csv` with columns: `team_id,name,seed,region`
- `games.csv` with columns: `game_id,round,left_team_id,right_team_id,left_game_id,right_game_id`
- `entries.json` as:
  - list of objects `{entry_id, entry_name, picks}`
  - `picks` is a `game_id -> winner` mapping
  - `winner` may be a canonical team id or alias
- `ratings.csv` with columns: `team,rating,tempo`
  - `team` may be a canonical team name or alias

Optional files under `--raw`:

- `constraints.json` as list of `{game_id, winner}` rows where `winner` may be team id or alias
- `aliases.csv` with columns: `alias,team_id`

## Prepared Output Contract

`--out` contains simulation-ready files:

- `teams.json`
- `games.json`
- `entries.json`
- `constraints.json`
- `ratings.csv`
  - columns: `team_id,rating,tempo`

## Validation Defaults

- Fail-fast: first invalid condition exits with an error and no partial output.
- Duplicate aliases and alias collisions are rejected.
- Unknown aliases in entries, constraints, or ratings are rejected.
- Ratings must cover every canonical team exactly once.
