# National Picks Refresh

This command downloads ESPN's public national pick counts for a Tournament Challenge bracket and stores a local snapshot for later analysis.

## Command

```bash
uv run --extra dev python -m bracket_sim.infrastructure.cli.main refresh-national-picks \
  --challenge tournament-challenge-bracket-2026 \
  --out /path/to/national-picks
```

`--challenge` accepts any of:

- a public bracket URL such as `https://fantasy.espn.com/games/tournament-challenge-bracket-2026/bracket`
- a group URL such as `https://fantasy.espn.com/games/tournament-challenge-bracket-2026/group?id=...`
- a bare challenge key such as `tournament-challenge-bracket-2026`

## Output Contract

The command replaces `--out` atomically and writes:

- `national_picks.csv`
- `metadata.json`
- `snapshots/challenge.json`

`national_picks.csv` columns:

- `game_id`
- `round`
- `display_order`
- `outcome_id`
- `team_id`
- `team_name`
- `seed`
- `region`
- `matchup_position`
- `pick_count`
- `pick_percentage`

`metadata.json` fields:

- `schema_version`
- `challenge_key`
- `challenge_id`
- `challenge_name`
- `challenge_state`
- `fetched_at`
- `source_url`
- `proposition_lock_date`
- `proposition_lock_date_passed`
- `games`
- `rows`
- `total_brackets`
- `round_counts`
- `api_shape_hints`
- `canonical_hash`

## Validation Rules

The refresh fails if ESPN's payload shape is not what the parser expects. v1 validates:

- exactly 63 propositions
- round counts of `32/16/8/4/2/1`
- exactly one `choiceCounter` per outcome
- `scoringFormatId == 5`
- no missing counters
- identical summed `pick_count` totals across every proposition

## Notes

- This feature is API-first. It reads the public challenge JSON and does not use browser automation.
- v1 has no HTML, Playwright, or click-through fallback. If ESPN changes the API shape, the command fails loudly.
- The snapshot is overwrite-latest only. Historical time-series storage is out of scope for this version.
