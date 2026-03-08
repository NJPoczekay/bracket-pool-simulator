# Bracket Pool Simulator: Core Logic and Key Features

This project simulates NCAA tournament outcomes and estimates each ESPN bracket entry's chance to win a specific pool.

## Core Logic

1. Build the current tournament state from three sources:
   - Bracket structure and teams
   - Completed game results
   - KenPom team ratings
2. Load bracket entries for a pool (from local HTML snapshots, or scrape them if missing).
3. Run many Monte Carlo tournament simulations.
4. Score each entry on each simulated tournament.
5. Compute how often each entry wins the pool (including tie splits) and report win odds.

## Key Features and How They Are Implemented

- Pool entry ingestion and parsing
  - Entry HTML files are parsed with regex.
  - For each team, the number of times it appears in an entry's picks indicates how far that entry picked the team to advance.
  - A team picked six times is treated as that entry's champion.

- Automatic pool scraping
  - If local entry files are not present, the tool visits the ESPN group page, collects bracket links, opens each entry, and saves the HTML locally.
  - This creates a reusable local cache of pool entries under `html_sources/<groupID>/`.

- Team name normalization
  - A mapping file reconciles naming differences across ESPN pages, KenPom files, and scoreboard files.
  - This avoids mismatches during scoring and simulation.

- Partial-tournament conditioning
  - Completed real games are read from a scoreboard file.
  - Teams are assigned minimum guaranteed wins so already-finished outcomes are enforced in every simulation.
  - The `day` and `hour` inputs control how much of the real tournament is locked in.

- Win-probability model for each game
  - Matchup strength is calculated from KenPom efficiency difference and tempo.
  - A polynomial converts that spread into a win probability.
  - Each game outcome is sampled stochastically from that probability.

- Bracket progression simulation
  - Teams are paired each round and winners advance to face the next opponent.
  - Losers are marked eliminated at the round they lose.
  - Surviving final team gets championship win depth.
  - Each simulation stores per-team total wins in compact binary format.

- Fast, vectorized entry scoring
  - For each team in each simulation, points are based on `2^min(predicted_depth, actual_wins) - 1`.
  - This reproduces ESPN-style escalating round value.
  - Team points are summed to produce each entry's total simulated score.

- Pool winner determination and tie handling
  - For each simulation, entries with top score are identified.
  - If multiple entries tie for first, they split credit equally.
  - Aggregated credit across all simulations becomes each entry's estimated pool win probability.

- Caching and repeatability
  - Tournament simulation outputs are saved to disk and reused.
  - Pool result tables can also be stored to avoid recomputation.
  - A refresh option can force regeneration of simulation data.

- Data update utilities
  - A scoreboard updater scrapes completed NCAA tournament results and writes time-indexed winner logs.
  - A KenPom updater scrapes current ratings and tempo and writes dated rating files.

- Team-level diagnostic reporting
  - A separate simulation report can show team outcome distributions (for example, odds of at least N wins).
  - This is useful for validating the tournament model independently of pool scoring.
