# Bracket Tools Product Roadmap

## Goal

Ship BracketVoodoo-style bracket tools in manageable stages on top of the current Python engine:
- simple web app + API
- national-public-pick based analysis
- bracket completion and optimization
- strategy outputs for pool size and scoring systems

## Scope Defaults

- Product surface: API + simple web app
- Opponent model: national public picks only
- Account system: out of scope
- Billing/tiers: out of scope
- Standalone year-round rankings/projections product: out of scope
- Tournament-specific bracket tools: in scope

## Integration Note

`Bracket Lab` now lives inside the same local app shell as `Pool Tracker`, but the two workflows intentionally keep separate state:

- `Bracket Lab` is pre-tournament and exploratory.
- `Pool Tracker` is post-lock and tied to real pool entries.
- Tracker configuration is not the source of truth for optimizer assumptions such as eventual pool size.

## Phase 0: Product Foundation

### Deliverables
- Integrated web/API architecture added alongside the existing CLI
- Shared typed models for bracket editing, pool settings, scoring systems, analysis results, and optimization results
- Consistent cache and manifest strategy for reusable simulation artifacts
- Workflow metadata so the app can present `Bracket Lab` first and `Pool Tracker` second
- Updated docs describing app layers and the new staged roadmap

### Issues
- Add a lightweight server entrypoint and integrated app shell
- Define `PoolSettings`, `ScoringSystem`, `CompletionMode`, `BracketAnalysis`, `PickDiagnostic`, and `OptimizationResult`
- Separate reusable engine services from CLI formatting concerns
- Define dataset hash and cache key rules for analysis and optimization
- Keep optimizer/planning state separate from live pool-tracker configuration

### Exit Criteria
- Repo supports CLI plus a running local web/API surface with Bracket Lab and Pool Tracker side by side
- New types and service boundaries are stable enough for later phases
- No product logic is trapped in the UI layer

## Phase 1: Data Enrichment

### Deliverables
- Public-pick ingestion from ESPN Tournament Challenge data
- Support datasets for bracket completion modes
- Prepared data artifacts that include everything needed for analysis and optimization

### Issues
- Extend provider contracts to normalize per-outcome public pick percentages
- Persist public-pick data through Bracket Lab-specific refresh/prepare commands
- Add support for completion/ranking inputs: tournament seeds, public picks, and KenPom
- Treat `internal model rank` as an alias of `KenPom` until a distinct internal model exists
- Fail preparation when a required KenPom row is missing for a concrete team or unresolved play-in candidate
- Preserve play-in placeholder handling until First Four results resolve those slots

### Exit Criteria
- Prepared datasets contain public pick percentages and completion-mode inputs
- Refresh and prepare flows remain deterministic and fixture-testable
- Tournament field can be analyzed without ad hoc external calls

## Phase 2: Analyzer MVP

### Deliverables
- Full-bracket analysis endpoint and UI flow
- Pool-aware win probability using national public opponents
- Per-pick survival and value diagnostics

### Issues
- Add scoring support for `1-2-4-8-16-32`, `1-2-3-4-5-6`, `2-3-5-8-13-21`, and `round+seed`
- Build a sampled "average pool" opponent generator from public-pick data
- Add `analyze_bracket` service that returns:
  - chance to win
  - percentile versus similar public brackets
  - pick-level survival probability
  - `probWinIf` and `deltaWinIf`
  - tags for `bestPick`, `worstPick`, and `mostImportant`
- Build a simple Bracket Lab page with pool-size and scoring controls plus analyze action

### Exit Criteria
- A user can enter a full bracket and get a stable analysis result in the browser
- Analysis is deterministic for fixed inputs and seeds
- Pick diagnostics are internally consistent with the underlying simulation results

## Phase 3: Completion Tools

### Deliverables
- Partial-bracket completion modes
- Auto-fill workflow modeled after BracketVoodoo's "finish picks using" flow
- Pick Four wizard for forcing Final Four seeds before auto-completion

### Issues
- Add completion modes for Tournament Seeds, Popular Picks, internal model rank, and KenPom
- Support partial brackets with user-locked picks
- Add bracket validation for incomplete, complete, and auto-completed states
- Implement Pick Four helper that seeds semifinal/final search from chosen regional winners
- Show completion results before analysis/optimization without mutating locked picks

### Exit Criteria
- A user can start from an empty or partial bracket and legally complete it with any supported mode
- Locked picks are always preserved
- Completed brackets can immediately flow into analysis

## Phase 4: Optimizer MVP

### Deliverables
- Pool-size and scoring-aware optimizer
- Locked-pick optimization around user-selected gambits
- Multiple strong bracket alternatives instead of a single opaque result

### Issues
- Add `optimize_bracket` service using beam search plus local search over legal bracket states
- Use common-random-number evaluation so candidate comparisons are stable
- Seed optimizer from analyzed bracket, completion modes, and Pick Four configurations
- Return:
  - best bracket found
  - projected win probability
  - 2-3 diverse alternative gambits
  - changed picks summary relative to the user's current bracket
- Define a diversity rule so alternatives are meaningfully different, not tiny variants

### Exit Criteria
- Optimizer consistently beats simple public or chalk baselines in fixed fixtures
- Locked picks and pool settings materially affect results
- Users can optimize around their own favorite picks without losing control of the bracket

## Phase 5: Strategy Outputs

### Deliverables
- Strategy panel or page with tournament-wide tables
- Team advancement odds
- Over/under-picked value index
- Recommended gambits by pool type

### Issues
- Build `build_strategy_insights` service returning:
  - round-by-round advancement odds
  - public-pick versus win-probability value index
  - recommended gambits for small, medium, large, and upset pools
- Reuse the same public-pick and simulation base as analyzer/optimizer
- Add exportable JSON payloads for future report generation
- Keep this scoped to bracket strategy only, not general betting or daily matchup content

### Exit Criteria
- Strategy outputs are derived from the same engine as the analyzer and optimizer
- Users can understand why certain teams or gambits are valuable for their pool type
- No duplicate logic exists between strategy tables and bracket analysis

## Phase 6: Hardening And UX Polish

### Deliverables
- Faster interactive performance through caching and precomputation
- Better error handling and diagnostics
- End-to-end confidence for tournament-week usage

### Issues
- Cache simulation bases and sampled public-opponent catalogs by dataset hash and settings
- Add API contract tests and browser flow tests
- Add performance budgets for analyze and optimize operations
- Improve validation and recovery for missing data, unsupported scoring inputs, and stale prepared datasets
- Document an operator runbook for refresh, prepare, serve, and regression checks

### Exit Criteria
- Analyze remains interactive for normal pool sizes
- Optimize is slow-but-reliable rather than fragile
- Tournament-week workflow is documented and repeatable

## Test Plan

- Unit tests for scoring systems, public-pick normalization, completion legality, play-in placeholders, and pick-tag calculations
- Integration tests for refresh/prepare with public-pick data and ranking inputs
- Golden tests for deterministic analyzer outputs
- Optimizer regression tests showing improvement over baseline brackets
- API tests for analyze, complete, optimize, and strategy endpoints
- Browser tests for create/edit/analyze/optimize flows and local draft persistence

## Assumptions

- Local browser storage is sufficient for saved drafts in v1
- Actual ESPN pool-entry imports are deferred to a later roadmap
- Futures-bet comparisons and regular-season game projections are deferred
- The current simulator remains the canonical tournament engine and is extended rather than replaced
