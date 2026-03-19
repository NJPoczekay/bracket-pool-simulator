"""Deterministic Bracket Lab completion helpers."""

from __future__ import annotations

from bracket_sim.domain.bracket_graph import BracketGraph
from bracket_sim.domain.models import EntryPick, PoolEntry, RatingRecord
from bracket_sim.domain.product_models import (
    BracketCompletionResult,
    BracketCompletionState,
    BracketEditPick,
    CompleteBracketRequest,
    CompletionMode,
    EditableBracket,
    PickFourSelection,
)
from bracket_sim.domain.scoring import validate_entries

_MISSING_TEAM_RANK = 1_000_000


def normalize_completion_mode(mode: CompletionMode) -> CompletionMode:
    """Collapse request-visible aliases to the runtime completion mode."""

    if mode == CompletionMode.INTERNAL_MODEL_RANK:
        return CompletionMode.KENPOM
    return mode


def classify_bracket_state(bracket: EditableBracket) -> BracketCompletionState:
    """Classify an editable bracket by completeness and lock state."""

    if any(pick.winner_team_id is None for pick in bracket.picks):
        return BracketCompletionState.INCOMPLETE
    if any(not pick.locked for pick in bracket.picks):
        return BracketCompletionState.AUTO_COMPLETED
    return BracketCompletionState.COMPLETE


def build_initial_bracket(
    *,
    graph: BracketGraph,
    constraints_by_game_id: dict[str, str],
) -> EditableBracket:
    """Build the bootstrap editor state with completed-game constraints locked in."""

    return canonicalize_bracket(
        bracket=EditableBracket(picks=[]),
        graph=graph,
        constraints_by_game_id=constraints_by_game_id,
    )


def canonicalize_bracket(
    *,
    bracket: EditableBracket,
    graph: BracketGraph,
    constraints_by_game_id: dict[str, str],
) -> EditableBracket:
    """Expand a sparse bracket to the full tournament shape and enforce constraints."""

    picks_by_game_id = {pick.game_id: pick for pick in bracket.picks}
    unknown_game_ids = sorted(set(picks_by_game_id) - set(graph.games_by_id))
    if unknown_game_ids:
        msg = f"Bracket references unknown game ids: {unknown_game_ids[:5]}"
        raise ValueError(msg)

    canonical_picks: list[BracketEditPick] = []
    for game_id in ordered_game_ids(graph):
        original = picks_by_game_id.get(game_id)
        winner_team_id = original.winner_team_id if original is not None else None
        locked = original.locked if original is not None else False
        if (
            winner_team_id is not None
            and winner_team_id not in graph.possible_teams_by_game_id[game_id]
        ):
            msg = f"Pick {winner_team_id!r} is not possible in game {game_id}"
            raise ValueError(msg)

        constrained_winner = constraints_by_game_id.get(game_id)
        if constrained_winner is not None:
            if winner_team_id is not None and winner_team_id != constrained_winner:
                msg = f"Game {game_id} is locked to winner {constrained_winner}"
                raise ValueError(msg)
            winner_team_id = constrained_winner
            locked = True

        canonical_picks.append(
            BracketEditPick(
                game_id=game_id,
                winner_team_id=winner_team_id,
                locked=locked,
            )
        )
    return EditableBracket(picks=canonical_picks)


def editable_bracket_to_entry(*, bracket: EditableBracket, graph: BracketGraph) -> PoolEntry:
    """Convert one full editable bracket into a validated simulation entry."""

    pick_map = {pick.game_id: pick.winner_team_id for pick in bracket.picks}
    expected_game_ids = set(graph.games_by_id)
    if set(pick_map) != expected_game_ids:
        missing = sorted(expected_game_ids - set(pick_map))
        extra = sorted(set(pick_map) - expected_game_ids)
        msg = (
            "Bracket must include exactly one pick for every game. "
            f"Missing={missing[:5]} Extra={extra[:5]}"
        )
        raise ValueError(msg)

    missing_winners = sorted(game_id for game_id, team_id in pick_map.items() if team_id is None)
    if missing_winners:
        msg = f"Bracket is incomplete; winner_team_id is required for {missing_winners[:5]}"
        raise ValueError(msg)

    entry = PoolEntry(
        entry_id="user-bracket",
        entry_name="Your Bracket",
        picks=[
            EntryPick(game_id=game_id, winner_team_id=str(pick_map[game_id]))
            for game_id in sorted(pick_map)
        ],
    )
    validate_entries(entries=[entry], graph=graph)
    return entry


def derive_region_champion_game_ids(graph: BracketGraph) -> dict[str, str]:
    """Return the regional-final game id for each region."""

    region_champion_game_ids: dict[str, str] = {}
    for game_id, game in graph.games_by_id.items():
        if game.round != 4:
            continue
        regions = {
            graph.teams_by_id[team_id].region
            for team_id in graph.possible_teams_by_game_id[game_id]
        }
        if len(regions) != 1:
            msg = f"Regional final {game_id} must contain teams from exactly one region"
            raise ValueError(msg)
        region = next(iter(regions))
        if region in region_champion_game_ids:
            msg = f"Duplicate regional final detected for region {region!r}"
            raise ValueError(msg)
        region_champion_game_ids[region] = game_id

    if len(region_champion_game_ids) != 4:
        msg = f"Expected four regional finals, found {sorted(region_champion_game_ids)}"
        raise ValueError(msg)
    return dict(sorted(region_champion_game_ids.items()))


def derive_forced_winners_by_game_id(
    *,
    canonical_bracket: EditableBracket,
    graph: BracketGraph,
    region_champion_game_ids: dict[str, str],
    pick_four: PickFourSelection | None = None,
) -> dict[str, str]:
    """Return all game winners forced by locked picks and Pick Four constraints."""

    forced_winners_by_game_id: dict[str, str] = {}

    def require_winner(*, game_id: str, winner_team_id: str, origin: str) -> None:
        if winner_team_id not in graph.possible_teams_by_game_id[game_id]:
            msg = f"{origin} winner {winner_team_id!r} is not possible in game {game_id}"
            raise ValueError(msg)

        existing = forced_winners_by_game_id.get(game_id)
        if existing is not None and existing != winner_team_id:
            msg = (
                f"Conflicting forced winners for game {game_id}: "
                f"{existing!r} vs {winner_team_id!r}"
            )
            raise ValueError(msg)

        forced_winners_by_game_id[game_id] = winner_team_id

        game = graph.games_by_id[game_id]
        if game.round == 1:
            return

        left_child_id, right_child_id = graph.children_by_game_id[game_id]
        left_contains_winner = winner_team_id in graph.possible_teams_by_game_id[left_child_id]
        right_contains_winner = winner_team_id in graph.possible_teams_by_game_id[right_child_id]
        if left_contains_winner == right_contains_winner:
            msg = f"Could not route winner {winner_team_id!r} through game {game_id}"
            raise RuntimeError(msg)

        require_winner(
            game_id=left_child_id if left_contains_winner else right_child_id,
            winner_team_id=winner_team_id,
            origin=origin,
        )

    for pick in canonical_bracket.picks:
        if pick.locked and pick.winner_team_id is not None:
            require_winner(
                game_id=pick.game_id,
                winner_team_id=pick.winner_team_id,
                origin="Locked pick",
            )

    if pick_four is not None:
        for region, seed in sorted(pick_four.regional_winner_seeds.items()):
            champion_game_id = region_champion_game_ids.get(region)
            if champion_game_id is None:
                msg = f"Pick Four region {region!r} does not exist in the bracket"
                raise ValueError(msg)
            winner_team_id = _team_id_for_region_seed(
                graph=graph,
                region=region,
                seed=seed,
            )
            require_winner(
                game_id=champion_game_id,
                winner_team_id=winner_team_id,
                origin=f"Pick Four region {region}",
            )

    return forced_winners_by_game_id


def complete_bracket(
    *,
    request: CompleteBracketRequest,
    dataset_hash: str,
    graph: BracketGraph,
    constraints_by_game_id: dict[str, str],
    public_pick_weights_by_game: dict[str, dict[str, float]],
    rating_records_by_team_id: dict[str, RatingRecord],
    team_rank_by_team_id: dict[str, int],
    region_champion_game_ids: dict[str, str],
) -> BracketCompletionResult:
    """Auto-complete one bracket while preserving locked picks."""

    runtime_mode = normalize_completion_mode(request.completion_mode)
    if runtime_mode in {CompletionMode.MANUAL, CompletionMode.PICK_FOUR}:
        msg = (
            "completion_mode must be one of: tournament_seeds, popular_picks, "
            "kenpom, internal_model_rank"
        )
        raise ValueError(msg)

    canonical = canonicalize_bracket(
        bracket=request.bracket,
        graph=graph,
        constraints_by_game_id=constraints_by_game_id,
    )
    locked_picks_by_game_id = {
        pick.game_id: pick
        for pick in canonical.picks
        if pick.locked and pick.winner_team_id is not None
    }
    forced_winners_by_game_id = derive_forced_winners_by_game_id(
        canonical_bracket=canonical,
        graph=graph,
        region_champion_game_ids=region_champion_game_ids,
        pick_four=request.pick_four,
    )
    winners_by_game_id: dict[str, str | None] = dict.fromkeys(graph.games_by_id, None)
    winners_by_game_id.update(forced_winners_by_game_id)

    for game_id in graph.topological_game_ids:
        if winners_by_game_id[game_id] is not None:
            continue
        winners_by_game_id[game_id] = select_game_winner(
            game_id=game_id,
            graph=graph,
            winners_by_game_id=winners_by_game_id,
            completion_mode=runtime_mode,
            public_pick_weights_by_game=public_pick_weights_by_game,
            rating_records_by_team_id=rating_records_by_team_id,
            team_rank_by_team_id=team_rank_by_team_id,
        )

    completed_bracket = EditableBracket(
        picks=[
            BracketEditPick(
                game_id=pick.game_id,
                winner_team_id=winners_by_game_id[pick.game_id],
                locked=pick.locked,
            )
            for pick in canonical.picks
        ]
    )
    editable_bracket_to_entry(bracket=completed_bracket, graph=graph)

    return BracketCompletionResult(
        completed_bracket=completed_bracket,
        state=classify_bracket_state(completed_bracket),
        completion_mode=request.completion_mode,
        dataset_hash=dataset_hash,
        preserved_locked_pick_count=len(locked_picks_by_game_id),
        auto_filled_pick_count=sum(not pick.locked for pick in completed_bracket.picks),
    )


def ordered_game_ids(graph: BracketGraph) -> list[str]:
    """Return deterministic round-first ordering for editable brackets."""

    return sorted(
        graph.games_by_id,
        key=lambda game_id: (graph.games_by_id[game_id].round, game_id),
    )


def select_game_winner(
    *,
    game_id: str,
    graph: BracketGraph,
    winners_by_game_id: dict[str, str | None],
    completion_mode: CompletionMode,
    public_pick_weights_by_game: dict[str, dict[str, float]],
    rating_records_by_team_id: dict[str, RatingRecord],
    team_rank_by_team_id: dict[str, int],
) -> str:
    """Select one deterministic winner for a game under the chosen completion mode."""

    available_team_ids = available_team_ids_for_game(
        game_id=game_id,
        graph=graph,
        winners_by_game_id=winners_by_game_id,
    )
    if not available_team_ids:
        msg = f"No available teams remain for game {game_id}"
        raise ValueError(msg)

    if completion_mode == CompletionMode.TOURNAMENT_SEEDS:
        return min(
            available_team_ids,
            key=lambda team_id: _seed_rank_key(
                graph=graph,
                team_id=team_id,
                team_rank_by_team_id=team_rank_by_team_id,
            ),
        )

    if completion_mode == CompletionMode.POPULAR_PICKS:
        pick_weights = public_pick_weights_by_game.get(game_id, {})
        return min(
            available_team_ids,
            key=lambda team_id: (
                -pick_weights.get(team_id, 0.0),
                *_seed_rank_key(
                    graph=graph,
                    team_id=team_id,
                    team_rank_by_team_id=team_rank_by_team_id,
                ),
            ),
        )

    return min(
        available_team_ids,
        key=lambda team_id: (
            -rating_records_by_team_id[team_id].rating,
            *_seed_rank_key(
                graph=graph,
                team_id=team_id,
                team_rank_by_team_id=team_rank_by_team_id,
            ),
        ),
    )


def available_team_ids_for_game(
    *,
    game_id: str,
    graph: BracketGraph,
    winners_by_game_id: dict[str, str | None],
) -> list[str]:
    """Return currently available entrants for one game."""

    game = graph.games_by_id[game_id]
    if game.round == 1:
        assert game.left_team_id is not None
        assert game.right_team_id is not None
        return [game.left_team_id, game.right_team_id]

    left_child_id, right_child_id = graph.children_by_game_id[game_id]
    left_winner = winners_by_game_id[left_child_id]
    right_winner = winners_by_game_id[right_child_id]
    if left_winner is None or right_winner is None:
        msg = f"Game {game_id} cannot be resolved before its child games"
        raise ValueError(msg)
    return [left_winner, right_winner]


def _seed_rank_key(
    *,
    graph: BracketGraph,
    team_id: str,
    team_rank_by_team_id: dict[str, int],
) -> tuple[int, int, str]:
    team = graph.teams_by_id[team_id]
    return (
        team.seed,
        team_rank_by_team_id.get(team_id, _MISSING_TEAM_RANK),
        team_id,
    )


def _team_id_for_region_seed(
    *,
    graph: BracketGraph,
    region: str,
    seed: int,
) -> str:
    matches = [
        team_id
        for team_id, team in graph.teams_by_id.items()
        if team.region == region and team.seed == seed
    ]
    if len(matches) != 1:
        msg = f"Pick Four could not resolve a unique team for region {region!r} seed {seed}"
        raise ValueError(msg)
    return matches[0]
