"""Deterministic Bracket Lab optimizer service."""

from __future__ import annotations

from dataclasses import dataclass

from bracket_sim.application.bracket_completion import (
    available_team_ids_for_game,
    canonicalize_bracket,
    complete_bracket,
    derive_forced_winners_by_game_id,
    editable_bracket_to_entry,
    normalize_completion_mode,
    ordered_game_ids,
    select_game_winner,
)
from bracket_sim.application.bracket_lab_eval import (
    BracketFieldEvaluationContext,
    BracketLabRuntime,
    build_bracket_diagnostics,
    build_field_evaluation_context,
    derive_child_seed,
    seed_from_cache_key,
)
from bracket_sim.domain.product_models import (
    BracketPickChange,
    CompleteBracketRequest,
    CompletionMode,
    EditableBracket,
    OptimizationAlternative,
    OptimizationResult,
    OptimizeBracketRequest,
    PickDiagnostic,
    PickDiagnosticTag,
)
from bracket_sim.infrastructure.storage.cache_keys import build_cache_key

_OPTIMIZER_COARSE_N_SIMS = 4_000
_OPTIMIZER_FINAL_N_SIMS = 12_000
_OPTIMIZER_BEAM_WIDTH = 12
_OPTIMIZER_LOCAL_STARTS = 4
_OPTIMIZER_MAX_PRIORITY_GAMES = 20
_SUMMARY_CHANGE_LIMIT = 2


@dataclass(frozen=True)
class _RankedBracket:
    bracket: EditableBracket
    projected_win_probability: float


def optimize_bracket(
    *,
    request: OptimizeBracketRequest,
    runtime: BracketLabRuntime,
) -> OptimizationResult:
    """Optimize a complete user bracket against a fixed public field."""

    if request.completion_mode == CompletionMode.PICK_FOUR:
        msg = "completion_mode='pick_four' is only valid as a completion helper"
        raise ValueError(msg)

    canonical_bracket = canonicalize_bracket(
        bracket=request.bracket,
        graph=runtime.graph,
        constraints_by_game_id=runtime.constraints_by_game_id,
    )
    _require_complete_bracket(canonical_bracket=canonical_bracket, runtime=runtime)

    normalized_completion_mode = normalize_completion_mode(request.completion_mode)
    forced_winners_by_game_id = derive_forced_winners_by_game_id(
        canonical_bracket=canonical_bracket,
        graph=runtime.graph,
        region_champion_game_ids=runtime.region_champion_game_ids,
        pick_four=request.pick_four,
    )
    repair_mode = (
        normalized_completion_mode
        if normalized_completion_mode != CompletionMode.MANUAL
        else CompletionMode.KENPOM
    )

    cache_key = build_cache_key(
        artifact_kind="optimization",
        dataset_hash=runtime.dataset_hash,
        settings={
            "bracket": canonical_bracket,
            "pool_settings": request.pool_settings,
            "completion_mode": normalized_completion_mode,
            "pick_four": request.pick_four,
        },
    )
    base_seed = seed_from_cache_key(cache_key)
    coarse_context = build_field_evaluation_context(
        runtime=runtime,
        pool_settings=request.pool_settings,
        opponent_seed=derive_child_seed(seed=base_seed, child_index=1),
        simulation_seed=derive_child_seed(seed=base_seed, child_index=2),
        n_sims=_OPTIMIZER_COARSE_N_SIMS,
    )
    final_context = build_field_evaluation_context(
        runtime=runtime,
        pool_settings=request.pool_settings,
        opponent_seed=derive_child_seed(seed=base_seed, child_index=3),
        simulation_seed=derive_child_seed(seed=base_seed, child_index=4),
        n_sims=_OPTIMIZER_FINAL_N_SIMS,
    )

    _, diagnostics = build_bracket_diagnostics(
        context=final_context,
        bracket=canonical_bracket,
    )
    priority_game_ids = prioritize_mutation_games(
        bracket=canonical_bracket,
        diagnostics=diagnostics,
        mutable_game_ids=[
            game_id
            for game_id in ordered_game_ids(runtime.graph)
            if game_id not in forced_winners_by_game_id
        ],
        max_games=_OPTIMIZER_MAX_PRIORITY_GAMES,
        runtime=runtime,
    )

    seed_brackets = build_seed_brackets(
        request=request,
        canonical_bracket=canonical_bracket,
        runtime=runtime,
    )
    beam_ranked = beam_search(
        seed_brackets=seed_brackets,
        priority_game_ids=priority_game_ids,
        forced_winners_by_game_id=forced_winners_by_game_id,
        repair_mode=repair_mode,
        context=coarse_context,
        runtime=runtime,
    )
    local_ranked = [
        climb_locally(
            ranked_bracket=ranked_bracket,
            priority_game_ids=priority_game_ids,
            forced_winners_by_game_id=forced_winners_by_game_id,
            repair_mode=repair_mode,
            context=coarse_context,
            runtime=runtime,
        )
        for ranked_bracket in beam_ranked[:_OPTIMIZER_LOCAL_STARTS]
    ]

    final_candidates = dedupe_brackets(
        [
            canonical_bracket,
            *seed_brackets,
            *(ranked.bracket for ranked in beam_ranked),
            *(ranked.bracket for ranked in local_ranked),
        ]
    )
    final_win_probabilities, _ = final_context.evaluate_brackets(final_candidates)
    final_ranked = sorted(
        (
            _RankedBracket(bracket=bracket, projected_win_probability=float(win_probability))
            for bracket, win_probability in zip(
                final_candidates,
                final_win_probabilities,
                strict=True,
            )
        ),
        key=lambda ranked: (ranked.projected_win_probability, bracket_signature(ranked.bracket)),
        reverse=True,
    )

    recommended = final_ranked[0]
    recommended_changes = build_pick_changes(
        baseline=canonical_bracket,
        candidate=recommended.bracket,
        runtime=runtime,
    )
    alternatives = build_diverse_alternatives(
        ranked_candidates=final_ranked[1:],
        baseline=canonical_bracket,
        top_bracket=recommended.bracket,
        runtime=runtime,
        limit=3,
    )

    return OptimizationResult(
        pool_settings=request.pool_settings,
        completion_mode=request.completion_mode,
        dataset_hash=runtime.dataset_hash,
        cache_key=cache_key,
        recommended_bracket=recommended.bracket,
        projected_win_probability=recommended.projected_win_probability,
        changed_pick_count=len(recommended_changes),
        changed_picks=recommended_changes,
        summary=summarize_pick_changes(recommended_changes),
        alternatives=alternatives,
    )


def build_seed_brackets(
    *,
    request: OptimizeBracketRequest,
    canonical_bracket: EditableBracket,
    runtime: BracketLabRuntime,
) -> list[EditableBracket]:
    """Seed the optimizer from the current bracket and completion-mode variants."""

    locked_only_bracket = EditableBracket(
        picks=[pick for pick in canonical_bracket.picks if pick.locked]
    )
    seeds: list[EditableBracket] = [canonical_bracket]
    for mode in (
        CompletionMode.TOURNAMENT_SEEDS,
        CompletionMode.POPULAR_PICKS,
        CompletionMode.KENPOM,
    ):
        completion = complete_bracket(
            request=CompleteBracketRequest(
                bracket=locked_only_bracket,
                completion_mode=mode,
                pick_four=request.pick_four,
            ),
            dataset_hash=runtime.dataset_hash,
            graph=runtime.graph,
            constraints_by_game_id=runtime.constraints_by_game_id,
            public_pick_weights_by_game=runtime.public_pick_weights_by_game,
            rating_records_by_team_id=runtime.rating_records_by_team_id,
            team_rank_by_team_id=runtime.team_rank_by_team_id,
            region_champion_game_ids=runtime.region_champion_game_ids,
        )
        seeds.append(completion.completed_bracket)
    return dedupe_brackets(seeds)


def prioritize_mutation_games(
    *,
    bracket: EditableBracket,
    diagnostics: list[PickDiagnostic],
    mutable_game_ids: list[str],
    max_games: int,
    runtime: BracketLabRuntime,
) -> list[str]:
    """Rank mutable games using starting diagnostics and late-round leverage."""

    diagnostic_by_game_id = {diagnostic.game_id: diagnostic for diagnostic in diagnostics}
    picks_by_game_id = {pick.game_id: pick for pick in bracket.picks}
    prioritized = sorted(
        mutable_game_ids,
        key=lambda game_id: (
            -runtime.graph.games_by_id[game_id].round,
            0
            if PickDiagnosticTag.MOST_IMPORTANT in diagnostic_by_game_id[game_id].tags
            else 1,
            diagnostic_by_game_id[game_id].delta_win_probability_if_picked,
            picks_by_game_id[game_id].winner_team_id or "",
            game_id,
        ),
    )
    return prioritized[:max_games]


def beam_search(
    *,
    seed_brackets: list[EditableBracket],
    priority_game_ids: list[str],
    forced_winners_by_game_id: dict[str, str],
    repair_mode: CompletionMode,
    context: BracketFieldEvaluationContext,
    runtime: BracketLabRuntime,
) -> list[_RankedBracket]:
    """Run a small beam search over high-priority mutable games."""

    beam = rank_brackets(
        brackets=seed_brackets,
        context=context,
    )[:_OPTIMIZER_BEAM_WIDTH]
    for game_id in priority_game_ids:
        next_candidates = [ranked.bracket for ranked in beam]
        for ranked in beam:
            mutated = mutate_bracket(
                bracket=ranked.bracket,
                game_id=game_id,
                forced_winners_by_game_id=forced_winners_by_game_id,
                repair_mode=repair_mode,
                runtime=runtime,
            )
            if mutated is not None:
                next_candidates.append(mutated)
        beam = rank_brackets(
            brackets=dedupe_brackets(next_candidates),
            context=context,
        )[:_OPTIMIZER_BEAM_WIDTH]
    return beam


def climb_locally(
    *,
    ranked_bracket: _RankedBracket,
    priority_game_ids: list[str],
    forced_winners_by_game_id: dict[str, str],
    repair_mode: CompletionMode,
    context: BracketFieldEvaluationContext,
    runtime: BracketLabRuntime,
) -> _RankedBracket:
    """Run best-improvement local search from one beam candidate."""

    current = ranked_bracket
    while True:
        candidates: list[EditableBracket] = []
        for game_id in priority_game_ids:
            mutated = mutate_bracket(
                bracket=current.bracket,
                game_id=game_id,
                forced_winners_by_game_id=forced_winners_by_game_id,
                repair_mode=repair_mode,
                runtime=runtime,
            )
            if mutated is not None:
                candidates.append(mutated)
        if not candidates:
            return current

        ranked_candidates = rank_brackets(
            brackets=dedupe_brackets(candidates),
            context=context,
        )
        best = ranked_candidates[0]
        if best.projected_win_probability <= current.projected_win_probability:
            return current
        current = best


def rank_brackets(
    *,
    brackets: list[EditableBracket],
    context: BracketFieldEvaluationContext,
) -> list[_RankedBracket]:
    """Score candidate brackets under a fixed common-random evaluation context."""

    if not brackets:
        return []
    win_probabilities, _ = context.evaluate_brackets(brackets)
    return sorted(
        (
            _RankedBracket(bracket=bracket, projected_win_probability=float(win_probability))
            for bracket, win_probability in zip(brackets, win_probabilities, strict=True)
        ),
        key=lambda ranked: (ranked.projected_win_probability, bracket_signature(ranked.bracket)),
        reverse=True,
    )


def mutate_bracket(
    *,
    bracket: EditableBracket,
    game_id: str,
    forced_winners_by_game_id: dict[str, str],
    repair_mode: CompletionMode,
    runtime: BracketLabRuntime,
) -> EditableBracket | None:
    """Flip one game winner and deterministically repair any downstream conflicts."""

    if game_id in forced_winners_by_game_id:
        return None

    winners_by_game_id = {
        pick.game_id: pick.winner_team_id
        for pick in bracket.picks
    }
    available_team_ids = available_team_ids_for_game(
        game_id=game_id,
        graph=runtime.graph,
        winners_by_game_id=winners_by_game_id,
    )
    current_winner = winners_by_game_id[game_id]
    alternatives = [
        team_id
        for team_id in available_team_ids
        if team_id != current_winner
    ]
    if not alternatives:
        return None

    winners_by_game_id[game_id] = alternatives[0]
    current_game_id = game_id
    while runtime.graph.parents_by_game_id[current_game_id]:
        parent_game_id = runtime.graph.parents_by_game_id[current_game_id][0]
        forced_parent_winner = forced_winners_by_game_id.get(parent_game_id)
        if forced_parent_winner is not None:
            winners_by_game_id[parent_game_id] = forced_parent_winner
            current_game_id = parent_game_id
            continue

        available_parent_team_ids = available_team_ids_for_game(
            game_id=parent_game_id,
            graph=runtime.graph,
            winners_by_game_id=winners_by_game_id,
        )
        if winners_by_game_id[parent_game_id] not in available_parent_team_ids:
            winners_by_game_id[parent_game_id] = select_game_winner(
                game_id=parent_game_id,
                graph=runtime.graph,
                winners_by_game_id=winners_by_game_id,
                completion_mode=repair_mode,
                public_pick_weights_by_game=runtime.public_pick_weights_by_game,
                rating_records_by_team_id=runtime.rating_records_by_team_id,
                team_rank_by_team_id=runtime.team_rank_by_team_id,
            )
        current_game_id = parent_game_id

    mutated = EditableBracket(
        picks=[
            pick.model_copy(update={"winner_team_id": winners_by_game_id[pick.game_id]})
            for pick in bracket.picks
        ]
    )
    if bracket_signature(mutated) == bracket_signature(bracket):
        return None
    _require_complete_bracket(canonical_bracket=mutated, runtime=runtime)
    return mutated


def build_diverse_alternatives(
    *,
    ranked_candidates: list[_RankedBracket],
    baseline: EditableBracket,
    top_bracket: EditableBracket,
    runtime: BracketLabRuntime,
    limit: int,
) -> list[OptimizationAlternative]:
    """Return high-quality alternatives with unique regional-winner tuples."""

    top_region_tuple = region_winner_tuple(bracket=top_bracket, runtime=runtime)
    seen_region_tuples = {top_region_tuple}
    alternatives: list[OptimizationAlternative] = []

    for ranked in ranked_candidates:
        candidate_region_tuple = region_winner_tuple(bracket=ranked.bracket, runtime=runtime)
        if candidate_region_tuple in seen_region_tuples:
            continue
        if candidate_region_tuple == top_region_tuple:
            continue

        changes = build_pick_changes(
            baseline=baseline,
            candidate=ranked.bracket,
            runtime=runtime,
        )
        seen_region_tuples.add(candidate_region_tuple)
        alternatives.append(
            OptimizationAlternative(
                label=f"Alternative {len(alternatives) + 1}",
                bracket=ranked.bracket,
                projected_win_probability=ranked.projected_win_probability,
                changed_pick_count=len(changes),
                changed_picks=changes,
                summary=summarize_pick_changes(changes),
            )
        )
        if len(alternatives) >= limit:
            break

    return alternatives


def build_pick_changes(
    *,
    baseline: EditableBracket,
    candidate: EditableBracket,
    runtime: BracketLabRuntime,
) -> list[BracketPickChange]:
    """Return explicit pick-level differences sorted by round importance."""

    baseline_by_game_id = {pick.game_id: pick for pick in baseline.picks}
    candidate_by_game_id = {pick.game_id: pick for pick in candidate.picks}
    changes: list[BracketPickChange] = []
    for game_id in runtime.graph.games_by_id:
        baseline_winner = baseline_by_game_id[game_id].winner_team_id
        candidate_winner = candidate_by_game_id[game_id].winner_team_id
        if baseline_winner == candidate_winner:
            continue
        if baseline_winner is None or candidate_winner is None:
            msg = f"Optimizer change comparison requires complete brackets for game {game_id}"
            raise ValueError(msg)
        changes.append(
            BracketPickChange(
                game_id=game_id,
                round=runtime.graph.games_by_id[game_id].round,
                from_team_id=baseline_winner,
                from_team_name=runtime.graph.teams_by_id[baseline_winner].name,
                to_team_id=candidate_winner,
                to_team_name=runtime.graph.teams_by_id[candidate_winner].name,
            )
        )
    return sorted(
        changes,
        key=lambda change: (-change.round, change.game_id),
    )


def summarize_pick_changes(changes: list[BracketPickChange]) -> str:
    """Build a short human-readable summary from the highest-impact pick changes."""

    if not changes:
        return "No pick changes from your current bracket."

    highlights = [
        f"{round_label(change.round)}: {change.from_team_name} -> {change.to_team_name}"
        for change in changes[:_SUMMARY_CHANGE_LIMIT]
    ]
    if len(changes) <= _SUMMARY_CHANGE_LIMIT:
        return "; ".join(highlights)
    remaining = len(changes) - _SUMMARY_CHANGE_LIMIT
    suffix = "change" if remaining == 1 else "changes"
    return f"{'; '.join(highlights)}; {remaining} more {suffix}"


def round_label(round_number: int) -> str:
    labels = {
        1: "Round of 64",
        2: "Round of 32",
        3: "Sweet 16",
        4: "Elite 8",
        5: "Final Four",
        6: "Championship",
    }
    return labels.get(round_number, f"Round {round_number}")


def dedupe_brackets(brackets: list[EditableBracket]) -> list[EditableBracket]:
    """Keep brackets in order while dropping duplicates by pick signature."""

    deduped: list[EditableBracket] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for bracket in brackets:
        signature = bracket_signature(bracket)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(bracket)
    return deduped


def bracket_signature(bracket: EditableBracket) -> tuple[str, ...]:
    return tuple(pick.winner_team_id or "" for pick in bracket.picks)


def region_winner_tuple(
    *,
    bracket: EditableBracket,
    runtime: BracketLabRuntime,
) -> tuple[str, ...]:
    picks_by_game_id = {pick.game_id: pick.winner_team_id or "" for pick in bracket.picks}
    return tuple(
        picks_by_game_id[game_id]
        for _, game_id in sorted(runtime.region_champion_game_ids.items())
    )


def _require_complete_bracket(
    *,
    canonical_bracket: EditableBracket,
    runtime: BracketLabRuntime,
) -> None:
    try:
        editable_bracket_to_entry(bracket=canonical_bracket, graph=runtime.graph)
    except ValueError as exc:
        raise ValueError(f"Optimizer requires a complete bracket: {exc}") from exc
