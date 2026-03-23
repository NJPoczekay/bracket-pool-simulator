"""Build tonight-focused tracker viewing guides from report artifacts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, tzinfo
from pathlib import Path

from bracket_sim.domain.models import (
    EntryTopGameSummaryRow,
    EntryViewingGuideRow,
    Game,
    TonightWatchlistItem,
    TrackerViewingGuide,
    ViewingGuideEntryOption,
)
from bracket_sim.infrastructure.storage._file_io import load_required_csv_rows
from bracket_sim.infrastructure.storage.normalized_loader import load_normalized_input

_ENTRY_SUMMARY_REQUIRED_COLUMNS = {
    "rank",
    "entry_id",
    "entry_name",
    "win_percentage",
}
_GAME_OUTCOME_REQUIRED_COLUMNS = {
    "game_id",
    "round",
    "game_label",
    "outcome_team_id",
    "outcome_team_name",
    "outcome_probability",
    "entry_id",
    "baseline_win_percentage",
    "conditional_win_percentage",
    "win_percentage_point_delta",
    "outcome_total_win_percentage_point_swing",
}


@dataclass(frozen=True)
class _TonightGameContext:
    game_id: str
    round: int
    game_label: str
    matchup: str
    tipoff_local: datetime
    tipoff_local_label: str


@dataclass(frozen=True)
class _GameOutcomeRow:
    game_id: str
    round: int
    game_label: str
    outcome_team_id: str
    outcome_team_name: str
    outcome_probability: float
    entry_id: str
    baseline_win_percentage: float
    conditional_win_percentage: float
    win_percentage_point_delta: float
    outcome_total_swing: float


def build_tracker_viewing_guide(
    *,
    input_dir: Path,
    report_dir: Path,
    now: datetime,
    timezone: tzinfo,
) -> TrackerViewingGuide:
    """Return tonight's watchlist plus per-entry rooting guides."""

    if now.tzinfo is None:
        msg = "now must be timezone-aware"
        raise ValueError(msg)

    local_now = now.astimezone(timezone)
    timezone_name = _timezone_name(timezone=timezone, reference=local_now)
    normalized = load_normalized_input(input_dir)
    entry_options = _load_entry_options(report_dir / "entry_summary.csv")
    if not entry_options:
        msg = "entry_summary.csv does not contain any entries"
        raise ValueError(msg)

    team_names_by_id = {team.team_id: team.name for team in normalized.teams}
    games_by_id = {game.game_id: game for game in normalized.games}
    constraints_by_game_id = {
        constraint.game_id: constraint.winner_team_id for constraint in normalized.constraints
    }
    tonight_games_by_id, unavailable_schedule_count = _build_tonight_games_by_id(
        games=normalized.games,
        games_by_id=games_by_id,
        constraints_by_game_id=constraints_by_game_id,
        team_names_by_id=team_names_by_id,
        local_now=local_now,
        timezone=timezone,
    )
    outcome_rows = _load_game_outcome_rows(report_dir / "game_outcome_sensitivity.csv")
    tonight_rows = [row for row in outcome_rows if row.game_id in tonight_games_by_id]

    entry_options_by_id = {entry.entry_id: entry for entry in entry_options}
    watchlist = _build_watchlist(
        tonight_rows=tonight_rows,
        tonight_games_by_id=tonight_games_by_id,
        entry_options_by_id=entry_options_by_id,
    )
    guides_by_entry_id, top_games_by_entry = _build_entry_guides(
        entry_options=entry_options,
        tonight_rows=tonight_rows,
        tonight_games_by_id=tonight_games_by_id,
    )

    return TrackerViewingGuide(
        local_date=local_now.date().isoformat(),
        timezone=timezone_name,
        default_entry_id=entry_options[0].entry_id,
        unavailable_schedule_count=unavailable_schedule_count,
        watchlist=watchlist,
        entry_options=entry_options,
        guides_by_entry_id=guides_by_entry_id,
        top_games_by_entry=top_games_by_entry,
    )


def _load_entry_options(path: Path) -> list[ViewingGuideEntryOption]:
    rows, fieldnames = load_required_csv_rows(
        path,
        missing_prefix="Required report artifact is missing",
    )
    missing = sorted(_ENTRY_SUMMARY_REQUIRED_COLUMNS - set(fieldnames))
    if missing:
        msg = f"entry_summary.csv is missing required columns: {missing}"
        raise ValueError(msg)

    loaded_rows = [
        ViewingGuideEntryOption(
            rank=int(row["rank"]),
            entry_id=row["entry_id"],
            entry_name=row["entry_name"],
            win_percentage=float(row["win_percentage"]),
        )
        for row in rows
        if row.get("entry_id")
    ]
    return sorted(
        loaded_rows,
        key=lambda row: (row.rank, row.entry_name.casefold(), row.entry_id),
    )


def _load_game_outcome_rows(path: Path) -> list[_GameOutcomeRow]:
    rows, fieldnames = load_required_csv_rows(
        path,
        missing_prefix="Required report artifact is missing",
    )
    missing = sorted(_GAME_OUTCOME_REQUIRED_COLUMNS - set(fieldnames))
    if missing:
        msg = f"game_outcome_sensitivity.csv is missing required columns: {missing}"
        raise ValueError(msg)

    return [
        _GameOutcomeRow(
            game_id=row["game_id"],
            round=int(row["round"]),
            game_label=row["game_label"],
            outcome_team_id=row["outcome_team_id"],
            outcome_team_name=row["outcome_team_name"],
            outcome_probability=float(row["outcome_probability"]),
            entry_id=row["entry_id"],
            baseline_win_percentage=float(row["baseline_win_percentage"]),
            conditional_win_percentage=float(row["conditional_win_percentage"]),
            win_percentage_point_delta=float(row["win_percentage_point_delta"]),
            outcome_total_swing=float(row["outcome_total_win_percentage_point_swing"]),
        )
        for row in rows
        if row.get("game_id") and row.get("entry_id")
    ]


def _build_tonight_games_by_id(
    *,
    games: list[Game],
    games_by_id: dict[str, Game],
    constraints_by_game_id: dict[str, str],
    team_names_by_id: dict[str, str],
    local_now: datetime,
    timezone: tzinfo,
) -> tuple[dict[str, _TonightGameContext], int]:
    tonight_games_by_id: dict[str, _TonightGameContext] = {}
    unavailable_schedule_count = 0
    for game in games:
        if game.game_id in constraints_by_game_id:
            continue
        if game.scheduled_at_utc is None:
            unavailable_schedule_count += 1
            continue

        local_tipoff = game.scheduled_at_utc.astimezone(timezone)
        if local_tipoff.date() != local_now.date():
            continue

        tonight_games_by_id[game.game_id] = _TonightGameContext(
            game_id=game.game_id,
            round=game.round,
            game_label=_game_label(game),
            matchup=_build_matchup_label(
                game=game,
                games_by_id=games_by_id,
                constraints_by_game_id=constraints_by_game_id,
                team_names_by_id=team_names_by_id,
            ),
            tipoff_local=local_tipoff,
            tipoff_local_label=_format_tipoff(local_tipoff),
        )
    return tonight_games_by_id, unavailable_schedule_count


def _build_watchlist(
    *,
    tonight_rows: list[_GameOutcomeRow],
    tonight_games_by_id: dict[str, _TonightGameContext],
    entry_options_by_id: dict[str, ViewingGuideEntryOption],
) -> list[TonightWatchlistItem]:
    rows_by_outcome: dict[tuple[str, str], list[_GameOutcomeRow]] = defaultdict(list)
    for row in tonight_rows:
        rows_by_outcome[(row.game_id, row.outcome_team_id)].append(row)

    outcome_groups_by_game: dict[str, list[list[_GameOutcomeRow]]] = defaultdict(list)
    for (game_id, _outcome_team_id), grouped_rows in rows_by_outcome.items():
        outcome_groups_by_game[game_id].append(grouped_rows)

    unsorted_items: list[TonightWatchlistItem] = []
    for game_id, outcome_groups in outcome_groups_by_game.items():
        context = tonight_games_by_id.get(game_id)
        if context is None:
            continue

        pivotal_group = max(
            outcome_groups,
            key=lambda group: (
                group[0].outcome_total_swing,
                group[0].outcome_probability,
                group[0].outcome_team_name,
                group[0].outcome_team_id,
            ),
        )
        representative = pivotal_group[0]
        top_gainer = max(
            pivotal_group,
            key=lambda row: (
                row.win_percentage_point_delta,
                row.outcome_probability,
                row.entry_id,
            ),
        )
        top_loser = min(
            pivotal_group,
            key=lambda row: (
                row.win_percentage_point_delta,
                row.outcome_probability,
                row.entry_id,
            ),
        )
        unsorted_items.append(
            TonightWatchlistItem(
                rank=1,
                game_id=game_id,
                round=context.round,
                game_label=context.game_label,
                matchup=context.matchup,
                tipoff_local_iso=context.tipoff_local,
                tipoff_local_label=context.tipoff_local_label,
                recommended_outcome_team_id=representative.outcome_team_id,
                recommended_outcome_team_name=representative.outcome_team_name,
                recommended_outcome_label=_build_outcome_label(
                    matchup=context.matchup,
                    outcome_team_name=representative.outcome_team_name,
                    game_label=context.game_label,
                ),
                outcome_probability=representative.outcome_probability,
                total_pool_swing=representative.outcome_total_swing,
                top_gainer_entry_id=top_gainer.entry_id,
                top_gainer_entry_name=_entry_name(entry_options_by_id, top_gainer.entry_id),
                top_gainer_win_percentage_point_delta=top_gainer.win_percentage_point_delta,
                top_loser_entry_id=top_loser.entry_id,
                top_loser_entry_name=_entry_name(entry_options_by_id, top_loser.entry_id),
                top_loser_win_percentage_point_delta=top_loser.win_percentage_point_delta,
            )
        )

    sorted_items = sorted(
        unsorted_items,
        key=lambda row: (
            -row.total_pool_swing,
            row.tipoff_local_iso,
            row.game_id,
        ),
    )
    return [
        row.model_copy(update={"rank": rank})
        for rank, row in enumerate(sorted_items, start=1)
    ]


def _build_entry_guides(
    *,
    entry_options: list[ViewingGuideEntryOption],
    tonight_rows: list[_GameOutcomeRow],
    tonight_games_by_id: dict[str, _TonightGameContext],
) -> tuple[dict[str, list[EntryViewingGuideRow]], list[EntryTopGameSummaryRow]]:
    rows_by_entry_id: dict[str, list[_GameOutcomeRow]] = defaultdict(list)
    for row in tonight_rows:
        rows_by_entry_id[row.entry_id].append(row)

    guides_by_entry_id: dict[str, list[EntryViewingGuideRow]] = {}
    top_games_by_entry: list[EntryTopGameSummaryRow] = []

    for entry in entry_options:
        rows_by_game_id: dict[str, list[_GameOutcomeRow]] = defaultdict(list)
        for row in rows_by_entry_id.get(entry.entry_id, []):
            rows_by_game_id[row.game_id].append(row)

        guide_rows = [
            _build_entry_guide_row(
                context=tonight_games_by_id[game_id],
                row=max(
                    grouped_rows,
                    key=lambda candidate: (
                        candidate.win_percentage_point_delta,
                        candidate.outcome_probability,
                        candidate.outcome_team_name,
                        candidate.outcome_team_id,
                    ),
                ),
            )
            for game_id, grouped_rows in rows_by_game_id.items()
            if game_id in tonight_games_by_id
        ]
        guide_rows.sort(
            key=lambda row: (
                -row.win_percentage_point_delta,
                row.tipoff_local_iso,
                row.game_id,
            ),
        )
        guides_by_entry_id[entry.entry_id] = guide_rows
        top_games_by_entry.append(_build_top_game_summary(entry=entry, guide_rows=guide_rows))

    return guides_by_entry_id, top_games_by_entry


def _build_entry_guide_row(
    *,
    context: _TonightGameContext,
    row: _GameOutcomeRow,
) -> EntryViewingGuideRow:
    return EntryViewingGuideRow(
        game_id=row.game_id,
        round=context.round,
        game_label=context.game_label,
        matchup=context.matchup,
        tipoff_local_iso=context.tipoff_local,
        tipoff_local_label=context.tipoff_local_label,
        recommended_outcome_team_id=row.outcome_team_id,
        recommended_outcome_team_name=row.outcome_team_name,
        recommended_outcome_label=_build_outcome_label(
            matchup=context.matchup,
            outcome_team_name=row.outcome_team_name,
            game_label=context.game_label,
        ),
        outcome_probability=row.outcome_probability,
        baseline_win_percentage=row.baseline_win_percentage,
        conditional_win_percentage=row.conditional_win_percentage,
        win_percentage_point_delta=row.win_percentage_point_delta,
    )


def _build_top_game_summary(
    *,
    entry: ViewingGuideEntryOption,
    guide_rows: list[EntryViewingGuideRow],
) -> EntryTopGameSummaryRow:
    if not guide_rows:
        return EntryTopGameSummaryRow(
            entry_rank=entry.rank,
            entry_id=entry.entry_id,
            entry_name=entry.entry_name,
            win_percentage=entry.win_percentage,
        )

    top_row = guide_rows[0]
    return EntryTopGameSummaryRow(
        entry_rank=entry.rank,
        entry_id=entry.entry_id,
        entry_name=entry.entry_name,
        win_percentage=entry.win_percentage,
        game_id=top_row.game_id,
        game_label=top_row.game_label,
        matchup=top_row.matchup,
        tipoff_local_iso=top_row.tipoff_local_iso,
        tipoff_local_label=top_row.tipoff_local_label,
        recommended_outcome_team_id=top_row.recommended_outcome_team_id,
        recommended_outcome_team_name=top_row.recommended_outcome_team_name,
        recommended_outcome_label=top_row.recommended_outcome_label,
        outcome_probability=top_row.outcome_probability,
        baseline_win_percentage=top_row.baseline_win_percentage,
        conditional_win_percentage=top_row.conditional_win_percentage,
        win_percentage_point_delta=top_row.win_percentage_point_delta,
    )


def _build_matchup_label(
    *,
    game: Game,
    games_by_id: dict[str, Game],
    constraints_by_game_id: dict[str, str],
    team_names_by_id: dict[str, str],
) -> str:
    left_name = _resolve_team_name(
        team_id=game.left_team_id,
        upstream_game_id=game.left_game_id,
        games_by_id=games_by_id,
        constraints_by_game_id=constraints_by_game_id,
        team_names_by_id=team_names_by_id,
    )
    right_name = _resolve_team_name(
        team_id=game.right_team_id,
        upstream_game_id=game.right_game_id,
        games_by_id=games_by_id,
        constraints_by_game_id=constraints_by_game_id,
        team_names_by_id=team_names_by_id,
    )

    if left_name is None and right_name is None:
        return _game_label(game)

    return f"{left_name or 'TBD'} vs {right_name or 'TBD'}"


def _resolve_team_name(
    *,
    team_id: str | None,
    upstream_game_id: str | None,
    games_by_id: dict[str, Game],
    constraints_by_game_id: dict[str, str],
    team_names_by_id: dict[str, str],
) -> str | None:
    resolved_team_id = team_id
    if resolved_team_id is None and upstream_game_id is not None:
        resolved_team_id = constraints_by_game_id.get(upstream_game_id)
    if resolved_team_id is None:
        return None
    if resolved_team_id in team_names_by_id:
        return team_names_by_id[resolved_team_id]
    if upstream_game_id is not None and upstream_game_id in games_by_id:
        return _game_label(games_by_id[upstream_game_id])
    return None


def _build_outcome_label(*, matchup: str, outcome_team_name: str, game_label: str) -> str:
    if " vs " not in matchup or "TBD" in matchup:
        return f"{outcome_team_name} in {game_label}"

    left_name, right_name = matchup.split(" vs ", maxsplit=1)
    if outcome_team_name == left_name:
        return f"{left_name} over {right_name}"
    if outcome_team_name == right_name:
        return f"{right_name} over {left_name}"
    return f"{outcome_team_name} in {game_label}"


def _entry_name(
    entry_options_by_id: dict[str, ViewingGuideEntryOption],
    entry_id: str,
) -> str:
    entry = entry_options_by_id.get(entry_id)
    return entry.entry_name if entry is not None else entry_id


def _game_label(game: Game) -> str:
    return f"Round {game.round} Game {game.game_id}"


def _format_tipoff(value: datetime) -> str:
    return value.strftime("%I:%M %p").lstrip("0")


def _timezone_name(*, timezone: tzinfo, reference: datetime) -> str:
    key = getattr(timezone, "key", None)
    if isinstance(key, str) and key:
        return key
    return reference.tzname() or "local"
