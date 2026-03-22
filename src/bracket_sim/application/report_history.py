"""Historical win-percentage plot helpers for tracker report bundles."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib
import numpy as np
import numpy.typing as npt

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.transforms import blended_transform_factory

from bracket_sim.domain.bracket_graph import BracketGraph
from bracket_sim.domain.constraints import validate_constraints
from bracket_sim.domain.models import (
    CompletedGameConstraint,
    EntryReportRow,
    Game,
    PoolEntry,
    RatingRecord,
    ReportConfig,
    Team,
)
from bracket_sim.domain.scoring import aggregate_win_share_totals, score_entries
from bracket_sim.domain.scoring_systems import ScoringSpec, resolve_scoring_spec
from bracket_sim.domain.simulator import simulate_tournament
from bracket_sim.infrastructure.storage.cache_keys import capture_dataset_file_hashes
from bracket_sim.infrastructure.storage.report_bundle import slugify_report_name
from bracket_sim.infrastructure.storage.run_artifacts import write_json_atomic

_HISTORY_CACHE_SCHEMA_VERSION = 1
_STATIC_HISTORY_INPUTS = frozenset({"teams.json", "games.json", "entries.json", "ratings.csv"})
_LOGO_TIMEOUT_SECONDS = 3.0
_LogoImage = npt.NDArray[np.float32] | npt.NDArray[np.float64]
_SERIES_ALPHA_CYCLE = (0.95, 0.72)


@dataclass(frozen=True)
class HistoryPoint:
    """One historical plot point after a completed game."""

    game_id: str
    round: int
    label: str
    top_logo_url: str | None
    bottom_logo_url: str | None
    entry_win_shares: dict[str, float]


@dataclass(frozen=True)
class RoundCompletionMarker:
    """Vertical annotation for the last completed game in a finished round."""

    x_position: float
    round: int
    label: str


def build_win_percentage_history_plot(
    *,
    teams: list[Team],
    games: list[Game],
    entries: list[PoolEntry],
    constraints: list[CompletedGameConstraint],
    predicted_wins: npt.NDArray[np.int16],
    team_seeds: npt.NDArray[np.int16],
    team_ids: list[str],
    rating_records_by_team_id: dict[str, RatingRecord],
    graph: BracketGraph,
    config: ReportConfig,
    entry_rows: list[EntryReportRow],
) -> bytes:
    """Build the cached win-percentage history plot for one tracker report."""

    history_points = _build_history_points(
        teams=teams,
        games=games,
        entries=entries,
        constraints=constraints,
        predicted_wins=predicted_wins,
        team_seeds=team_seeds,
        team_ids=team_ids,
        rating_records_by_team_id=rating_records_by_team_id,
        graph=graph,
        config=config,
        entry_rows=entry_rows,
    )
    round_completion_markers = _build_round_completion_markers(
        history_points=history_points,
        games=games,
    )
    return _render_history_plot(
        history_points=history_points,
        entry_rows=entry_rows,
        report_name=config.report_name,
        round_completion_markers=round_completion_markers,
    )


def _build_history_points(
    *,
    teams: list[Team],
    games: list[Game],
    entries: list[PoolEntry],
    constraints: list[CompletedGameConstraint],
    predicted_wins: npt.NDArray[np.int16],
    team_seeds: npt.NDArray[np.int16],
    team_ids: list[str],
    rating_records_by_team_id: dict[str, RatingRecord],
    graph: BracketGraph,
    config: ReportConfig,
    entry_rows: list[EntryReportRow],
) -> list[HistoryPoint]:
    scoring_spec = resolve_scoring_spec(config.scoring_system)
    teams_by_id = {team.team_id: team for team in teams}
    games_by_id = {game.game_id: game for game in games}
    completed_constraints = _ordered_completed_constraints(
        constraints=constraints,
        games_by_id=games_by_id,
    )
    if not completed_constraints:
        return []

    cache_path = _history_cache_path(
        history_cache_dir=config.history_cache_dir,
        report_name=config.report_name,
    )
    base_key = _history_cache_base_key(config=config)
    cached_points = _load_history_cache(path=cache_path, base_key=base_key)
    cache_dirty = False

    full_result_by_entry_id = {row.entry_id: row.win_share for row in entry_rows}
    constraints_by_game_id = {row.game_id: row.winner_team_id for row in constraints}
    history_points: list[HistoryPoint] = []
    prefix_constraints: list[CompletedGameConstraint] = []

    for index, completed_constraint in enumerate(completed_constraints):
        prefix_constraints.append(completed_constraint)
        prefix_key = _history_prefix_key(prefix_constraints)
        entry_win_shares: dict[str, float] | None

        if index == len(completed_constraints) - 1:
            entry_win_shares = dict(full_result_by_entry_id)
        else:
            entry_win_shares = cached_points.get(prefix_key)
            if entry_win_shares is None:
                entry_win_shares = _simulate_entry_win_shares_for_prefix(
                    entries=entries,
                    prefix_constraints=prefix_constraints,
                    predicted_wins=predicted_wins,
                    team_seeds=team_seeds,
                    team_ids=team_ids,
                    rating_records_by_team_id=rating_records_by_team_id,
                    graph=graph,
                    config=config,
                    scoring_spec=scoring_spec,
                )
                cached_points[prefix_key] = entry_win_shares
                cache_dirty = True

        assert entry_win_shares is not None

        left_team_id, right_team_id = _resolve_completed_matchup_team_ids(
            game_id=completed_constraint.game_id,
            games_by_id=games_by_id,
            constraints_by_game_id=constraints_by_game_id,
        )
        left_team = teams_by_id.get(left_team_id) if left_team_id is not None else None
        right_team = teams_by_id.get(right_team_id) if right_team_id is not None else None
        top_logo_url, bottom_logo_url = _winner_first_logo_url(
            winner_team_id=completed_constraint.winner_team_id,
            left_team=left_team,
            right_team=right_team,
        )
        history_points.append(
            HistoryPoint(
                game_id=completed_constraint.game_id,
                round=games_by_id[completed_constraint.game_id].round,
                label=_matchup_label(
                    game_id=completed_constraint.game_id,
                    left_team=left_team,
                    right_team=right_team,
                ),
                top_logo_url=top_logo_url,
                bottom_logo_url=bottom_logo_url,
                entry_win_shares=entry_win_shares,
            )
        )

    if cache_path is not None:
        cached_points[_history_prefix_key(prefix_constraints)] = dict(full_result_by_entry_id)
        if cache_dirty or not cache_path.exists():
            _write_history_cache(path=cache_path, base_key=base_key, points=cached_points)

    return history_points


def _build_round_completion_markers(
    *,
    history_points: list[HistoryPoint],
    games: list[Game],
) -> list[RoundCompletionMarker]:
    if not history_points:
        return []

    total_games_by_round = Counter(game.round for game in games)
    completed_games_by_round: defaultdict[int, int] = defaultdict(int)
    markers: list[RoundCompletionMarker] = []

    for index, history_point in enumerate(history_points):
        completed_games_by_round[history_point.round] += 1
        if (
            completed_games_by_round[history_point.round]
            != total_games_by_round[history_point.round]
        ):
            continue
        markers.append(
            RoundCompletionMarker(
                x_position=float(index),
                round=history_point.round,
                label=_history_round_label(history_point.round),
            )
        )

    return markers


def _history_round_label(round_number: int) -> str:
    labels = {
        1: "Round of 64",
        2: "Round of 32",
        3: "Sweet 16",
        4: "Elite 8",
        5: "Final Four",
        6: "Championship",
    }
    return labels.get(round_number, f"Round {round_number}")


def _ordered_completed_constraints(
    *,
    constraints: list[CompletedGameConstraint],
    games_by_id: dict[str, Game],
) -> list[CompletedGameConstraint]:
    return sorted(
        constraints,
        key=lambda row: _constraint_sort_key(constraint=row, games_by_id=games_by_id),
    )


def _constraint_sort_key(
    *,
    constraint: CompletedGameConstraint,
    games_by_id: dict[str, Game],
) -> tuple[float, float, int, str]:
    game = games_by_id[constraint.game_id]
    completed = (
        game.completed_at_utc.timestamp()
        if game.completed_at_utc is not None
        else float("inf")
    )
    scheduled = (
        game.scheduled_at_utc.timestamp()
        if game.scheduled_at_utc is not None
        else float("inf")
    )
    return (completed, scheduled, game.display_order or 10_000, game.game_id)


def _simulate_entry_win_shares_for_prefix(
    *,
    entries: list[PoolEntry],
    prefix_constraints: list[CompletedGameConstraint],
    predicted_wins: npt.NDArray[np.int16],
    team_seeds: npt.NDArray[np.int16],
    team_ids: list[str],
    rating_records_by_team_id: dict[str, RatingRecord],
    graph: BracketGraph,
    config: ReportConfig,
    scoring_spec: ScoringSpec,
) -> dict[str, float]:
    constraints_by_game_id = validate_constraints(constraints=prefix_constraints, graph=graph)
    win_share_totals = np.zeros(len(entries), dtype=np.float64)
    total_batches = (config.n_sims + config.effective_batch_size - 1) // config.effective_batch_size

    for batch_index in range(total_batches):
        batch_n_sims = min(
            config.effective_batch_size,
            config.n_sims - (batch_index * config.effective_batch_size),
        )
        batch_seed = _derive_batch_seed(
            seed=config.seed,
            batch_index=batch_index,
            total_batches=total_batches,
        )
        simulation = simulate_tournament(
            graph=graph,
            rating_records_by_team_id=rating_records_by_team_id,
            constraints_by_game_id=constraints_by_game_id,
            n_sims=batch_n_sims,
            seed=batch_seed,
            point_spread_std_dev=config.rating_scale,
            engine=config.engine,
        )
        scores = score_entries(
            predicted_wins=predicted_wins,
            actual_wins=simulation.team_wins,
            round_values=scoring_spec.round_values,
            team_seeds=team_seeds,
            seed_bonus_rounds=scoring_spec.seed_bonus_rounds,
        )
        win_share_totals += aggregate_win_share_totals(scores)

    return {
        entry.entry_id: float(win_share_totals[index] / config.n_sims)
        for index, entry in enumerate(entries)
    }


def _history_cache_path(*, history_cache_dir: Path | None, report_name: str) -> Path | None:
    if history_cache_dir is None:
        return None
    filename = f"{slugify_report_name(report_name)}_win_percentage_history_cache.json"
    return history_cache_dir / filename


def _history_cache_base_key(*, config: ReportConfig) -> str:
    file_hashes = capture_dataset_file_hashes(config.input_dir)
    payload = {
        "input_hashes": {
            name: digest
            for name, digest in file_hashes.items()
            if name in _STATIC_HISTORY_INPUTS
        },
        "n_sims": config.n_sims,
        "seed": config.seed,
        "rating_scale": config.rating_scale,
        "batch_size": config.effective_batch_size,
        "engine": config.engine,
        "scoring_system": config.scoring_system.value,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _history_prefix_key(prefix_constraints: list[CompletedGameConstraint]) -> str:
    payload = [constraint.model_dump(mode="json") for constraint in prefix_constraints]
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_history_cache(path: Path | None, *, base_key: str) -> dict[str, dict[str, float]]:
    if path is None or not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}
    if payload.get("schema_version") != _HISTORY_CACHE_SCHEMA_VERSION:
        return {}
    if payload.get("base_key") != base_key:
        return {}

    points = payload.get("points")
    if not isinstance(points, dict):
        return {}

    normalized: dict[str, dict[str, float]] = {}
    for prefix_key, point_payload in points.items():
        if not isinstance(prefix_key, str) or not isinstance(point_payload, dict):
            continue
        entry_win_shares = point_payload.get("entry_win_shares")
        if not isinstance(entry_win_shares, dict):
            continue
        normalized[prefix_key] = {
            str(entry_id): float(win_share)
            for entry_id, win_share in entry_win_shares.items()
        }
    return normalized


def _write_history_cache(
    *,
    path: Path,
    base_key: str,
    points: dict[str, dict[str, float]],
) -> None:
    payload = {
        "schema_version": _HISTORY_CACHE_SCHEMA_VERSION,
        "base_key": base_key,
        "points": {
            prefix_key: {
                "entry_win_shares": entry_win_shares,
            }
            for prefix_key, entry_win_shares in sorted(points.items())
        },
    }
    write_json_atomic(path=path, payload=payload)


def _resolve_completed_matchup_team_ids(
    *,
    game_id: str,
    games_by_id: dict[str, Game],
    constraints_by_game_id: dict[str, str],
) -> tuple[str | None, str | None]:
    game = games_by_id[game_id]
    if game.round == 1:
        return game.left_team_id, game.right_team_id

    left_team_id = (
        constraints_by_game_id.get(game.left_game_id)
        if game.left_game_id is not None
        else None
    )
    right_team_id = (
        constraints_by_game_id.get(game.right_game_id)
        if game.right_game_id is not None
        else None
    )
    return left_team_id, right_team_id


def _matchup_label(*, game_id: str, left_team: Team | None, right_team: Team | None) -> str:
    if left_team is None or right_team is None:
        return game_id
    return f"{left_team.abbrev or left_team.name} vs {right_team.abbrev or right_team.name}"


def _winner_first_logo_url(
    *,
    winner_team_id: str,
    left_team: Team | None,
    right_team: Team | None,
) -> tuple[str | None, str | None]:
    if left_team is None or right_team is None:
        return (None, None)
    if winner_team_id == left_team.team_id:
        return (left_team.logo_url, right_team.logo_url)
    if winner_team_id == right_team.team_id:
        return (right_team.logo_url, left_team.logo_url)
    return (left_team.logo_url, right_team.logo_url)


def _render_history_plot(
    *,
    history_points: list[HistoryPoint],
    entry_rows: list[EntryReportRow],
    report_name: str,
    round_completion_markers: list[RoundCompletionMarker],
) -> bytes:
    point_count = max(len(history_points), 1)
    figure_width = max(12.0, min(28.0, 4.0 + (point_count * 0.55)))
    figure, axis = plt.subplots(figsize=(figure_width, 8.0))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.grid(True, axis="y", color="#d7dbe2", linewidth=0.8, alpha=0.9)
    axis.grid(True, axis="x", color="#e4e8ef", linewidth=0.8, alpha=0.85)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8d96a6")
    axis.spines["bottom"].set_color("#8d96a6")
    axis.tick_params(axis="y", colors="#3f4652")
    axis.set_ylabel("Win Percentage")
    axis.set_title(f"{report_name} Win Percentage by Completed Game")

    if not history_points:
        axis.set_xticks([])
        axis.text(
            0.5,
            0.5,
            "No completed games yet",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="#44506b",
        )
        figure.subplots_adjust(bottom=0.18, right=0.82)
        return _save_figure_bytes(figure)

    x_positions = np.arange(len(history_points), dtype=np.float64)
    series_values: list[list[float]] = []
    series_styles = _build_series_styles(len(entry_rows))
    line_labels: list[tuple[float, str, tuple[float, float, float, float], float]] = []
    for index, entry_row in enumerate(entry_rows):
        series = [
            history_point.entry_win_shares.get(entry_row.entry_id, 0.0) * 100
            for history_point in history_points
        ]
        series_values.append(series)
        latest_win_percentage = series[-1] if series else 0.0
        style = series_styles[index]
        axis.plot(
            x_positions,
            series,
            linewidth=2.2,
            color=style["color"],
            alpha=style["alpha"],
        )
        line_labels.append(
            (
                series[-1] if series else 0.0,
                f"{entry_row.entry_name} {latest_win_percentage:.2f}%",
                style["color"],
                float(style["alpha"]),
            )
        )

    y_min, y_max = _history_y_limits(series_values)
    axis.set_ylim(y_min, y_max)
    axis.set_xticks(x_positions)
    axis.set_xticklabels([])
    axis.set_xlim(-0.5, len(history_points) - 0.5)
    axis.tick_params(axis="x", length=0)
    _add_round_completion_markers(
        axis=axis,
        round_completion_markers=round_completion_markers,
    )
    _add_end_labels(
        axis=axis,
        x_position=x_positions[-1] if len(x_positions) > 0 else 0.0,
        labels=line_labels,
        y_min=y_min,
        y_max=y_max,
    )

    any_logos = False
    for x_position, history_point in zip(x_positions, history_points, strict=True):
        top_logo = _load_logo_image(history_point.top_logo_url)
        bottom_logo = _load_logo_image(history_point.bottom_logo_url)
        if top_logo is None or bottom_logo is None:
            continue
        any_logos = True
        _add_logo(axis=axis, x_position=x_position, image=top_logo, y_offset=-14)
        _add_matchup_separator(axis=axis, x_position=x_position)
        _add_logo(axis=axis, x_position=x_position, image=bottom_logo, y_offset=-54)

    figure.subplots_adjust(bottom=0.3 if any_logos else 0.18, right=0.76)
    return _save_figure_bytes(figure)


def _add_round_completion_markers(
    *,
    axis: Axes,
    round_completion_markers: list[RoundCompletionMarker],
) -> None:
    if not round_completion_markers:
        return

    text_transform = blended_transform_factory(axis.transData, axis.transAxes)
    for marker in round_completion_markers:
        axis.axvline(
            marker.x_position,
            color="#7b8798",
            linewidth=1.0,
            linestyle=(0, (4, 4)),
            alpha=0.85,
            zorder=0.5,
        )
        axis.text(
            marker.x_position,
            0.985,
            marker.label,
            transform=text_transform,
            rotation=90,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#4e596a",
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
        )


def _build_series_styles(
    series_count: int,
) -> list[dict[str, float | tuple[float, float, float, float]]]:
    if series_count <= 0:
        return []

    cmap = plt.get_cmap("tab20c")
    color_order = _tab20c_color_order(cmap.N)
    styles: list[dict[str, float | tuple[float, float, float, float]]] = []
    for index in range(series_count):
        styles.append(
            {
                "color": cmap(color_order[index % len(color_order)]),
                "alpha": _SERIES_ALPHA_CYCLE[
                    (index // len(color_order)) % len(_SERIES_ALPHA_CYCLE)
                ],
            }
        )
    return styles


def _tab20c_color_order(color_count: int) -> list[int]:
    if color_count <= 0:
        return []

    order: list[int] = []
    for start in range(4):
        for index in range(start, color_count, 4):
            order.append(index)
    return order


def _add_end_labels(
    *,
    axis: Axes,
    x_position: float,
    labels: list[tuple[float, str, tuple[float, float, float, float], float]],
    y_min: float,
    y_max: float,
) -> None:
    if not labels:
        return

    label_count = len(labels)
    span = max(y_max - y_min, 1.0)
    sorted_labels = sorted(labels, key=lambda row: row[0])
    base_gap = min(max(span * 0.028, 0.55), 0.95)
    packing_gap = max(span / max(label_count + 5, 6), 0.35)
    minimum_gap = min(base_gap, packing_gap)
    adjusted_y_positions: list[float] = []

    for original_y, _, _, _ in sorted_labels:
        if not adjusted_y_positions:
            adjusted_y_positions.append(max(y_min, min(y_max, original_y)))
            continue
        adjusted_y_positions.append(max(original_y, adjusted_y_positions[-1] + minimum_gap))

    overflow = adjusted_y_positions[-1] - y_max
    if overflow > 0:
        adjusted_y_positions = [value - overflow for value in adjusted_y_positions]

    if adjusted_y_positions[0] < y_min:
        underflow = y_min - adjusted_y_positions[0]
        adjusted_y_positions = [value + underflow for value in adjusted_y_positions]

    original_center = sum(label[0] for label in sorted_labels) / label_count
    adjusted_center = sum(adjusted_y_positions) / label_count
    recenter_shift = original_center - adjusted_center
    max_up_shift = y_max - adjusted_y_positions[-1]
    max_down_shift = y_min - adjusted_y_positions[0]
    recenter_shift = min(max(recenter_shift, max_down_shift), max_up_shift)
    if abs(recenter_shift) > 1e-9:
        adjusted_y_positions = [value + recenter_shift for value in adjusted_y_positions]

    for adjusted_y, (original_y, label, color, alpha) in zip(
        adjusted_y_positions,
        sorted_labels,
        strict=True,
    ):
        if abs(adjusted_y - original_y) > 0.15:
            axis.plot(
                [x_position, x_position + 0.18],
                [original_y, adjusted_y],
                color=color,
                alpha=max(alpha * 0.75, 0.45),
                linewidth=1.0,
                solid_capstyle="round",
                clip_on=False,
            )
        axis.text(
            x_position + 0.22,
            adjusted_y,
            label,
            color=color,
            alpha=alpha,
            fontsize=9,
            va="center",
            ha="left",
            clip_on=False,
        )


def _history_y_limits(series_values: list[list[float]]) -> tuple[float, float]:
    if not series_values:
        return (0.0, 100.0)

    flattened = [value for series in series_values for value in series]
    if not flattened:
        return (0.0, 100.0)

    observed_min = min(flattened)
    observed_max = max(flattened)
    observed_span = observed_max - observed_min
    target_span = max(observed_span + 6.0, 12.0)
    midpoint = (observed_min + observed_max) / 2.0
    lower = max(0.0, midpoint - (target_span / 2.0))
    upper = min(100.0, midpoint + (target_span / 2.0))

    if upper - lower < 12.0:
        shortfall = 12.0 - (upper - lower)
        lower = max(0.0, lower - (shortfall / 2.0))
        upper = min(100.0, upper + (shortfall / 2.0))

    if lower <= 0.0:
        upper = min(100.0, max(upper, 12.0))
    if upper >= 100.0:
        lower = max(0.0, min(lower, 88.0))

    return (lower, upper)


def _add_logo(
    *,
    axis: Axes,
    x_position: float,
    image: _LogoImage,
    y_offset: int,
) -> None:
    annotation = AnnotationBbox(
        OffsetImage(image, zoom=0.055),
        (x_position, 0.0),
        xycoords=("data", "axes fraction"),
        xybox=(0, y_offset),
        boxcoords="offset points",
        frameon=False,
    )
    axis.add_artist(annotation)


def _add_matchup_separator(*, axis: Axes, x_position: float) -> None:
    axis.annotate(
        "o.",
        (x_position, 0.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, -34),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="semibold",
        color="#4e596a",
        annotation_clip=False,
    )


def _load_logo_image(logo_url: str | None) -> _LogoImage | None:
    if logo_url is None or logo_url == "":
        return None

    try:
        with urlopen(logo_url, timeout=_LOGO_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except (OSError, URLError, ValueError):
        return None

    try:
        return plt.imread(BytesIO(payload), format="png")
    except Exception:
        return None


def _save_figure_bytes(figure: Figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def _derive_batch_seed(*, seed: int, batch_index: int, total_batches: int) -> int:
    if total_batches == 1 and batch_index == 0:
        return seed

    sequence = np.random.SeedSequence([seed, batch_index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])
