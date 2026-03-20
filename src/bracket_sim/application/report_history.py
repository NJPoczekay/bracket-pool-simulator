"""Historical win-percentage plot helpers for tracker report bundles."""

from __future__ import annotations

import hashlib
import json
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


@dataclass(frozen=True)
class HistoryPoint:
    """One historical plot point after a completed game."""

    game_id: str
    label: str
    left_logo_url: str | None
    right_logo_url: str | None
    entry_win_shares: dict[str, float]


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
    return _render_history_plot(
        history_points=history_points,
        entry_rows=entry_rows,
        report_name=config.report_name,
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
        history_points.append(
            HistoryPoint(
                game_id=completed_constraint.game_id,
                label=_matchup_label(
                    game_id=completed_constraint.game_id,
                    left_team=left_team,
                    right_team=right_team,
                ),
                left_logo_url=left_team.logo_url if left_team is not None else None,
                right_logo_url=right_team.logo_url if right_team is not None else None,
                entry_win_shares=entry_win_shares,
            )
        )

    if cache_path is not None:
        cached_points[_history_prefix_key(prefix_constraints)] = dict(full_result_by_entry_id)
        if cache_dirty or not cache_path.exists():
            _write_history_cache(path=cache_path, base_key=base_key, points=cached_points)

    return history_points


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


def _render_history_plot(
    *,
    history_points: list[HistoryPoint],
    entry_rows: list[EntryReportRow],
    report_name: str,
) -> bytes:
    point_count = max(len(history_points), 1)
    figure_width = max(12.0, min(28.0, 4.0 + (point_count * 0.55)))
    figure, axis = plt.subplots(figsize=(figure_width, 8.0))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#fbfbfd")
    axis.grid(True, axis="y", color="#d9dde7", linewidth=0.8, alpha=0.8)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Win Percentage")
    axis.set_xlabel("Games")
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
    for entry_row in entry_rows:
        series = [
            history_point.entry_win_shares.get(entry_row.entry_id, 0.0) * 100
            for history_point in history_points
        ]
        axis.plot(
            x_positions,
            series,
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label=entry_row.entry_name,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [history_point.label for history_point in history_points],
        rotation=58,
        ha="right",
        fontsize=9,
    )
    axis.set_xlim(-0.5, len(history_points) - 0.5)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
        title="Entries",
        title_fontsize=10,
    )

    any_logos = False
    for x_position, history_point in zip(x_positions, history_points, strict=True):
        left_logo = _load_logo_image(history_point.left_logo_url)
        right_logo = _load_logo_image(history_point.right_logo_url)
        if left_logo is None or right_logo is None:
            continue
        any_logos = True
        _add_logo(axis=axis, x_position=x_position, image=left_logo, x_offset=-12)
        _add_logo(axis=axis, x_position=x_position, image=right_logo, x_offset=12)

    figure.subplots_adjust(bottom=0.34 if any_logos else 0.28, right=0.8)
    return _save_figure_bytes(figure)


def _add_logo(
    *,
    axis: Axes,
    x_position: float,
    image: _LogoImage,
    x_offset: int,
) -> None:
    annotation = AnnotationBbox(
        OffsetImage(image, zoom=0.12),
        (x_position, 0.0),
        xycoords=("data", "axes fraction"),
        xybox=(x_offset, -44),
        boxcoords="offset points",
        frameon=False,
    )
    axis.add_artist(annotation)


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
