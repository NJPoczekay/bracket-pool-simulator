"""Build per-entry pivotal outcome tables from report bundle artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from bracket_sim.domain.models import Game
from bracket_sim.infrastructure.storage._file_io import load_required_csv_rows
from bracket_sim.infrastructure.storage.normalized_loader import load_normalized_input


@dataclass(frozen=True)
class EntryPivotalOutcomeRow:
    """One entry's highest-swing unresolved outcome in a target round."""

    entry_rank: int
    entry_id: str
    entry_name: str
    game_id: str
    round: int
    game_label: str
    matchup: str
    outcome_team_id: str
    outcome_team_name: str
    outcome_label: str
    outcome_probability: float
    win_percentage_point_delta: float
    baseline_win_percentage: float
    conditional_win_percentage: float


@dataclass(frozen=True)
class EntryPivotalOutcomesResult:
    """Structured output for the entry-pivotal-outcomes CLI command."""

    input_dir: Path
    report_dir: Path
    round_number: int
    rows: list[EntryPivotalOutcomeRow]

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the result."""

        return {
            "input_dir": str(self.input_dir),
            "report_dir": str(self.report_dir),
            "round": self.round_number,
            "rows": [asdict(row) for row in self.rows],
        }


def generate_entry_pivotal_outcomes(
    *,
    input_dir: Path,
    report_dir: Path,
    round_number: int = 2,
) -> EntryPivotalOutcomesResult:
    """Return one highest-swing unresolved outcome per entry for a target round."""

    normalized = load_normalized_input(input_dir)
    report_rows, fieldnames = load_required_csv_rows(
        report_dir / "game_outcome_sensitivity.csv",
        missing_prefix="Required report artifact is missing",
    )
    _validate_report_fieldnames(fieldnames)

    teams_by_id = {team.team_id: team for team in normalized.teams}
    constraints_by_game_id = {
        constraint.game_id: constraint.winner_team_id for constraint in normalized.constraints
    }
    games_by_id = {game.game_id: game for game in normalized.games}

    unresolved_game_ids = {
        game.game_id
        for game in normalized.games
        if game.round == round_number and game.game_id not in constraints_by_game_id
    }

    best_by_entry_id: dict[str, EntryPivotalOutcomeRow] = {}
    for report_row in report_rows:
        game_id = report_row["game_id"]
        if game_id not in unresolved_game_ids:
            continue

        game = games_by_id.get(game_id)
        if game is None:
            msg = f"Report row references unknown game id: {game_id!r}"
            raise ValueError(msg)

        matchup = _build_matchup_label(
            game_id=game_id,
            games_by_id=games_by_id,
            constraints_by_game_id=constraints_by_game_id,
            team_names_by_id={team_id: team.name for team_id, team in teams_by_id.items()},
        )
        outcome_team_name = report_row["outcome_team_name"]
        row = EntryPivotalOutcomeRow(
            entry_rank=int(report_row["entry_rank"]),
            entry_id=report_row["entry_id"],
            entry_name=report_row["entry_name"],
            game_id=game_id,
            round=int(report_row["round"]),
            game_label=report_row["game_label"],
            matchup=matchup,
            outcome_team_id=report_row["outcome_team_id"],
            outcome_team_name=outcome_team_name,
            outcome_label=_build_outcome_label(
                matchup=matchup,
                outcome_team_name=outcome_team_name,
                game_label=report_row["game_label"],
            ),
            outcome_probability=float(report_row["outcome_probability"]),
            win_percentage_point_delta=float(report_row["win_percentage_point_delta"]),
            baseline_win_percentage=float(report_row["baseline_win_percentage"]),
            conditional_win_percentage=float(report_row["conditional_win_percentage"]),
        )

        previous = best_by_entry_id.get(row.entry_id)
        if previous is None or _sort_key(row) > _sort_key(previous):
            best_by_entry_id[row.entry_id] = row

    rows = sorted(
        best_by_entry_id.values(),
        key=lambda row: (row.entry_rank, row.entry_name, row.entry_id),
    )
    return EntryPivotalOutcomesResult(
        input_dir=input_dir,
        report_dir=report_dir,
        round_number=round_number,
        rows=rows,
    )


def _validate_report_fieldnames(fieldnames: list[str]) -> None:
    expected_columns = {
        "game_id",
        "round",
        "game_label",
        "outcome_team_id",
        "outcome_team_name",
        "outcome_probability",
        "entry_rank",
        "entry_id",
        "entry_name",
        "baseline_win_percentage",
        "conditional_win_percentage",
        "win_percentage_point_delta",
    }
    missing = sorted(expected_columns - set(fieldnames))
    if missing:
        msg = f"game_outcome_sensitivity.csv is missing required columns: {missing}"
        raise ValueError(msg)


def _build_matchup_label(
    *,
    game_id: str,
    games_by_id: dict[str, Game],
    constraints_by_game_id: dict[str, str],
    team_names_by_id: dict[str, str],
) -> str:
    game = games_by_id[game_id]

    left_team_id = game.left_team_id
    if left_team_id is None and game.left_game_id is not None:
        left_team_id = constraints_by_game_id.get(game.left_game_id)

    right_team_id = game.right_team_id
    if right_team_id is None and game.right_game_id is not None:
        right_team_id = constraints_by_game_id.get(game.right_game_id)

    if left_team_id is None or right_team_id is None:
        return f"Round {game.round} Game {game_id}"

    left_name = team_names_by_id.get(left_team_id)
    right_name = team_names_by_id.get(right_team_id)
    if left_name is None or right_name is None:
        return f"Round {game.round} Game {game_id}"

    return f"{left_name} vs {right_name}"


def _build_outcome_label(*, matchup: str, outcome_team_name: str, game_label: str) -> str:
    if " vs " not in matchup:
        return f"{outcome_team_name} in {game_label}"

    left_name, right_name = matchup.split(" vs ", maxsplit=1)
    if outcome_team_name == left_name:
        return f"{left_name} over {right_name}"
    if outcome_team_name == right_name:
        return f"{right_name} over {left_name}"
    return f"{outcome_team_name} in {game_label}"


def _sort_key(row: EntryPivotalOutcomeRow) -> tuple[float, float, int, str, str]:
    return (
        abs(row.win_percentage_point_delta),
        row.win_percentage_point_delta,
        -row.entry_rank,
        row.outcome_label,
        row.entry_id,
    )
