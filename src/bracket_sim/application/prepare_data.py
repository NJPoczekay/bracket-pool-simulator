"""Data preparation orchestration entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bracket_sim.domain.bracket_graph import build_bracket_graph
from bracket_sim.domain.constraints import validate_constraints
from bracket_sim.domain.models import (
    CompletedGameConstraint,
    EntryPick,
    PoolEntry,
    RatingRecord,
    Team,
)
from bracket_sim.domain.scoring import validate_entries
from bracket_sim.infrastructure.storage.alias_resolver import AliasResolver
from bracket_sim.infrastructure.storage.prepared_writer import (
    PreparedDataset,
    write_prepared_dataset,
)
from bracket_sim.infrastructure.storage.raw_loader import RawConstraint, RawInput, load_raw_input


@dataclass(frozen=True)
class PrepareDataSummary:
    """Summary information returned after successful dataset preparation."""

    output_dir: Path
    teams: int
    games: int
    entries: int
    constraints: int
    ratings: int
    aliases: int


def prepare_data(*, raw_dir: Path, out_dir: Path) -> PrepareDataSummary:
    """Normalize canonical raw data, validate it, and persist prepared artifacts."""

    if raw_dir.resolve() == out_dir.resolve():
        msg = "--raw and --out must point to different directories"
        raise ValueError(msg)

    raw = load_raw_input(raw_dir)
    teams = sorted(raw.teams, key=lambda team: team.team_id)
    games = sorted(raw.games, key=lambda game: (game.round, game.game_id))
    teams_by_id = {team.team_id: team for team in teams}

    resolver = AliasResolver.build(teams=teams, aliases=raw.aliases)
    entries = _normalize_entries(raw=raw, resolver=resolver)
    constraints = _normalize_constraints(constraints=raw.constraints, resolver=resolver)
    ratings = _normalize_ratings(
        ratings=raw.ratings,
        teams=teams,
        teams_by_id=teams_by_id,
        resolver=resolver,
    )

    graph = build_bracket_graph(teams=teams, games=games)
    validate_constraints(constraints=constraints, graph=graph)
    validate_entries(entries=entries, graph=graph)

    prepared = PreparedDataset(
        teams=teams,
        games=games,
        entries=entries,
        constraints=constraints,
        ratings=ratings,
    )
    write_prepared_dataset(raw_dir=raw_dir, out_dir=out_dir, dataset=prepared)

    return PrepareDataSummary(
        output_dir=out_dir,
        teams=len(teams),
        games=len(games),
        entries=len(entries),
        constraints=len(constraints),
        ratings=len(ratings),
        aliases=len(raw.aliases),
    )


def _normalize_entries(*, raw: RawInput, resolver: AliasResolver) -> list[PoolEntry]:
    entries: list[PoolEntry] = []
    for row in raw.entries:
        picks: list[EntryPick] = []
        for game_id, winner in sorted(row.picks.items()):
            winner_team_id = resolver.resolve_team_id(
                winner,
                context=f"entry '{row.entry_id}' pick for game '{game_id}'",
            )
            picks.append(EntryPick(game_id=game_id, winner_team_id=winner_team_id))

        entries.append(PoolEntry(entry_id=row.entry_id, entry_name=row.entry_name, picks=picks))

    entries.sort(key=lambda entry: entry.entry_id)
    return entries


def _normalize_constraints(
    *,
    constraints: list[RawConstraint],
    resolver: AliasResolver,
) -> list[CompletedGameConstraint]:
    normalized = [
        CompletedGameConstraint(
            game_id=constraint.game_id,
            winner_team_id=resolver.resolve_team_id(
                constraint.winner,
                context=f"constraint for game '{constraint.game_id}'",
            ),
        )
        for constraint in constraints
    ]
    normalized.sort(key=lambda constraint: constraint.game_id)
    return normalized


def _normalize_ratings(
    *,
    ratings: list[RatingRecord],
    teams: list[Team],
    teams_by_id: dict[str, Team],
    resolver: AliasResolver,
) -> list[RatingRecord]:
    ratings_by_team_id: dict[str, RatingRecord] = {}
    for idx, rating in enumerate(ratings, start=1):
        team_id = resolver.resolve_team_id(
            rating.team,
            context=f"ratings.csv row {idx}",
        )
        if team_id in ratings_by_team_id:
            team_name = teams_by_id[team_id].name
            msg = f"Duplicate rating for team '{team_name}'"
            raise ValueError(msg)
        ratings_by_team_id[team_id] = rating

    expected_team_ids = {team.team_id for team in teams}
    missing_team_ids = sorted(expected_team_ids - set(ratings_by_team_id))
    if missing_team_ids:
        msg = f"Missing ratings for teams: {missing_team_ids[:5]}"
        raise ValueError(msg)

    normalized = [
        RatingRecord(
            team=teams_by_id[team_id].name,
            rating=ratings_by_team_id[team_id].rating,
            tempo=ratings_by_team_id[team_id].tempo,
        )
        for team_id in sorted(expected_team_ids, key=lambda key: teams_by_id[key].name.casefold())
    ]
    return normalized
