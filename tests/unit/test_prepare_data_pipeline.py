from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from bracket_sim.domain.models import Team
from bracket_sim.infrastructure.storage.alias_resolver import AliasResolver
from bracket_sim.infrastructure.storage.raw_loader import RawAlias, load_raw_input


def test_load_raw_input_requires_required_files(
    raw_canonical_dir: Path,
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw_missing_required"
    shutil.copytree(raw_canonical_dir, raw_dir)
    (raw_dir / "teams.csv").unlink()

    with pytest.raises(ValueError, match="Required raw file is missing: teams.csv"):
        load_raw_input(raw_dir)


def test_load_raw_input_allows_missing_optional_files(
    raw_canonical_dir: Path,
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw_without_optional"
    shutil.copytree(raw_canonical_dir, raw_dir)
    (raw_dir / "constraints.json").unlink()

    loaded = load_raw_input(raw_dir)
    assert loaded.constraints == []
    assert loaded.aliases == []


def test_alias_resolver_rejects_duplicate_aliases() -> None:
    teams = [
        Team(team_id="east-01", name="East Team 1", seed=1, region="east"),
        Team(team_id="east-02", name="East Team 2", seed=2, region="east"),
    ]
    aliases = [
        RawAlias(alias="Blue Devils", team_id="east-01"),
        RawAlias(alias="blue devils", team_id="east-01"),
    ]

    with pytest.raises(ValueError, match="Duplicate alias declared"):
        AliasResolver.build(teams=teams, aliases=aliases)


def test_alias_resolver_rejects_collisions() -> None:
    teams = [
        Team(team_id="east-01", name="East Team 1", seed=1, region="east"),
        Team(team_id="east-02", name="East Team 2", seed=2, region="east"),
    ]
    aliases = [RawAlias(alias="East Team 2", team_id="east-01")]

    with pytest.raises(ValueError, match="Alias collision"):
        AliasResolver.build(teams=teams, aliases=aliases)


def test_alias_resolver_resolves_casefolded_alias() -> None:
    teams = [Team(team_id="east-01", name="East Team 1", seed=1, region="east")]
    aliases = [RawAlias(alias="  BLUE DEVILS  ", team_id="east-01")]
    resolver = AliasResolver.build(teams=teams, aliases=aliases)

    resolved = resolver.resolve_team_id("blue devils", context="unit-test")
    assert resolved == "east-01"
