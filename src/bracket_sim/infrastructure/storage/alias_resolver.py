"""Alias normalization and team-id resolution helpers."""

from __future__ import annotations

import html
from dataclasses import dataclass

from bracket_sim.domain.models import Team
from bracket_sim.infrastructure.storage.raw_loader import RawAlias


@dataclass(frozen=True)
class AliasResolver:
    """Resolve canonical team ids from team ids, canonical names, and configured aliases."""

    _team_id_by_alias_key: dict[str, str]

    @classmethod
    def build(cls, teams: list[Team], aliases: list[RawAlias]) -> AliasResolver:
        team_ids = {team.team_id for team in teams}
        alias_map: dict[str, str] = {}

        for team in teams:
            _register_alias(
                alias_map=alias_map,
                alias=team.team_id,
                team_id=team.team_id,
                source=f"team id '{team.team_id}'",
                allow_same_target=True,
            )
            _register_alias(
                alias_map=alias_map,
                alias=team.name,
                team_id=team.team_id,
                source=f"team name '{team.name}'",
                allow_same_target=True,
            )

        seen_alias_file_keys: set[str] = set()
        for alias in aliases:
            alias_key = normalize_alias_key(alias.alias)
            if not alias_key:
                msg = "Alias rows must not contain blank alias values"
                raise ValueError(msg)
            if alias_key in seen_alias_file_keys:
                msg = f"Duplicate alias declared in aliases.csv: '{alias.alias}'"
                raise ValueError(msg)
            seen_alias_file_keys.add(alias_key)

            if alias.team_id not in team_ids:
                msg = f"Alias '{alias.alias}' references unknown team_id '{alias.team_id}'"
                raise ValueError(msg)

            _register_alias(
                alias_map=alias_map,
                alias=alias.alias,
                team_id=alias.team_id,
                source=f"alias '{alias.alias}'",
                allow_same_target=False,
            )

        return cls(_team_id_by_alias_key=alias_map)

    def resolve_team_id(self, raw_value: str, *, context: str) -> str:
        """Resolve team id from alias-like input and raise on unknown aliases."""

        alias_key = normalize_alias_key(raw_value)
        if alias_key == "":
            msg = f"Blank team reference in {context}"
            raise ValueError(msg)

        if alias_key not in self._team_id_by_alias_key:
            msg = f"Unknown alias '{raw_value}' in {context}"
            raise ValueError(msg)

        return self._team_id_by_alias_key[alias_key]


def normalize_alias_key(value: str) -> str:
    """Normalize aliases for consistent lookups."""

    return html.unescape(value).strip().casefold()


def _register_alias(
    *,
    alias_map: dict[str, str],
    alias: str,
    team_id: str,
    source: str,
    allow_same_target: bool,
) -> None:
    alias_key = normalize_alias_key(alias)
    if alias_key == "":
        msg = f"Blank alias encountered while registering {source}"
        raise ValueError(msg)

    if alias_key in alias_map:
        existing_team_id = alias_map[alias_key]
        if existing_team_id != team_id:
            msg = (
                f"Alias collision for '{alias}': maps to both '{existing_team_id}' "
                f"and '{team_id}'"
            )
            raise ValueError(msg)
        if not allow_same_target:
            msg = f"Alias '{alias}' duplicates an existing canonical mapping"
            raise ValueError(msg)
        return

    alias_map[alias_key] = team_id
