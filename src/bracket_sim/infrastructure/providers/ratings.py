"""Ratings providers for refresh-data (local default, optional KenPom)."""

from __future__ import annotations

import csv
import difflib
import html
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from bracket_sim.infrastructure.providers.contracts import (
    RatingsData,
    RatingsProvider,
    RawAliasRow,
    RawRatingRow,
    RawTeamRow,
)

_SPECIAL_ALIAS_EQUIVALENTS: dict[str, set[str]] = {
    "uconn": {"connecticut"},
    "connecticut": {"uconn"},
    "ole miss": {"mississippi"},
    "mississippi": {"ole miss"},
    "omaha": {"nebraska omaha"},
    "nebraska omaha": {"omaha"},
}


@dataclass(frozen=True)
class _InputRatingRow:
    team_raw: str
    rating: float
    tempo: float


class LocalRatingsProvider(RatingsProvider):
    """Load ratings from a local CSV or cached snapshot."""

    def __init__(
        self,
        *,
        ratings_file: Path | None = None,
        fallback_dir: Path | None = None,
    ) -> None:
        self._ratings_file = ratings_file
        self._fallback_dir = fallback_dir

    def fetch_ratings(self, *, teams: list[RawTeamRow]) -> RatingsData:
        ratings_path = self._resolve_ratings_path()
        input_rows = _load_ratings_csv(ratings_path)
        normalized_ratings, aliases = _normalize_ratings_rows(input_rows=input_rows, teams=teams)
        return RatingsData(
            ratings=normalized_ratings,
            aliases=aliases,
            source=f"local:{ratings_path}",
        )

    def _resolve_ratings_path(self) -> Path:
        if self._ratings_file is not None:
            if not self._ratings_file.exists():
                msg = f"Ratings file does not exist: {self._ratings_file}"
                raise ValueError(msg)
            return self._ratings_file

        if self._fallback_dir is not None:
            cached_path = self._fallback_dir / "ratings.csv"
            if cached_path.exists():
                return cached_path

        msg = (
            "No ratings source available. Provide --ratings-file or run refresh-data in a raw "
            "directory that already contains ratings.csv"
        )
        raise ValueError(msg)


class KenPomRatingsProvider(RatingsProvider):
    """Fetch ratings from KenPom with an authenticated session cookie."""

    def __init__(
        self,
        *,
        cookie_env_var: str = "KENPOM_COOKIE",
        url: str = "https://kenpom.com/",
        timeout_seconds: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._cookie_env_var = cookie_env_var
        self._url = url
        self._client = client or httpx.Client(timeout=timeout_seconds, follow_redirects=True)
        self._owns_client = client is None

    def close(self) -> None:
        """Close owned HTTP resources."""

        if self._owns_client:
            self._client.close()

    def fetch_ratings(self, *, teams: list[RawTeamRow]) -> RatingsData:
        cookie_value = os.getenv(self._cookie_env_var, "").strip()
        if cookie_value == "":
            msg = (
                f"{self._cookie_env_var} is required when KenPom ratings are enabled. "
                "Set it to your authenticated KenPom cookie string."
            )
            raise ValueError(msg)

        try:
            response = self._client.get(self._url, headers={"Cookie": cookie_value})
        except httpx.HTTPError as exc:
            msg = f"Failed KenPom request: {exc}"
            raise ValueError(msg) from exc

        if response.status_code in {401, 403}:
            msg = "KenPom authentication failed (401/403). Refresh KENPOM_COOKIE and retry."
            raise ValueError(msg)

        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            msg = f"KenPom request failed with status {response.status_code}: {exc}"
            raise ValueError(msg) from exc

        input_rows = _parse_kenpom_table(response.text)
        normalized_ratings, aliases = _normalize_ratings_rows(input_rows=input_rows, teams=teams)
        return RatingsData(ratings=normalized_ratings, aliases=aliases, source="kenpom")


def _load_ratings_csv(path: Path) -> list[_InputRatingRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        expected_variants = ({"team", "rating", "tempo"}, {"team_id", "rating", "tempo"})

        if set(fieldnames) not in expected_variants:
            msg = (
                f"{path.name} must have columns ['rating', 'team', 'tempo'] or "
                f"['rating', 'team_id', 'tempo'], got {fieldnames}"
            )
            raise ValueError(msg)

        team_key = "team" if "team" in fieldnames else "team_id"

        rows: list[_InputRatingRow] = []
        for idx, row in enumerate(reader, start=2):
            team_raw = _clean_text(row.get(team_key))
            rating_raw = _clean_text(row.get("rating")).replace("+", "")
            tempo_raw = _clean_text(row.get("tempo"))

            if team_raw == "":
                msg = f"{path.name} row {idx} has blank team value"
                raise ValueError(msg)

            try:
                rating = float(rating_raw)
                tempo = float(tempo_raw)
            except ValueError as exc:
                msg = f"{path.name} row {idx} has invalid numeric rating/tempo"
                raise ValueError(msg) from exc

            rows.append(_InputRatingRow(team_raw=team_raw, rating=rating, tempo=tempo))

    if not rows:
        msg = f"{path.name} must contain at least one rating row"
        raise ValueError(msg)

    return rows


def _parse_kenpom_table(payload: str) -> list[_InputRatingRow]:
    table_match = re.search(
        r"<table[^>]*id=[\"']ratings-table[\"'][^>]*>(.*?)</table>",
        payload,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if table_match is None:
        msg = "KenPom payload did not include a ratings table (id='ratings-table')"
        raise ValueError(msg)

    table_html = table_match.group(1)
    row_html_list = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.IGNORECASE | re.DOTALL)

    header_map: dict[str, int] = {}
    parsed_rows: list[_InputRatingRow] = []

    for row_html in row_html_list:
        header_cells = re.findall(r"<th[^>]*>(.*?)</th>", row_html, flags=re.IGNORECASE | re.DOTALL)
        if header_cells:
            normalized_headers = [_normalize_header(_strip_tags(cell)) for cell in header_cells]
            header_map = {header: idx for idx, header in enumerate(normalized_headers)}
            continue

        data_cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.IGNORECASE | re.DOTALL)
        if not data_cells:
            continue

        if not header_map:
            # Fall back to common KenPom ordering when headers are unavailable.
            header_map = {
                "team": 1,
                "adjem": 4,
                "adjt": 8,
            }

        required_headers = {"team", "adjem", "adjt"}
        if not required_headers.issubset(header_map):
            msg = "KenPom ratings table missing expected columns Team/AdjEM/AdjT"
            raise ValueError(msg)

        if max(header_map[header] for header in required_headers) >= len(data_cells):
            continue

        team_raw = _strip_tags(data_cells[header_map["team"]])
        rating_raw = _strip_tags(data_cells[header_map["adjem"]]).replace("+", "")
        tempo_raw = _strip_tags(data_cells[header_map["adjt"]])

        if team_raw == "":
            continue

        try:
            rating = float(rating_raw)
            tempo = float(tempo_raw)
        except ValueError:
            continue

        parsed_rows.append(_InputRatingRow(team_raw=team_raw, rating=rating, tempo=tempo))

    if not parsed_rows:
        msg = "KenPom ratings table was found but no rows could be parsed"
        raise ValueError(msg)

    return parsed_rows


def _normalize_ratings_rows(
    *,
    input_rows: list[_InputRatingRow],
    teams: list[RawTeamRow],
) -> tuple[list[RawRatingRow], list[RawAliasRow]]:
    alias_lookup = _build_alias_lookup(teams)
    team_name_by_id = {team.team_id: team.name for team in teams}
    unresolved_source_names: list[str] = []

    ratings_by_team_id: dict[str, RawRatingRow] = {}
    alias_rows: dict[str, RawAliasRow] = {}

    for row in input_rows:
        resolved_team_id = _resolve_team_id(row.team_raw, alias_lookup)
        if resolved_team_id is None:
            unresolved_source_names.append(row.team_raw)
            continue

        if resolved_team_id in ratings_by_team_id:
            msg = f"Duplicate rating row resolved to team '{resolved_team_id}'"
            raise ValueError(msg)

        ratings_by_team_id[resolved_team_id] = RawRatingRow(
            team=resolved_team_id,
            rating=row.rating,
            tempo=row.tempo,
        )

        normalized_team_raw = _normalize_team_key(row.team_raw)
        canonical_name = next(team.name for team in teams if team.team_id == resolved_team_id)
        if normalized_team_raw not in {
            _normalize_team_key(resolved_team_id),
            _normalize_team_key(canonical_name),
        }:
            alias_rows[normalized_team_raw] = RawAliasRow(
                alias=row.team_raw,
                team_id=resolved_team_id,
            )

    expected_team_ids = {team.team_id for team in teams}
    missing_team_ids = sorted(expected_team_ids - set(ratings_by_team_id))
    if missing_team_ids:
        missing_descriptions: list[str] = []
        for team_id in missing_team_ids[:8]:
            team_name = team_name_by_id[team_id]
            source_suggestions = difflib.get_close_matches(
                team_name,
                unresolved_source_names,
                n=3,
                cutoff=0.4,
            )
            if source_suggestions:
                missing_descriptions.append(
                    f"{team_name} (did you mean source rows: {', '.join(source_suggestions)})"
                )
            else:
                missing_descriptions.append(team_name)

        msg = (
            "Missing ratings for tournament teams after alias normalization: "
            f"{'; '.join(missing_descriptions)}"
        )
        raise ValueError(msg)

    ratings = [ratings_by_team_id[team_id] for team_id in sorted(expected_team_ids)]
    aliases = sorted(alias_rows.values(), key=lambda alias: _normalize_team_key(alias.alias))
    return ratings, aliases


def _build_alias_lookup(teams: list[RawTeamRow]) -> dict[str, str]:
    alias_lookup: dict[str, str] = {}

    for team in teams:
        for alias in _alias_variants(team.team_id):
            alias_lookup.setdefault(alias, team.team_id)
        for alias in _alias_variants(team.name):
            alias_lookup.setdefault(alias, team.team_id)

    return alias_lookup


def _resolve_team_id(team_raw: str, alias_lookup: dict[str, str]) -> str | None:
    variants = _alias_variants(team_raw)
    matched_team_ids = {alias_lookup[variant] for variant in variants if variant in alias_lookup}

    if not matched_team_ids:
        return None
    if len(matched_team_ids) > 1:
        msg = f"Ambiguous ratings team name '{team_raw}' resolved to {sorted(matched_team_ids)}"
        raise ValueError(msg)

    return next(iter(matched_team_ids))


def _alias_variants(value: str) -> set[str]:
    base = _normalize_team_key(value)
    if base == "":
        return set()

    variants = {base}

    tokens = base.split()
    if tokens:
        swapped_state = ["state" if token in {"st", "st."} else token for token in tokens]
        variants.add(" ".join(swapped_state))

        swapped_st = ["st" if token == "state" else token for token in tokens]
        variants.add(" ".join(swapped_st))

        swapped_saint = ["saint" if token in {"st", "st."} else token for token in tokens]
        variants.add(" ".join(swapped_saint))

        swapped_to_st = ["st" if token == "saint" else token for token in tokens]
        variants.add(" ".join(swapped_to_st))

    variants.add(base.replace(" and ", " "))
    variants.add(base.replace(" ", ""))

    expanded: set[str] = set(variants)
    for variant in list(variants):
        for special_alias in _SPECIAL_ALIAS_EQUIVALENTS.get(variant, set()):
            expanded.add(special_alias)
            expanded.add(special_alias.replace(" ", ""))

    variants = expanded
    return {variant for variant in variants if variant != ""}


def _normalize_team_key(value: str) -> str:
    decoded = html.unescape(value)
    lowered = decoded.casefold()
    lowered = lowered.replace("&", " and ")
    lowered = lowered.replace("'", "")
    lowered = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _normalize_header(value: str) -> str:
    key = _normalize_team_key(value).replace(" ", "")
    if key in {"team", "school"}:
        return "team"
    if key in {"adjem"}:
        return "adjem"
    if key in {"adjt", "adjtempo", "tempo"}:
        return "adjt"
    return key


def _strip_tags(value: str) -> str:
    stripped = re.sub(r"<[^>]+>", "", value)
    return _clean_text(stripped)


def _clean_text(value: object) -> str:
    return html.unescape(str(value or "")).strip()
