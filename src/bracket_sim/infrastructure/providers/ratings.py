"""Ratings providers for refresh-data (local default, optional KenPom)."""

from __future__ import annotations

import csv
import difflib
import html
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from bracket_sim.infrastructure.providers.contracts import (
    RatingsData,
    RatingSourceData,
    RatingSourceProvider,
    RatingsProvider,
    RawAliasRow,
    RawRatingRow,
    RawTeamRow,
)

_SPECIAL_ALIAS_EQUIVALENTS: dict[str, set[str]] = {
    "ca baptist": {"cal baptist", "california baptist"},
    "cal baptist": {"ca baptist", "california baptist"},
    "california baptist": {"ca baptist", "cal baptist"},
    "leh": {"lehigh", "lehigh mountain hawks"},
    "lehigh": {"leh", "lehigh mountain hawks"},
    "lehigh mountain hawks": {"leh", "lehigh"},
    "liu": {"long island"},
    "long island": {"liu"},
    "m oh": {"miami oh", "miami ohio"},
    "miami": {"miami fl"},
    "miami fl": {"miami"},
    "miami oh": {"m oh", "miami ohio"},
    "miami ohio": {"m oh", "miami oh"},
    "n dakota st": {"north dakota st", "north dakota state"},
    "north dakota st": {"n dakota st", "ndsu"},
    "north dakota state": {"n dakota st", "ndsu"},
    "ndsu": {"n dakota st", "north dakota st", "north dakota state"},
    "uconn": {"connecticut"},
    "connecticut": {"uconn"},
    "ole miss": {"mississippi"},
    "mississippi": {"ole miss"},
    "omaha": {"nebraska omaha"},
    "nebraska omaha": {"omaha"},
    "pv": {"prairie view", "prairie view a and m", "pvamu", "pva and m"},
    "prairie view": {"pv", "prairie view a and m", "pvamu", "pva and m"},
    "prairie view a and m": {"pv", "prairie view", "pvamu", "pva and m"},
    "pvamu": {"pv", "prairie view", "prairie view a and m", "pva and m"},
    "pva and m": {"pv", "prairie view", "prairie view a and m", "pvamu"},
}

DEFAULT_KENPOM_SNAPSHOT_DIR = Path("data") / "kenpom_snapshots"
_KENPOM_SNAPSHOT_STEM = "Pomeroy College Basketball Ratings"


@dataclass(frozen=True)
class _InputRatingRow:
    team_raw: str
    rating: float
    tempo: float


class LocalRatingSourceProvider(RatingSourceProvider):
    """Load raw rating rows from a local CSV or cached snapshot."""

    def __init__(
        self,
        *,
        ratings_file: Path | None = None,
        fallback_dir: Path | None = None,
    ) -> None:
        self._ratings_file = ratings_file
        self._fallback_dir = fallback_dir

    def fetch_rating_source(self) -> RatingSourceData:
        ratings_path = self._resolve_ratings_path()
        return RatingSourceData(
            ratings=load_source_rating_rows(ratings_path),
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


class LocalRatingsProvider(RatingsProvider):
    """Load ratings from a local CSV or cached snapshot."""

    def __init__(
        self,
        *,
        ratings_file: Path | None = None,
        fallback_dir: Path | None = None,
    ) -> None:
        self._source_provider = LocalRatingSourceProvider(
            ratings_file=ratings_file,
            fallback_dir=fallback_dir,
        )

    def fetch_ratings(self, *, teams: list[RawTeamRow]) -> RatingsData:
        source_data = self._source_provider.fetch_rating_source()
        normalized_ratings, aliases = normalize_rating_rows(
            input_rows=source_data.ratings,
            teams=teams,
        )
        return RatingsData(
            ratings=normalized_ratings,
            aliases=aliases,
            source=source_data.source,
        )


class KenPomRatingSourceProvider(RatingSourceProvider):
    """Fetch raw rating rows from a local KenPom snapshot or live page."""

    def __init__(
        self,
        *,
        url: str = "https://kenpom.com/",
        timeout_seconds: float = 20.0,
        client: httpx.Client | None = None,
        snapshot_path: Path | None = None,
        snapshot_dir: Path | None = None,
        season: str | None = None,
    ) -> None:
        self._url = url
        self._client = client or httpx.Client(timeout=timeout_seconds, follow_redirects=True)
        self._owns_client = client is None
        self._snapshot_path = snapshot_path
        self._snapshot_dir = snapshot_dir
        self._season = season

    def close(self) -> None:
        """Close owned HTTP resources."""

        if self._owns_client:
            self._client.close()

    def fetch_rating_source(self) -> RatingSourceData:
        snapshot_path = self._resolve_snapshot_path()
        if snapshot_path is not None:
            try:
                payload = snapshot_path.read_text(encoding="utf-8")
            except OSError as exc:
                msg = f"Failed to read KenPom snapshot at {snapshot_path}: {exc}"
                raise ValueError(msg) from exc

            return RatingSourceData(
                ratings=parse_kenpom_source_rows(payload),
                source=f"kenpom_snapshot:{snapshot_path}",
            )

        try:
            response = self._client.get(self._url)
        except httpx.HTTPError as exc:
            msg = f"Failed KenPom request: {exc}"
            raise ValueError(msg) from exc

        if response.status_code in {401, 403}:
            msg = (
                "KenPom request was rejected (401/403). "
                "The public ratings page may have changed or become unavailable."
            )
            raise ValueError(msg)

        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            msg = f"KenPom request failed with status {response.status_code}: {exc}"
            raise ValueError(msg) from exc

        return RatingSourceData(
            ratings=parse_kenpom_source_rows(response.text),
            source="kenpom",
        )

    def _resolve_snapshot_path(self) -> Path | None:
        if self._snapshot_path is not None:
            if not self._snapshot_path.exists():
                msg = f"KenPom snapshot does not exist: {self._snapshot_path}"
                raise ValueError(msg)
            return self._snapshot_path

        if self._snapshot_dir is None:
            return None

        if not self._snapshot_dir.exists():
            msg = (
                f"KenPom snapshot directory does not exist: {self._snapshot_dir}. "
                "Save an HTML snapshot there or use --ratings-file."
            )
            raise ValueError(msg)

        season = self._season or str(datetime.now(UTC).year)
        exact_candidates = [
            self._snapshot_dir / f"{season} {_KENPOM_SNAPSHOT_STEM}.html",
            self._snapshot_dir / f"{season} {_KENPOM_SNAPSHOT_STEM}.htm",
        ]
        for candidate in exact_candidates:
            if candidate.exists():
                return candidate

        pattern = f"{season}*{_KENPOM_SNAPSHOT_STEM}*.htm*"
        fallback_candidates = sorted(
            self._snapshot_dir.glob(pattern),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        if fallback_candidates:
            return fallback_candidates[0]

        msg = (
            f"No KenPom snapshot found for season {season} in {self._snapshot_dir}. "
            f"Expected a file like '{season} {_KENPOM_SNAPSHOT_STEM}.html' or use --ratings-file."
        )
        raise ValueError(msg)


class KenPomRatingsProvider(RatingsProvider):
    """Fetch ratings from a local KenPom snapshot or live page."""

    def __init__(
        self,
        *,
        url: str = "https://kenpom.com/",
        timeout_seconds: float = 20.0,
        client: httpx.Client | None = None,
        snapshot_path: Path | None = None,
        snapshot_dir: Path | None = None,
        season: str | None = None,
    ) -> None:
        self._source_provider = KenPomRatingSourceProvider(
            url=url,
            timeout_seconds=timeout_seconds,
            client=client,
            snapshot_path=snapshot_path,
            snapshot_dir=snapshot_dir,
            season=season,
        )

    def close(self) -> None:
        """Close owned HTTP resources."""

        self._source_provider.close()

    def fetch_ratings(self, *, teams: list[RawTeamRow]) -> RatingsData:
        source_data = self._source_provider.fetch_rating_source()
        normalized_ratings, aliases = normalize_rating_rows(
            input_rows=source_data.ratings,
            teams=teams,
        )
        return RatingsData(ratings=normalized_ratings, aliases=aliases, source=source_data.source)


def load_source_rating_rows(path: Path) -> list[RawRatingRow]:
    """Load raw rating source rows from a local CSV."""

    return [
        RawRatingRow(team=row.team_raw, rating=row.rating, tempo=row.tempo)
        for row in _load_ratings_csv(path)
    ]


def parse_kenpom_source_rows(payload: str) -> list[RawRatingRow]:
    """Parse raw rating source rows from a KenPom HTML payload."""

    return [
        RawRatingRow(team=row.team_raw, rating=row.rating, tempo=row.tempo)
        for row in _parse_kenpom_table(payload)
    ]


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
        header_cells = re.findall(
            r"<th([^>]*)>(.*?)</th>",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if header_cells:
            normalized_headers: list[str] = []
            for attrs, cell in header_cells:
                normalized_header = _normalize_header(_strip_tags(cell))
                colspan_match = re.search(
                    r"colspan\s*=\s*[\"']?(\d+)[\"']?",
                    attrs,
                    flags=re.IGNORECASE,
                )
                colspan = int(colspan_match.group(1)) if colspan_match is not None else 1
                normalized_headers.extend([normalized_header] * colspan)
            header_map = {}
            for idx, header in enumerate(normalized_headers):
                header_map.setdefault(header, idx)
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
            msg = "KenPom ratings table missing expected columns Team/AdjEM(or NetRtg)/AdjT"
            raise ValueError(msg)

        if max(header_map[header] for header in required_headers) >= len(data_cells):
            continue

        team_cell_html = re.sub(
            r"<span[^>]*class=[\"'][^\"']*\bseed\b[^\"']*[\"'][^>]*>.*?</span>",
            "",
            data_cells[header_map["team"]],
            flags=re.IGNORECASE | re.DOTALL,
        )
        team_raw = _strip_tags(team_cell_html)
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


def normalize_rating_rows(
    *,
    input_rows: list[RawRatingRow],
    teams: list[RawTeamRow],
    aliases: list[RawAliasRow] | None = None,
) -> tuple[list[RawRatingRow], list[RawAliasRow]]:
    """Normalize raw source rows to tournament team ids."""

    manual_aliases = aliases or []
    alias_lookup = _build_alias_lookup(teams)
    team_name_by_id = {team.team_id: team.name for team in teams}
    unresolved_source_names: list[str] = []

    ratings_by_team_id: dict[str, RawRatingRow] = {}
    alias_rows: dict[str, RawAliasRow] = {}

    seen_manual_aliases: set[str] = set()
    team_ids = {team.team_id for team in teams}
    for alias in manual_aliases:
        normalized_alias = _normalize_team_key(alias.alias)
        if normalized_alias == "":
            msg = "Alias rows must not contain blank alias values"
            raise ValueError(msg)
        if normalized_alias in seen_manual_aliases:
            msg = f"Duplicate alias declared in aliases.csv: '{alias.alias}'"
            raise ValueError(msg)
        seen_manual_aliases.add(normalized_alias)
        if alias.team_id not in team_ids:
            msg = f"Alias '{alias.alias}' references unknown team_id '{alias.team_id}'"
            raise ValueError(msg)
        existing = alias_lookup.get(normalized_alias)
        if existing is not None and existing != alias.team_id:
            msg = (
                f"Alias collision for '{alias.alias}': maps to both '{existing}' "
                f"and '{alias.team_id}'"
            )
            raise ValueError(msg)
        alias_lookup[normalized_alias] = alias.team_id

    for row in input_rows:
        resolved_team_id = _resolve_team_id(row.team, alias_lookup)
        if resolved_team_id is None:
            unresolved_source_names.append(row.team)
            continue

        if resolved_team_id in ratings_by_team_id:
            msg = f"Duplicate rating row resolved to team '{resolved_team_id}'"
            raise ValueError(msg)

        ratings_by_team_id[resolved_team_id] = RawRatingRow(
            team=resolved_team_id,
            rating=row.rating,
            tempo=row.tempo,
        )

        normalized_team_raw = _normalize_team_key(row.team)
        canonical_name = next(team.name for team in teams if team.team_id == resolved_team_id)
        if normalized_team_raw not in {
            _normalize_team_key(resolved_team_id),
            _normalize_team_key(canonical_name),
        }:
            alias_rows[normalized_team_raw] = RawAliasRow(
                alias=row.team,
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
    if key in {"adjem", "netrtg", "adjnetrtg", "efficiencymargin"}:
        return "adjem"
    if key in {"adjt", "adjtempo", "tempo"}:
        return "adjt"
    return key


def _strip_tags(value: str) -> str:
    stripped = re.sub(r"<[^>]+>", "", value)
    return _clean_text(stripped)


def _clean_text(value: object) -> str:
    return html.unescape(str(value or "")).strip()
