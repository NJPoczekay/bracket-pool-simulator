from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bracket_sim.infrastructure.providers.contracts import RawAliasRow, RawRatingRow, RawTeamRow
from bracket_sim.infrastructure.providers.ratings import (
    KenPomRatingSourceProvider,
    KenPomRatingsProvider,
    LocalRatingSourceProvider,
    LocalRatingsProvider,
    normalize_rating_rows,
)


def test_local_ratings_provider_surfaces_unresolved_names_with_suggestions(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "team,rating,tempo\n"
        "UConnn,30.1,66.2\n"
        "Florida,28.8,68.1\n",
        encoding="utf-8",
    )

    provider = LocalRatingsProvider(ratings_file=ratings_path)
    teams = [
        RawTeamRow(team_id="uconn", name="UConn", seed=1, region="east"),
        RawTeamRow(team_id="florida", name="Florida", seed=1, region="west"),
    ]

    with pytest.raises(ValueError, match="did you mean source rows"):
        provider.fetch_ratings(teams=teams)


def test_local_ratings_provider_normalizes_to_team_ids(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "team,rating,tempo\n"
        "UConn,30.1,66.2\n"
        "Florida,28.8,68.1\n",
        encoding="utf-8",
    )

    provider = LocalRatingsProvider(ratings_file=ratings_path)
    teams = [
        RawTeamRow(team_id="uconn", name="UConn", seed=1, region="east"),
        RawTeamRow(team_id="florida", name="Florida", seed=1, region="west"),
    ]

    ratings = provider.fetch_ratings(teams=teams)
    assert [row.team for row in ratings.ratings] == ["florida", "uconn"]


def test_local_rating_source_provider_preserves_raw_team_names(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "team,rating,tempo\n"
        "Miami OH,18.4,67.0\n"
        "SMU,20.1,69.2\n",
        encoding="utf-8",
    )

    provider = LocalRatingSourceProvider(ratings_file=ratings_path)
    source = provider.fetch_rating_source()

    assert [row.team for row in source.ratings] == ["Miami OH", "SMU"]


def test_normalize_rating_rows_supports_manual_aliases_for_play_in_candidates() -> None:
    ratings, aliases = normalize_rating_rows(
        input_rows=[
            RawRatingRow(team="Miami OH", rating=18.4, tempo=67.0),
            RawRatingRow(team="SMU", rating=20.1, tempo=69.2),
        ],
        teams=[
            RawTeamRow(team_id="playin-m-oh", name="M-OH", seed=11, region="south"),
            RawTeamRow(team_id="playin-smu", name="SMU", seed=11, region="south"),
        ],
        aliases=[
            RawAliasRow(alias="Miami OH", team_id="playin-m-oh"),
        ],
    )

    assert [row.team for row in ratings] == ["playin-m-oh", "playin-smu"]
    assert aliases == [RawAliasRow(alias="Miami OH", team_id="playin-m-oh")]


def test_kenpom_provider_requires_cookie(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KENPOM_COOKIE", raising=False)
    provider = KenPomRatingsProvider(
        client=httpx.Client(transport=httpx.MockTransport(_ok_handler))
    )

    with pytest.raises(ValueError, match="KENPOM_COOKIE"):
        provider.fetch_ratings(teams=[RawTeamRow(team_id="a", name="A", seed=1, region="x")])


def test_kenpom_provider_surfaces_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KENPOM_COOKIE", "session=abc")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    provider = KenPomRatingsProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(ValueError, match="authentication failed"):
        provider.fetch_ratings(teams=[RawTeamRow(team_id="a", name="A", seed=1, region="x")])


def test_kenpom_provider_surfaces_parse_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KENPOM_COOKIE", "session=abc")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>no table here</body></html>")

    provider = KenPomRatingsProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(ValueError, match="ratings table"):
        provider.fetch_ratings(teams=[RawTeamRow(team_id="a", name="A", seed=1, region="x")])


def test_kenpom_provider_parses_minimal_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KENPOM_COOKIE", "session=abc")

    html_payload = """
    <table id="ratings-table">
      <tr>
        <th>Rk</th><th>Team</th><th>Conf</th><th>W-L</th><th>AdjEM</th>
        <th>AdjO</th><th>AdjD</th><th>AdjEM.1</th><th>AdjT</th>
      </tr>
      <tr>
        <td>1</td><td><a href='team.php?team=UConn'>UConn</a></td><td>BE</td>
        <td>32-3</td><td>30.1</td><td>1</td><td>2</td><td>1</td><td>66.2</td>
      </tr>
      <tr>
        <td>2</td><td>Florida</td><td>SEC</td><td>30-5</td>
        <td>28.8</td><td>2</td><td>8</td><td>2</td><td>68.1</td>
      </tr>
    </table>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html_payload)

    provider = KenPomRatingsProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    teams = [
        RawTeamRow(team_id="uconn", name="UConn", seed=1, region="east"),
        RawTeamRow(team_id="florida", name="Florida", seed=1, region="west"),
    ]

    ratings = provider.fetch_ratings(teams=teams)
    assert [row.team for row in ratings.ratings] == ["florida", "uconn"]


def test_kenpom_rating_source_provider_returns_raw_team_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KENPOM_COOKIE", "session=abc")

    html_payload = """
    <table id="ratings-table">
      <tr>
        <th>Rk</th><th>Team</th><th>Conf</th><th>W-L</th><th>AdjEM</th>
        <th>AdjO</th><th>AdjD</th><th>AdjEM.1</th><th>AdjT</th>
      </tr>
      <tr>
        <td>1</td><td>Miami OH</td><td>MAC</td><td>25-9</td><td>18.4</td>
        <td>1</td><td>2</td><td>1</td><td>67.0</td>
      </tr>
    </table>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html_payload)

    provider = KenPomRatingSourceProvider(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    source = provider.fetch_rating_source()

    assert [row.team for row in source.ratings] == ["Miami OH"]


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="<html></html>")
