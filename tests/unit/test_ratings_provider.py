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


def test_normalize_rating_rows_supports_curated_espn_to_kenpom_aliases() -> None:
    ratings, aliases = normalize_rating_rows(
        input_rows=[
            RawRatingRow(team="LIU", rating=-3.96, tempo=67.8),
            RawRatingRow(team="Miami FL", rating=20.67, tempo=67.6),
            RawRatingRow(team="North Dakota St.", rating=5.0, tempo=66.3),
            RawRatingRow(team="Cal Baptist", rating=5.99, tempo=65.8),
            RawRatingRow(team="Miami OH", rating=8.27, tempo=70.0),
            RawRatingRow(team="Prairie View A&M", rating=-10.69, tempo=71.0),
            RawRatingRow(team="Lehigh", rating=-10.41, tempo=66.9),
        ],
        teams=[
            RawTeamRow(team_id="liu", name="Long Island", seed=16, region="east"),
            RawTeamRow(team_id="miami", name="Miami", seed=7, region="east"),
            RawTeamRow(team_id="ndsu", name="N Dakota St", seed=14, region="midwest"),
            RawTeamRow(team_id="cbu", name="CA Baptist", seed=13, region="midwest"),
            RawTeamRow(team_id="playin-m-oh", name="M-OH", seed=11, region="south"),
            RawTeamRow(team_id="playin-pv", name="PV", seed=16, region="west"),
            RawTeamRow(team_id="playin-leh", name="LEH", seed=16, region="west"),
        ],
    )

    assert [row.team for row in ratings] == [
        "cbu",
        "liu",
        "miami",
        "ndsu",
        "playin-leh",
        "playin-m-oh",
        "playin-pv",
    ]
    assert aliases == [
        RawAliasRow(alias="Cal Baptist", team_id="cbu"),
        RawAliasRow(alias="Lehigh", team_id="playin-leh"),
        RawAliasRow(alias="Miami FL", team_id="miami"),
        RawAliasRow(alias="Miami OH", team_id="playin-m-oh"),
        RawAliasRow(alias="North Dakota St.", team_id="ndsu"),
        RawAliasRow(alias="Prairie View A&M", team_id="playin-pv"),
    ]


def test_kenpom_provider_uses_public_request_without_cookie(
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
        <td>1</td><td>A</td><td>X</td><td>1-0</td><td>30.1</td>
        <td>1</td><td>2</td><td>1</td><td>66.2</td>
      </tr>
    </table>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert "cookie" not in request.headers
        return httpx.Response(200, text=html_payload)

    provider = KenPomRatingsProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    ratings = provider.fetch_ratings(teams=[RawTeamRow(team_id="a", name="A", seed=1, region="x")])

    assert [row.team for row in ratings.ratings] == ["a"]


def test_kenpom_provider_surfaces_auth_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    provider = KenPomRatingsProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(ValueError, match="rejected"):
        provider.fetch_ratings(teams=[RawTeamRow(team_id="a", name="A", seed=1, region="x")])


def test_kenpom_provider_surfaces_parse_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>no table here</body></html>")

    provider = KenPomRatingsProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(ValueError, match="ratings table"):
        provider.fetch_ratings(teams=[RawTeamRow(team_id="a", name="A", seed=1, region="x")])


def test_kenpom_provider_parses_minimal_table() -> None:
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
) -> None:
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


def test_kenpom_provider_reads_local_snapshot_for_requested_season(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "kenpom_snapshots"
    snapshot_dir.mkdir()
    snapshot_path = snapshot_dir / "2026 Pomeroy College Basketball Ratings.html"
    snapshot_path.write_text(
        """
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
        """,
        encoding="utf-8",
    )

    provider = KenPomRatingSourceProvider(snapshot_dir=snapshot_dir, season="2026")
    source = provider.fetch_rating_source()

    assert [row.team for row in source.ratings] == ["Miami OH"]
    assert source.source == f"kenpom_snapshot:{snapshot_path}"


def test_kenpom_provider_requires_expected_local_snapshot_when_configured(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "kenpom_snapshots"
    snapshot_dir.mkdir()

    provider = KenPomRatingSourceProvider(snapshot_dir=snapshot_dir, season="2026")

    with pytest.raises(ValueError, match="No KenPom snapshot found for season 2026"):
        provider.fetch_rating_source()


def test_kenpom_provider_parses_current_snapshot_header_names() -> None:
    html_payload = """
    <table id="ratings-table">
      <thead>
        <tr class="thead1">
          <th></th><th></th><th colspan="3"></th><th colspan="4"></th>
          <th colspan="2"></th><th colspan="2"></th><th colspan="6">Strength of Schedule</th>
        </tr>
        <tr class="thead2">
          <th>Rk</th><th>Team</th><th>Conf</th><th>W-L</th><th>NetRtg</th>
          <th colspan="2">ORtg</th><th colspan="2">DRtg</th>
          <th colspan="2">AdjT</th><th colspan="2">Luck</th>
          <th colspan="2">NetRtg</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td><td><a href="team.php?team=Duke">Duke</a> <span class="seed">1</span></td>
          <td>ACC</td><td>32-2</td><td>+38.88</td>
          <td>128.0</td><td>4</td><td>89.1</td><td>2</td><td>65.4</td><td>287</td>
          <td>+.049</td><td>62</td><td>+14.29</td><td>15</td>
        </tr>
      </tbody>
    </table>
    """

    rows = KenPomRatingSourceProvider(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text=html_payload))
        )
    ).fetch_rating_source().ratings

    assert rows == [RawRatingRow(team="Duke", rating=38.88, tempo=65.4)]
