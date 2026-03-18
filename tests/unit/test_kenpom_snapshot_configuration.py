from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

import bracket_sim.application.refresh_bracket_lab_data as refresh_bracket_lab_module
import bracket_sim.application.refresh_data as refresh_data_module
from bracket_sim.application.refresh_bracket_lab_data import refresh_bracket_lab_data
from bracket_sim.application.refresh_data import refresh_data
from bracket_sim.infrastructure.providers.contracts import (
    ChallengeSnapshotData,
    EntriesData,
    NationalPicksData,
    RatingsData,
    RatingSourceData,
    RawEntryRow,
    RawRatingRow,
    RawTeamRow,
    ResultsData,
)


class _StubResultsProvider:
    def fetch_results(self) -> ResultsData:
        return ResultsData(
            teams=[RawTeamRow(team_id="team-a", name="Team A", seed=1, region="east")],
            games=[],
            constraints=[],
            outcome_team_id_by_outcome_id={},
            proposition_status_counts={},
            correct_outcome_counts={},
            challenge_state=None,
            challenge_scoring_status=None,
            challenge_key="mock-challenge-2026",
            api_shape_hints={},
            raw_snapshot={},
        )


class _StubEntriesProvider:
    def fetch_entries(
        self,
        *,
        proposition_ids: set[str],
        outcome_team_id_by_outcome_id: dict[str, str],
    ) -> EntriesData:
        return EntriesData(
            entries=[RawEntryRow(entry_id="entry-1", entry_name="Entry 1", picks={})],
            total_entries=1,
            skipped_entries=[],
            retry_attempted=False,
            api_shape_hints={},
            raw_snapshot={},
            raw_retry_snapshot=None,
        )


class _StubChallengeSnapshotProvider:
    def fetch_challenge_snapshot(self) -> ChallengeSnapshotData:
        return ChallengeSnapshotData(
            results=ResultsData(
                teams=[RawTeamRow(team_id="team-a", name="Team A", seed=1, region="east")],
                games=[],
                constraints=[],
                outcome_team_id_by_outcome_id={},
                proposition_status_counts={},
                correct_outcome_counts={},
                challenge_state=None,
                challenge_scoring_status=None,
                challenge_key="mock-challenge-2026",
                api_shape_hints={},
                raw_snapshot={},
            ),
            national_picks=NationalPicksData(
                rows=[],
                total_brackets=0,
                round_counts={},
                challenge_id=None,
                challenge_key="mock-challenge-2026",
                challenge_name=None,
                challenge_state=None,
                proposition_lock_date=None,
                proposition_lock_date_passed=None,
                source_url="https://example.test/mock-challenge-2026",
                api_shape_hints={},
                raw_snapshot={},
            ),
        )


def test_refresh_data_builds_kenpom_provider_with_seasonal_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    init_kwargs: dict[str, object] = {}

    class FakeKenPomRatingsProvider:
        def __init__(self, **kwargs: object) -> None:
            init_kwargs.update(kwargs)

        def close(self) -> None:
            pass

        def fetch_ratings(self, *, teams: list[RawTeamRow]) -> RatingsData:
            assert [team.team_id for team in teams] == ["team-a"]
            return RatingsData(
                ratings=[RawRatingRow(team="team-a", rating=20.1, tempo=67.0)],
                aliases=[],
                source="kenpom_snapshot:fake",
            )

    monkeypatch.setattr(refresh_data_module, "KenPomRatingsProvider", FakeKenPomRatingsProvider)

    summary = refresh_data(
        group_url="https://fantasy.espn.com/games/mock-challenge-2026/group?id=mock-group-2026",
        raw_dir=tmp_path / "raw",
        use_kenpom=True,
        results_provider=_StubResultsProvider(),
        entries_provider=_StubEntriesProvider(),
        fetched_at=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )

    assert summary.ratings == 1
    assert init_kwargs == {
        "season": "2026",
        "snapshot_dir": Path("data") / "kenpom_snapshots",
    }


def test_refresh_bracket_lab_data_builds_kenpom_provider_with_seasonal_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    init_kwargs: dict[str, object] = {}

    class FakeKenPomRatingSourceProvider:
        def __init__(self, **kwargs: object) -> None:
            init_kwargs.update(kwargs)

        def close(self) -> None:
            pass

        def fetch_rating_source(self) -> RatingSourceData:
            return RatingSourceData(
                ratings=[RawRatingRow(team="Team A", rating=20.1, tempo=67.0)],
                source="kenpom_snapshot:fake",
            )

    monkeypatch.setattr(
        refresh_bracket_lab_module,
        "KenPomRatingSourceProvider",
        FakeKenPomRatingSourceProvider,
    )

    summary = refresh_bracket_lab_data(
        challenge="mock-challenge-2026",
        raw_dir=tmp_path / "raw_lab",
        use_kenpom=True,
        challenge_provider=_StubChallengeSnapshotProvider(),
        fetched_at=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )

    assert summary.kenpom_rows == 1
    assert init_kwargs == {
        "season": "2026",
        "snapshot_dir": Path("data") / "kenpom_snapshots",
    }
