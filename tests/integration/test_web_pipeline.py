from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from tests.helpers.mock_espn_payloads import build_mock_payloads

from bracket_sim.application.run_pool_pipeline import PoolPipelineResult, run_pool_pipeline
from bracket_sim.infrastructure.providers.espn_api import EspnApiProvider
from bracket_sim.infrastructure.providers.ratings import LocalRatingsProvider
from bracket_sim.infrastructure.web.config import PoolProfile, PoolRegistry
from bracket_sim.infrastructure.web.main import create_app
from bracket_sim.infrastructure.web.service import PoolService


def test_web_run_endpoint_executes_full_pipeline_with_fixture_backed_data(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    challenge_payload, group_payload, _ = build_mock_payloads(
        fixture_dir=synthetic_input_dir,
        completed_game_ids=set(),
    )
    ratings_path = tmp_path / "ratings.csv"
    _write_team_id_ratings(ratings_path, synthetic_input_dir)

    registry = PoolRegistry(
        pools=[
            PoolProfile(
                id="alpha",
                name="Alpha Pool",
                group_url=(
                    "https://fantasy.espn.com/games/mock-challenge-2026/group?id=mock-group-2026"
                ),
                raw_dir=tmp_path / "raw",
                prepared_dir=tmp_path / "prepared",
                reports_root=tmp_path / "reports",
                ratings_file=ratings_path,
                use_kenpom=False,
                min_usable_entries=1,
                n_sims=120,
                seed=7,
                batch_size=40,
                engine="numpy",
            )
        ]
    )

    def runner(config: PoolProfile, *, started_at: datetime) -> PoolPipelineResult:
        provider = _build_provider(
            challenge_payload=challenge_payload,
            group_payloads=[group_payload],
        )
        return run_pool_pipeline(
            config,
            started_at=started_at,
            results_provider=provider,
            entries_provider=provider,
            ratings_provider=LocalRatingsProvider(ratings_file=ratings_path),
        )

    service = PoolService(registry, runner=runner)

    app = create_app(
        config_path=tmp_path / "unused.toml",
        service=service,
        enable_scheduler=False,
    )
    with TestClient(app) as client:
        run_response = client.post("/api/pools/alpha/run")
        dashboard_response = client.get("/")
        latest_response = client.get("/api/pools/alpha/latest-report")

    assert run_response.status_code == 200
    assert dashboard_response.status_code == 200
    assert latest_response.status_code == 200

    latest_payload = latest_response.json()["latest_report"]
    report_dir = Path(latest_payload["report_dir"])

    assert (tmp_path / "raw" / "teams.csv").exists()
    assert (tmp_path / "raw" / "entries.json").exists()
    assert (tmp_path / "prepared" / "teams.json").exists()
    assert (tmp_path / "prepared" / "ratings.csv").exists()
    assert (report_dir / "summary.json").exists()
    assert (report_dir / "manifest.json").exists()
    assert (report_dir / "entry_summary.csv").exists()
    assert (tmp_path / "reports" / "latest" / "summary.json").exists()
    assert "Bracket Lab" in dashboard_response.text
    assert "Pool Tracker" in dashboard_response.text
    assert "Top Entries" in dashboard_response.text
    assert "Top Champions" in dashboard_response.text


def test_web_shell_renders_live_bracket_lab_editor_analysis_and_optimizer_api(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    app = create_app(
        bracket_lab_input=prepared_bracket_lab_dir,
        enable_scheduler=False,
    )
    with TestClient(app) as client:
        dashboard_response = client.get("/")
        complete_response = client.post(
            "/api/bracket-lab/complete",
            json={
                "bracket": {
                    "picks": [
                        {
                            **_editable_bracket_payload(synthetic_input_dir)[0],
                            "locked": True,
                        }
                    ],
                },
                "completion_mode": "tournament_seeds",
                "pick_four": {
                    "regional_winner_seeds": {
                        "east": 1,
                        "west": 1,
                        "south": 1,
                        "midwest": 1,
                    }
                },
            },
        )
        analyze_response = client.post(
            "/api/bracket-lab/analyze",
            json={
                "bracket": {
                    "picks": _editable_bracket_payload(synthetic_input_dir),
                },
                "pool_settings": {
                    "pool_size": 12,
                    "scoring_system": "round+seed",
                },
                "completion_mode": "manual",
            },
        )
        optimize_response = client.post(
            "/api/bracket-lab/optimize",
            json={
                "bracket": {
                    "picks": _editable_bracket_payload(synthetic_input_dir),
                },
                "pool_settings": {
                    "pool_size": 12,
                    "scoring_system": "round+seed",
                },
                "completion_mode": "manual",
            },
        )

    assert dashboard_response.status_code == 200
    assert "Completion + Analyzer + Optimizer" in dashboard_response.text
    assert "Finish Picks" in dashboard_response.text
    assert "Pick Four" in dashboard_response.text
    assert "Analyze Bracket" in dashboard_response.text
    assert "Optimize Bracket" in dashboard_response.text
    assert "Dataset hash" in dashboard_response.text
    assert 'id="bracket-lab-editor-layout"' in dashboard_response.text
    assert 'id="bracket-mobile-tabs"' in dashboard_response.text
    assert 'id="bracket-lab-desktop"' in dashboard_response.text
    assert (
        "63 picks remaining before analysis. Manual picks stay locked."
        in dashboard_response.text
    )

    assert complete_response.status_code == 200
    completion_payload = complete_response.json()
    assert completion_payload["state"] == "auto_completed"
    assert completion_payload["completion_mode"] == "tournament_seeds"
    assert completion_payload["auto_filled_pick_count"] == 62

    assert analyze_response.status_code == 200
    payload = analyze_response.json()
    assert payload["public_percentile"] is None
    assert payload["pool_settings"]["scoring_system"] == "round+seed"
    assert payload["pick_diagnostics"][0]["tags"] is not None

    assert optimize_response.status_code == 200
    optimizer_payload = optimize_response.json()
    assert optimizer_payload["cache_key"].startswith("optimization-")
    assert optimizer_payload["summary"]
    assert len(optimizer_payload["alternatives"]) <= 3


@pytest.mark.parametrize(
    "scoring_system",
    ["round-of-64-flat", "round-of-64-seed"],
)
def test_web_bracket_lab_analysis_accepts_round_of_64_scoring_systems(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
    scoring_system: str,
) -> None:
    app = create_app(
        bracket_lab_input=prepared_bracket_lab_dir,
        enable_scheduler=False,
    )
    with TestClient(app) as client:
        analyze_response = client.post(
            "/api/bracket-lab/analyze",
            json={
                "bracket": {
                    "picks": _editable_bracket_payload(synthetic_input_dir),
                },
                "pool_settings": {
                    "pool_size": 12,
                    "scoring_system": scoring_system,
                },
                "completion_mode": "manual",
            },
        )

    assert analyze_response.status_code == 200
    payload = analyze_response.json()
    assert payload["pool_settings"]["scoring_system"] == scoring_system
    assert payload["public_percentile"] is None
    assert len(payload["pick_diagnostics"]) == 63
    assert payload["cache_key"].startswith("analysis-")


@pytest.mark.parametrize(
    "scoring_system",
    ["round-of-64-flat", "round-of-64-seed"],
)
def test_web_bracket_lab_optimizer_accepts_round_of_64_scoring_systems(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
    scoring_system: str,
) -> None:
    app = create_app(
        bracket_lab_input=prepared_bracket_lab_dir,
        enable_scheduler=False,
    )
    with TestClient(app) as client:
        optimize_response = client.post(
            "/api/bracket-lab/optimize",
            json={
                "bracket": {
                    "picks": _editable_bracket_payload(synthetic_input_dir),
                },
                "pool_settings": {
                    "pool_size": 12,
                    "scoring_system": scoring_system,
                },
                "completion_mode": "manual",
            },
        )

    assert optimize_response.status_code == 200
    payload = optimize_response.json()
    assert payload["pool_settings"]["scoring_system"] == scoring_system
    assert payload["cache_key"].startswith("optimization-")
    assert payload["changed_pick_count"] == len(payload["changed_picks"])


def _build_provider(
    *,
    challenge_payload: dict[str, Any],
    group_payloads: list[dict[str, Any]],
) -> EspnApiProvider:
    group_call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal group_call_count

        if request.url.path == "/apis/v1/challenges/mock-challenge-2026":
            return httpx.Response(200, json=challenge_payload)

        if request.url.path == "/apis/v1/challenges/mock-challenge-2026/groups/mock-group-2026":
            payload = group_payloads[min(group_call_count, len(group_payloads) - 1)]
            group_call_count += 1
            return httpx.Response(200, json=payload)

        return httpx.Response(404, json={"error": "not-found"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return EspnApiProvider(
        group_url="https://fantasy.espn.com/games/mock-challenge-2026/group?id=mock-group-2026",
        api_base_url="https://mock.espn.test",
        client=client,
    )


def _write_team_id_ratings(path: Path, fixture_dir: Path) -> None:
    teams = json.loads((fixture_dir / "teams.json").read_text(encoding="utf-8"))
    lines = ["team,rating,tempo"]
    for idx, team in enumerate(sorted(teams, key=lambda row: row["team_id"]), start=1):
        lines.append(f"{team['team_id']},{200 - idx:.3f},{64 + (idx % 5):.1f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _editable_bracket_payload(synthetic_input_dir: Path) -> list[dict[str, object]]:
    entries = json.loads((synthetic_input_dir / "entries.json").read_text(encoding="utf-8"))
    return [
        {
            "game_id": game_id,
            "winner_team_id": winner_team_id,
            "locked": False,
        }
        for game_id, winner_team_id in sorted(entries[0]["picks"].items())
    ]
