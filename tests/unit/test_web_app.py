from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from bracket_sim.infrastructure.web.main import create_app


def test_health_endpoint_returns_status_and_version() -> None:
    client = TestClient(create_app())

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"]


def test_foundation_endpoint_exposes_analyzer_mvp_contracts() -> None:
    client = TestClient(create_app())

    response = client.get("/api/foundation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["roadmap_phase"] == "phase_2_analyzer_mvp"
    assert [workflow["key"] for workflow in payload["workflows"]] == [
        "bracket_lab",
        "pool_tracker",
    ]
    assert payload["workflows"][0]["state"] == "setup_required"
    assert payload["workflows"][1]["state"] == "setup_required"
    assert all(
        system["implemented"] is True
        for system in payload["scoring_systems"]
    )
    assert any(
        mode["mode"] == "manual" and mode["implemented"] is True
        for mode in payload["completion_modes"]
    )
    model_rank_mode = next(
        mode for mode in payload["completion_modes"] if mode["mode"] == "internal_model_rank"
    )
    assert model_rank_mode["alias_of"] == "kenpom"
    assert all(
        mode["mode"] not in {"ap_poll", "ncaa_net"} for mode in payload["completion_modes"]
    )


def test_cache_key_preview_endpoint_uses_shared_rules() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/cache-key",
        json={
            "artifact_kind": "analysis",
            "dataset_hash": "a" * 64,
            "pool_settings": {
                "pool_size": 42,
                "scoring_system": "1-2-4-8-16-32",
            },
            "completion_mode": "manual",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_kind"] == "analysis"
    assert payload["cache_key"].startswith("analysis-")


def test_root_serves_frontend_shell() -> None:
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "Build Brackets, then Track Pool Odds." in response.text
    assert "Bracket Lab" in response.text
    assert "Pool Tracker" in response.text


def test_root_and_pools_api_show_tracker_setup_state_without_config() -> None:
    client = TestClient(create_app())

    root_response = client.get("/")
    pools_response = client.get("/api/pools")

    assert root_response.status_code == 200
    assert "Bracket Lab Setup Required" in root_response.text
    assert "Tracker Setup Required" in root_response.text
    assert "config/pools.example.toml" in root_response.text
    assert pools_response.status_code == 200
    assert pools_response.json() == {"pools": []}


def test_bracket_lab_api_requires_configured_dataset() -> None:
    client = TestClient(create_app())

    response = client.get("/api/bracket-lab/bootstrap")

    assert response.status_code == 503
    assert response.json()["detail"] == "Bracket Lab is not configured"


def test_bracket_lab_bootstrap_and_analyze_endpoints(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    client = TestClient(
        create_app(
            bracket_lab_input=prepared_bracket_lab_dir,
            enable_scheduler=False,
        )
    )

    bootstrap_response = client.get("/api/bracket-lab/bootstrap")
    analyze_response = client.post(
        "/api/bracket-lab/analyze",
        json={
            "bracket": {
                "picks": _editable_bracket_payload(synthetic_input_dir),
            },
            "pool_settings": {
                "pool_size": 18,
                "scoring_system": "2-3-5-8-13-21",
            },
            "completion_mode": "manual",
        },
    )

    assert bootstrap_response.status_code == 200
    bootstrap = bootstrap_response.json()
    assert bootstrap["completion_mode"] == "manual"
    assert len(bootstrap["teams"]) == 64
    assert len(bootstrap["games"]) == 63

    assert analyze_response.status_code == 200
    analysis = analyze_response.json()
    assert analysis["public_percentile"] is None
    assert analysis["pool_settings"]["scoring_system"] == "2-3-5-8-13-21"
    assert len(analysis["pick_diagnostics"]) == 63
    assert analysis["cache_key"].startswith("analysis-")


def test_root_renders_empty_start_bracket_editor_when_bracket_lab_is_configured(
    prepared_bracket_lab_dir: Path,
) -> None:
    client = TestClient(
        create_app(
            bracket_lab_input=prepared_bracket_lab_dir,
            enable_scheduler=False,
        )
    )

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="bracket-lab-editor-layout"' in response.text
    assert 'id="bracket-lab-desktop"' in response.text
    assert 'id="bracket-mobile-tabs"' in response.text
    assert "63 picks remaining before analysis." in response.text
    assert 'id="analyze-bracket-button" type="button" disabled' in response.text


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
