from __future__ import annotations

from fastapi.testclient import TestClient

from bracket_sim.infrastructure.web.main import create_app


def test_health_endpoint_returns_status_and_version() -> None:
    client = TestClient(create_app())

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"]


def test_foundation_endpoint_exposes_phase_zero_contracts() -> None:
    client = TestClient(create_app())

    response = client.get("/api/foundation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["roadmap_phase"] == "integrated_bracket_lab_and_tracker"
    assert [workflow["key"] for workflow in payload["workflows"]] == [
        "bracket_lab",
        "pool_tracker",
    ]
    assert payload["workflows"][0]["state"] == "planned"
    assert payload["workflows"][1]["state"] == "setup_required"
    assert any(
        system["key"] == "1-2-4-8-16-32" and system["implemented"] is True
        for system in payload["scoring_systems"]
    )
    assert any(
        mode["mode"] == "manual" and mode["implemented"] is True
        for mode in payload["completion_modes"]
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
    assert "Tracker Setup Required" in root_response.text
    assert "config/pools.example.toml" in root_response.text
    assert pools_response.status_code == 200
    assert pools_response.json() == {"pools": []}
