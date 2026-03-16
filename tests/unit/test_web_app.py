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
    assert payload["roadmap_phase"] == "phase_0_foundation"
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
    assert "Bracket tools, staged for the browser." in response.text
