from __future__ import annotations

import json
from pathlib import Path

import pytest
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
    assert payload["roadmap_phase"] == "phase_4_optimizer_mvp"
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
    scoring_systems = {system["key"]: system for system in payload["scoring_systems"]}
    assert "round-of-64-flat" in scoring_systems
    assert "round-of-64-seed" in scoring_systems
    assert scoring_systems["round-of-64-flat"]["round_values"] == [1, 0, 0, 0, 0, 0]
    assert scoring_systems["round-of-64-seed"]["seed_bonus"] is True
    assert scoring_systems["round-of-64-seed"]["seed_bonus_rounds"] == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert any(
        mode["mode"] == "manual" and mode["implemented"] is True
        for mode in payload["completion_modes"]
    )
    model_rank_mode = next(
        mode for mode in payload["completion_modes"] if mode["mode"] == "internal_model_rank"
    )
    assert model_rank_mode["alias_of"] == "kenpom"
    assert model_rank_mode["base_mode"] is True
    pick_four_mode = next(
        mode for mode in payload["completion_modes"] if mode["mode"] == "pick_four"
    )
    assert pick_four_mode["helper_only"] is True
    assert pick_four_mode["requires_base_mode"] is True
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
    assert 'id="page-select"' in response.text
    assert 'id="optimizer-page"' in response.text
    assert 'id="pool-tracker-page"' in response.text
    assert "Build Brackets, then Track Pool Odds." not in response.text


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
    saved_brackets_response = client.get("/api/bracket-lab/saved-brackets")

    assert response.status_code == 503
    assert response.json()["detail"] == "Bracket Lab is not configured"
    assert saved_brackets_response.status_code == 503
    assert saved_brackets_response.json()["detail"] == "Bracket Lab is not configured"


def test_bracket_lab_bootstrap_analyze_and_optimize_endpoints(
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
    complete_response = client.post(
        "/api/bracket-lab/complete",
        json={
            "bracket": {
                "picks": _editable_bracket_payload(synthetic_input_dir)[:2],
            },
            "completion_mode": "kenpom",
        },
    )
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
    optimize_response = client.post(
        "/api/bracket-lab/optimize",
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
    assert len(bootstrap["initial_bracket"]["picks"]) == 63
    assert bootstrap["completion_inputs"]["available_modes"]
    assert len(bootstrap["teams"]) == 64
    assert len(bootstrap["games"]) == 63

    assert complete_response.status_code == 200
    completion = complete_response.json()
    assert completion["completion_mode"] == "kenpom"
    assert completion["state"] == "auto_completed"
    assert len(completion["completed_bracket"]["picks"]) == 63

    assert analyze_response.status_code == 200
    analysis = analyze_response.json()
    assert analysis["public_percentile"] is None
    assert analysis["pool_settings"]["scoring_system"] == "2-3-5-8-13-21"
    assert len(analysis["pick_diagnostics"]) == 63
    assert analysis["cache_key"].startswith("analysis-")

    assert optimize_response.status_code == 200
    optimization = optimize_response.json()
    assert optimization["cache_key"].startswith("optimization-")
    assert optimization["changed_pick_count"] == len(optimization["changed_picks"])
    assert optimization["summary"]
    assert len(optimization["alternatives"]) <= 3


def test_optimize_endpoint_rejects_incomplete_brackets(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    client = TestClient(
        create_app(
            bracket_lab_input=prepared_bracket_lab_dir,
            enable_scheduler=False,
        )
    )

    response = client.post(
        "/api/bracket-lab/optimize",
        json={
            "bracket": {
                "picks": _editable_bracket_payload(synthetic_input_dir)[:4],
            },
            "pool_settings": {
                "pool_size": 18,
                "scoring_system": "1-2-4-8-16-32",
            },
            "completion_mode": "manual",
        },
    )

    assert response.status_code == 400
    assert "Optimizer requires a complete bracket" in response.json()["detail"]


@pytest.mark.parametrize(
    "scoring_system",
    ["round-of-64-flat", "round-of-64-seed"],
)
def test_bracket_lab_analyze_endpoint_accepts_round_of_64_scoring_systems(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
    scoring_system: str,
) -> None:
    client = TestClient(
        create_app(
            bracket_lab_input=prepared_bracket_lab_dir,
            enable_scheduler=False,
        )
    )

    analyze_response = client.post(
        "/api/bracket-lab/analyze",
        json={
            "bracket": {
                "picks": _editable_bracket_payload(synthetic_input_dir),
            },
            "pool_settings": {
                "pool_size": 18,
                "scoring_system": scoring_system,
            },
            "completion_mode": "manual",
        },
    )

    assert analyze_response.status_code == 200
    analysis = analyze_response.json()
    assert analysis["pool_settings"]["scoring_system"] == scoring_system
    assert len(analysis["pick_diagnostics"]) == 63
    assert analysis["cache_key"].startswith("analysis-")


@pytest.mark.parametrize(
    "scoring_system",
    ["round-of-64-flat", "round-of-64-seed"],
)
def test_bracket_lab_optimize_endpoint_accepts_round_of_64_scoring_systems(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
    scoring_system: str,
) -> None:
    client = TestClient(
        create_app(
            bracket_lab_input=prepared_bracket_lab_dir,
            enable_scheduler=False,
        )
    )

    optimize_response = client.post(
        "/api/bracket-lab/optimize",
        json={
            "bracket": {
                "picks": _editable_bracket_payload(synthetic_input_dir),
            },
            "pool_settings": {
                "pool_size": 18,
                "scoring_system": scoring_system,
            },
            "completion_mode": "manual",
        },
    )

    assert optimize_response.status_code == 200
    optimization = optimize_response.json()
    assert optimization["pool_settings"]["scoring_system"] == scoring_system
    assert optimization["cache_key"].startswith("optimization-")
    assert optimization["changed_pick_count"] == len(optimization["changed_picks"])


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
    assert "Finish Picks" in response.text
    assert "Optimize Bracket" in response.text
    assert "Pick Four" in response.text
    assert "63 picks remaining before analysis. Manual picks stay locked." in response.text
    assert 'id="analyze-bracket-button" type="button" disabled' in response.text
    assert 'id="optimize-bracket-button" type="button" disabled' in response.text
    assert 'id="saved-bracket-select"' in response.text
    assert 'id="saved-bracket-name-input"' in response.text
    assert 'id="new-bracket-button"' in response.text
    assert 'id="load-bracket-button"' not in response.text
    assert 'id="optimizer-results"' in response.text


def test_saved_bracket_api_round_trip_persists_to_disk(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    store_dir = tmp_path / "saved-brackets"
    client = TestClient(
        create_app(
            bracket_lab_input=prepared_bracket_lab_dir,
            bracket_store_dir=store_dir,
            enable_scheduler=False,
        )
    )

    list_before = client.get("/api/bracket-lab/saved-brackets")
    save_response = client.post(
        "/api/bracket-lab/saved-brackets",
        json={
            "name": "My Test Bracket",
            "bracket": {
                "picks": _editable_bracket_payload(synthetic_input_dir),
            },
            "pool_settings": {
                "pool_size": 42,
                "scoring_system": "2-3-5-8-13-21",
            },
            "completion_mode": "manual",
        },
    )
    saved_bracket = save_response.json()
    list_after = client.get("/api/bracket-lab/saved-brackets")
    load_response = client.get(f"/api/bracket-lab/saved-brackets/{saved_bracket['bracket_id']}")

    assert list_before.status_code == 200
    assert list_before.json() == {"brackets": []}
    assert save_response.status_code == 200
    assert saved_bracket["name"] == "My Test Bracket"
    assert saved_bracket["pool_settings"]["pool_size"] == 42
    assert saved_bracket["pool_settings"]["scoring_system"] == "2-3-5-8-13-21"
    assert (store_dir / f"{saved_bracket['bracket_id']}.json").exists()
    assert list_after.status_code == 200
    assert len(list_after.json()["brackets"]) == 1
    assert list_after.json()["brackets"][0]["name"] == "My Test Bracket"
    assert list_after.json()["brackets"][0]["pool_settings"]["pool_size"] == 42
    assert load_response.status_code == 200
    assert load_response.json()["bracket"]["picks"][0]["game_id"]


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
