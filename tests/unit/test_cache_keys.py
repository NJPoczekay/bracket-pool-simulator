from __future__ import annotations

import shutil
from pathlib import Path

from bracket_sim.domain.product_models import CompletionMode, PoolSettings, ScoringSystemKey
from bracket_sim.infrastructure.storage.cache_keys import (
    build_cache_key,
    capture_dataset_file_hashes,
    capture_dataset_hash,
)


def test_capture_dataset_file_hashes_and_hash_are_stable(synthetic_input_dir: Path) -> None:
    first_hash = capture_dataset_hash(synthetic_input_dir)
    second_hash = capture_dataset_hash(synthetic_input_dir)

    assert first_hash == second_hash
    assert set(capture_dataset_file_hashes(synthetic_input_dir)) == {
        "constraints.json",
        "entries.json",
        "games.json",
        "ratings.csv",
        "teams.json",
    }


def test_capture_dataset_hash_changes_when_dataset_changes(
    synthetic_input_dir: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "prepared_copy"
    shutil.copytree(synthetic_input_dir, copied)

    original_hash = capture_dataset_hash(copied)
    entries_path = copied / "entries.json"
    entries_path.write_text(entries_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert capture_dataset_hash(copied) != original_hash


def test_build_cache_key_is_stable_and_settings_sensitive() -> None:
    base_settings = PoolSettings(pool_size=25, scoring_system=ScoringSystemKey.ESPN)
    updated_settings = base_settings.model_copy(update={"scoring_system": ScoringSystemKey.LINEAR})
    dataset_hash = "a" * 64

    first = build_cache_key(
        artifact_kind="analysis",
        dataset_hash=dataset_hash,
        settings={
            "pool_settings": base_settings,
            "completion_mode": CompletionMode.MANUAL,
        },
    )
    second = build_cache_key(
        artifact_kind="analysis",
        dataset_hash=dataset_hash,
        settings={
            "pool_settings": base_settings,
            "completion_mode": CompletionMode.MANUAL,
        },
    )
    changed = build_cache_key(
        artifact_kind="analysis",
        dataset_hash=dataset_hash,
        settings={
            "pool_settings": updated_settings,
            "completion_mode": CompletionMode.MANUAL,
        },
    )

    assert first == second
    assert changed != first
    assert first.startswith("analysis-")


def test_build_cache_key_changes_for_round_of_64_scoring_systems() -> None:
    dataset_hash = "a" * 64
    round_of_64_flat = PoolSettings(
        pool_size=25,
        scoring_system=ScoringSystemKey.ROUND_OF_64_FLAT,
    )
    round_of_64_seed = round_of_64_flat.model_copy(
        update={"scoring_system": ScoringSystemKey.ROUND_OF_64_SEED}
    )

    flat_key = build_cache_key(
        artifact_kind="analysis",
        dataset_hash=dataset_hash,
        settings={
            "pool_settings": round_of_64_flat,
            "completion_mode": CompletionMode.MANUAL,
        },
    )
    seed_key = build_cache_key(
        artifact_kind="analysis",
        dataset_hash=dataset_hash,
        settings={
            "pool_settings": round_of_64_seed,
            "completion_mode": CompletionMode.MANUAL,
        },
    )

    assert flat_key != seed_key
