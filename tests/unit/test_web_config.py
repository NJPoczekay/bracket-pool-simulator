from __future__ import annotations

from pathlib import Path

import pytest

from bracket_sim.infrastructure.web.config import load_pool_registry


def test_load_pool_registry_resolves_relative_paths_and_schedule(tmp_path: Path) -> None:
    config_path = tmp_path / "pools.toml"
    config_path.write_text(
        """
[[pools]]
id = "main"
name = "Main Pool"
group_url = "https://fantasy.espn.com/games/mock-challenge-2026/group?id=main"
raw_dir = "data/raw/main"
prepared_dir = "data/prepared/main"
reports_root = "reports/main"
ratings_file = "ratings/main.csv"
use_kenpom = false
min_usable_entries = 2
n_sims = 5000
seed = 11
batch_size = 1000
engine = "numpy"

[pools.schedule]
enabled = true
daily_time = "08:15:00"
timezone = "America/Los_Angeles"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    registry = load_pool_registry(config_path)

    assert len(registry.pools) == 1
    pool = registry.pools[0]
    assert pool.raw_dir == (tmp_path / "data/raw/main").resolve()
    assert pool.prepared_dir == (tmp_path / "data/prepared/main").resolve()
    assert pool.reports_root == (tmp_path / "reports/main").resolve()
    assert pool.ratings_file == (tmp_path / "ratings/main.csv").resolve()
    assert pool.schedule is not None
    assert pool.schedule.enabled is True
    assert pool.schedule.timezone == "America/Los_Angeles"


def test_load_pool_registry_defaults_tracker_dirs_from_pool_id(tmp_path: Path) -> None:
    config_path = tmp_path / "pools.toml"
    config_path.write_text(
        """
[[pools]]
id = "Main Pool"
name = "Main Pool"
group_url = "https://fantasy.espn.com/games/tournament-challenge-bracket-2026/group?id=main"
n_sims = 5000
seed = 11
engine = "numpy"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    registry = load_pool_registry(config_path)

    pool = registry.pools[0]
    assert pool.raw_dir == (tmp_path / "data/2026/tracker/main-pool/raw").resolve()
    assert pool.prepared_dir == (tmp_path / "data/2026/tracker/main-pool/prepared").resolve()
    assert pool.reports_root == (tmp_path / "reports/2026/tracker/main-pool").resolve()


def test_load_pool_registry_rejects_unknown_timezone(tmp_path: Path) -> None:
    config_path = tmp_path / "pools.toml"
    config_path.write_text(
        """
[[pools]]
id = "main"
name = "Main Pool"
group_url = "https://fantasy.espn.com/games/mock-challenge-2026/group?id=main"
raw_dir = "raw"
prepared_dir = "prepared"
reports_root = "reports"
n_sims = 1000
seed = 7
engine = "numpy"

[pools.schedule]
enabled = true
daily_time = "09:00:00"
timezone = "Mars/Olympus"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown timezone"):
        load_pool_registry(config_path)
