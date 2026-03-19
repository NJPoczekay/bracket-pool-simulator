"""Simulation orchestration entrypoints."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import numpy.typing as npt

from bracket_sim.domain.bracket_graph import build_bracket_graph
from bracket_sim.domain.constraints import validate_constraints
from bracket_sim.domain.models import (
    PoolEntry,
    RunCheckpoint,
    RunManifest,
    SimulationConfig,
    SimulationEntryResult,
    SimulationResult,
    SimulationRunMetadata,
)
from bracket_sim.domain.scoring import (
    ESPN_ROUND_VALUES,
    aggregate_win_share_totals,
    build_predicted_wins_matrix,
    score_entries,
    validate_entries,
)
from bracket_sim.domain.simulator import simulate_tournament
from bracket_sim.infrastructure.observability.logging import configure_structured_logging
from bracket_sim.infrastructure.storage.normalized_loader import load_normalized_input
from bracket_sim.infrastructure.storage.run_artifacts import (
    RunArtifactPaths,
    build_run_artifact_paths,
    build_run_manifest,
    ensure_run_dir,
    generate_run_id,
    load_run_checkpoint,
    load_run_manifest,
    load_simulation_result,
    verify_run_manifest,
    write_run_checkpoint,
    write_run_manifest,
    write_simulation_result,
)


@dataclass
class _Accumulator:
    completed_sims: int
    completed_batches: int
    win_share_totals: npt.NDArray[np.float64]
    score_totals: npt.NDArray[np.int64]
    champion_counts: Counter[str]


def simulate_pool(config: SimulationConfig) -> SimulationResult:
    """Load validated inputs, run deterministic simulations, and aggregate entry odds."""

    normalized = load_normalized_input(config.input_dir)
    graph = build_bracket_graph(teams=normalized.teams, games=normalized.games)

    constraints_by_game_id = validate_constraints(
        constraints=normalized.constraints,
        graph=graph,
    )

    validate_entries(entries=normalized.entries, graph=graph)
    entry_ids, team_ids, predicted_wins = build_predicted_wins_matrix(
        entries=normalized.entries,
        graph=graph,
    )

    rating_records_by_team_id = {
        record.team_id: record for record in normalized.ratings.records
    }
    missing_team_ids = sorted(set(team_ids) - set(rating_records_by_team_id))
    if missing_team_ids:
        msg = f"Missing rating for team id(s): {missing_team_ids[:5]}"
        raise ValueError(msg)

    artifact_paths, manifest, accumulator, resumed_from_checkpoint = _prepare_run_state(
        config=config,
        entry_ids=entry_ids,
        team_ids=team_ids,
    )
    logger = configure_structured_logging(
        level=config.log_level,
        log_path=artifact_paths.log_path if artifact_paths is not None else None,
    ).bind(
        run_id=manifest.run_id,
        seed=config.seed,
        n_sims=config.n_sims,
        batch_size=config.effective_batch_size,
        engine=config.engine,
    )

    if (
        artifact_paths is not None
        and config.resume
        and accumulator.completed_sims == config.n_sims
        and artifact_paths.result_path.exists()
    ):
        logger.info("simulation_run_loaded", completed_sims=accumulator.completed_sims)
        return load_simulation_result(artifact_paths.result_path)

    total_batches = math.ceil(config.n_sims / config.effective_batch_size)
    started_at = perf_counter()
    logger.info(
        "simulation_started",
        resumed_from_checkpoint=resumed_from_checkpoint,
        completed_sims=accumulator.completed_sims,
        total_batches=total_batches,
    )

    for batch_index in range(accumulator.completed_batches, total_batches):
        remaining = config.n_sims - accumulator.completed_sims
        if remaining <= 0:
            break

        batch_n_sims = min(config.effective_batch_size, remaining)
        batch_seed = _derive_batch_seed(
            seed=config.seed,
            batch_index=batch_index,
            total_batches=total_batches,
        )
        logger.info(
            "batch_started",
            batch_index=batch_index + 1,
            total_batches=total_batches,
            batch_n_sims=batch_n_sims,
            batch_seed=batch_seed,
        )

        simulation = simulate_tournament(
            graph=graph,
            rating_records_by_team_id=rating_records_by_team_id,
            constraints_by_game_id=constraints_by_game_id,
            n_sims=batch_n_sims,
            seed=batch_seed,
            point_spread_std_dev=config.rating_scale,
            engine=config.engine,
        )

        scores = score_entries(
            predicted_wins=predicted_wins,
            actual_wins=simulation.team_wins,
            round_values=ESPN_ROUND_VALUES,
        )
        accumulator.win_share_totals += aggregate_win_share_totals(scores)
        accumulator.score_totals += np.sum(scores, axis=1, dtype=np.int64)
        accumulator.completed_sims += batch_n_sims
        accumulator.completed_batches += 1
        _merge_champion_counts(
            champion_counts=accumulator.champion_counts,
            team_ids=simulation.team_ids,
            champions=simulation.champions,
        )

        if artifact_paths is not None:
            write_run_checkpoint(
                artifact_paths.checkpoint_path,
                _build_checkpoint(run_id=manifest.run_id, accumulator=accumulator),
            )

        logger.info(
            "batch_completed",
            batch_index=batch_index + 1,
            total_batches=total_batches,
            completed_sims=accumulator.completed_sims,
        )

    result = _build_result(
        config=config,
        entries=normalized.entries,
        accumulator=accumulator,
        run_id=manifest.run_id,
        artifact_paths=artifact_paths,
        resumed_from_checkpoint=resumed_from_checkpoint,
    )

    if artifact_paths is not None:
        write_run_checkpoint(
            artifact_paths.checkpoint_path,
            _build_checkpoint(run_id=manifest.run_id, accumulator=accumulator),
        )
        write_simulation_result(artifact_paths.result_path, result)

    logger.info(
        "simulation_completed",
        elapsed_ms=round((perf_counter() - started_at) * 1000, 3),
        completed_sims=accumulator.completed_sims,
    )
    return result


def _prepare_run_state(
    *,
    config: SimulationConfig,
    entry_ids: list[str],
    team_ids: list[str],
) -> tuple[RunArtifactPaths | None, RunManifest, _Accumulator, bool]:
    if config.run_dir is None:
        manifest = build_run_manifest(
            config=config,
            run_id=generate_run_id(config=config),
            entry_ids=entry_ids,
            team_ids=team_ids,
        )
        return None, manifest, _empty_accumulator(len(entry_ids)), False

    ensure_run_dir(config.run_dir)
    artifact_paths = build_run_artifact_paths(config.run_dir)

    if config.resume:
        manifest = load_run_manifest(artifact_paths.manifest_path)
        verify_run_manifest(
            manifest=manifest,
            config=config,
            entry_ids=entry_ids,
            team_ids=team_ids,
        )
        checkpoint = load_run_checkpoint(artifact_paths.checkpoint_path)
        if checkpoint.run_id != manifest.run_id:
            msg = "Run checkpoint does not match run manifest"
            raise ValueError(msg)
        accumulator = _accumulator_from_checkpoint(
            checkpoint=checkpoint,
            n_entries=len(entry_ids),
            valid_team_ids=set(team_ids),
            expected_n_sims=config.n_sims,
        )
        return artifact_paths, manifest, accumulator, accumulator.completed_sims > 0

    _ensure_fresh_run_dir(artifact_paths)
    manifest = build_run_manifest(
        config=config,
        run_id=generate_run_id(config=config),
        entry_ids=entry_ids,
        team_ids=team_ids,
    )
    write_run_manifest(artifact_paths.manifest_path, manifest)

    accumulator = _empty_accumulator(len(entry_ids))
    write_run_checkpoint(
        artifact_paths.checkpoint_path,
        _build_checkpoint(run_id=manifest.run_id, accumulator=accumulator),
    )
    return artifact_paths, manifest, accumulator, False


def _ensure_fresh_run_dir(artifact_paths: RunArtifactPaths) -> None:
    existing = [
        path
        for path in (
            artifact_paths.manifest_path,
            artifact_paths.checkpoint_path,
            artifact_paths.result_path,
            artifact_paths.log_path,
        )
        if path.exists()
    ]
    if existing:
        msg = (
            "Run directory already contains artifacts; use --resume or choose a new --run-dir: "
            f"{existing[0].parent}"
        )
        raise ValueError(msg)


def _empty_accumulator(n_entries: int) -> _Accumulator:
    return _Accumulator(
        completed_sims=0,
        completed_batches=0,
        win_share_totals=np.zeros(n_entries, dtype=np.float64),
        score_totals=np.zeros(n_entries, dtype=np.int64),
        champion_counts=Counter(),
    )


def _accumulator_from_checkpoint(
    *,
    checkpoint: RunCheckpoint,
    n_entries: int,
    valid_team_ids: set[str],
    expected_n_sims: int,
) -> _Accumulator:
    if len(checkpoint.win_share_totals) != n_entries or len(checkpoint.score_totals) != n_entries:
        msg = "Run checkpoint does not match the expected entry layout"
        raise ValueError(msg)
    if checkpoint.completed_sims > expected_n_sims:
        msg = "Run checkpoint exceeds the configured simulation count"
        raise ValueError(msg)
    unknown_team_ids = sorted(set(checkpoint.champion_counts) - valid_team_ids)
    if unknown_team_ids:
        msg = f"Run checkpoint contains unknown champion team ids: {unknown_team_ids[:5]}"
        raise ValueError(msg)

    return _Accumulator(
        completed_sims=checkpoint.completed_sims,
        completed_batches=checkpoint.completed_batches,
        win_share_totals=np.asarray(checkpoint.win_share_totals, dtype=np.float64),
        score_totals=np.asarray(checkpoint.score_totals, dtype=np.int64),
        champion_counts=Counter(checkpoint.champion_counts),
    )


def _build_checkpoint(*, run_id: str, accumulator: _Accumulator) -> RunCheckpoint:
    return RunCheckpoint(
        run_id=run_id,
        completed_sims=accumulator.completed_sims,
        completed_batches=accumulator.completed_batches,
        win_share_totals=accumulator.win_share_totals.tolist(),
        score_totals=[int(total) for total in accumulator.score_totals.tolist()],
        champion_counts={
            team_id: int(count)
            for team_id, count in sorted(
                accumulator.champion_counts.items(),
                key=lambda item: item[0],
            )
        },
    )


def _merge_champion_counts(
    *,
    champion_counts: Counter[str],
    team_ids: list[str],
    champions: npt.NDArray[np.int16],
) -> None:
    batch_counter = Counter(champions.tolist())
    for team_idx, count in batch_counter.items():
        champion_counts[team_ids[team_idx]] += int(count)


def _build_result(
    *,
    config: SimulationConfig,
    entries: Sequence[PoolEntry],
    accumulator: _Accumulator,
    run_id: str,
    artifact_paths: RunArtifactPaths | None,
    resumed_from_checkpoint: bool,
) -> SimulationResult:
    entry_results: list[SimulationEntryResult] = []
    for entry_idx, entry in enumerate(entries):
        entry_results.append(
            SimulationEntryResult(
                entry_id=entry.entry_id,
                entry_name=entry.entry_name,
                win_share=float(accumulator.win_share_totals[entry_idx] / config.n_sims),
                average_score=float(accumulator.score_totals[entry_idx] / config.n_sims),
            )
        )

    entry_results.sort(
        key=lambda result: (result.win_share, result.average_score, result.entry_id),
        reverse=True,
    )

    champion_counts = {
        team_id: int(count)
        for team_id, count in sorted(
            accumulator.champion_counts.items(),
            key=lambda item: item[0],
        )
    }

    return SimulationResult(
        n_sims=config.n_sims,
        seed=config.seed,
        entry_results=entry_results,
        champion_counts=champion_counts,
        run_metadata=SimulationRunMetadata(
            run_id=run_id,
            engine=config.engine,
            batch_size=config.effective_batch_size,
            batches_completed=accumulator.completed_batches,
            resumed_from_checkpoint=resumed_from_checkpoint,
            run_dir=artifact_paths.run_dir if artifact_paths is not None else None,
            manifest_path=artifact_paths.manifest_path if artifact_paths is not None else None,
            checkpoint_path=artifact_paths.checkpoint_path if artifact_paths is not None else None,
            result_path=artifact_paths.result_path if artifact_paths is not None else None,
            log_path=artifact_paths.log_path if artifact_paths is not None else None,
        ),
    )


def _derive_batch_seed(*, seed: int, batch_index: int, total_batches: int) -> int:
    if total_batches == 1 and batch_index == 0:
        return seed

    sequence = np.random.SeedSequence([seed, batch_index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])
