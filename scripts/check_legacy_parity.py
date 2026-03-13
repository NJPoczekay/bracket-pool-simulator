"""Compare current top entry win shares against a legacy baseline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from bracket_sim.application.simulate_pool import simulate_pool
from bracket_sim.domain.models import SimulationConfig, SimulationEntryResult


@dataclass(frozen=True)
class ParityDelta:
    """Top-entry win-share comparison row."""

    rank: int
    entry_id: str
    current_win_share: float
    legacy_win_share: float
    abs_delta: float


def _positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        msg = f"Expected a positive integer, got {raw_value}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _non_negative_float(raw_value: str) -> float:
    value = float(raw_value)
    if value < 0:
        msg = f"Expected a non-negative float, got {raw_value}"
        raise argparse.ArgumentTypeError(msg)
    return value


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("tests/fixtures/synthetic_64"),
        help="Directory containing normalized simulation inputs.",
    )
    parser.add_argument(
        "--legacy-json",
        type=Path,
        required=True,
        help="Path to legacy parity JSON with entry_win_shares mapping.",
    )
    parser.add_argument(
        "--n-sims",
        type=_positive_int,
        default=10_000,
        help="Number of simulations to run for current output.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for current output.",
    )
    parser.add_argument(
        "--top-n",
        type=_positive_int,
        default=10,
        help="Number of top entries to compare by current ranking.",
    )
    parser.add_argument(
        "--max-delta",
        type=_non_negative_float,
        default=0.02,
        help="Maximum allowed absolute win-share delta before failing.",
    )
    return parser.parse_args()


def load_legacy_win_shares(path: Path) -> dict[str, float]:
    """Load and validate expected legacy schema."""

    if not path.exists():
        msg = f"Legacy parity file does not exist: {path}"
        raise ValueError(msg)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Legacy parity file is not valid JSON: {path}"
        raise ValueError(msg) from exc

    if not isinstance(payload, dict):
        msg = "Legacy parity payload must be an object"
        raise ValueError(msg)

    raw_shares = payload.get("entry_win_shares")
    if not isinstance(raw_shares, dict):
        msg = "Legacy parity payload must include object field 'entry_win_shares'"
        raise ValueError(msg)

    shares: dict[str, float] = {}
    for raw_entry_id, raw_share in raw_shares.items():
        if not isinstance(raw_entry_id, str) or not raw_entry_id:
            msg = "Legacy parity entry ids must be non-empty strings"
            raise ValueError(msg)
        if not isinstance(raw_share, int | float):
            msg = f"Legacy win share for entry '{raw_entry_id}' must be numeric"
            raise ValueError(msg)
        shares[raw_entry_id] = float(raw_share)

    if not shares:
        msg = "Legacy parity entry_win_shares must not be empty"
        raise ValueError(msg)

    return shares


def compare_top_entries(
    current_entries: list[SimulationEntryResult],
    legacy_win_shares: dict[str, float],
    top_n: int,
) -> list[ParityDelta]:
    """Return absolute win-share deltas for top current entries."""

    if not current_entries:
        msg = "Current simulation produced no entries to compare"
        raise ValueError(msg)

    comparison_rows: list[ParityDelta] = []
    for rank, entry in enumerate(current_entries[: min(top_n, len(current_entries))], start=1):
        if entry.entry_id not in legacy_win_shares:
            msg = (
                "Legacy parity payload is missing win share for required top entry "
                f"'{entry.entry_id}'"
            )
            raise ValueError(msg)

        legacy_win_share = legacy_win_shares[entry.entry_id]
        abs_delta = abs(entry.win_share - legacy_win_share)
        comparison_rows.append(
            ParityDelta(
                rank=rank,
                entry_id=entry.entry_id,
                current_win_share=entry.win_share,
                legacy_win_share=legacy_win_share,
                abs_delta=abs_delta,
            )
        )

    return comparison_rows


def format_table(rows: list[ParityDelta]) -> str:
    """Render comparison rows for CLI output."""

    lines = [
        f"{'Rank':>4} {'Entry ID':<24} {'Current':>12} {'Legacy':>12} {'Abs Delta':>12}",
        f"{'-' * 4} {'-' * 24} {'-' * 12} {'-' * 12} {'-' * 12}",
    ]
    for row in rows:
        lines.append(
            f"{row.rank:>4} {row.entry_id:<24} {row.current_win_share:>12.6f} "
            f"{row.legacy_win_share:>12.6f} {row.abs_delta:>12.6f}"
        )
    return "\n".join(lines)


def run_parity_check(args: argparse.Namespace) -> int:
    """Execute parity comparison and return process exit code."""

    legacy_win_shares = load_legacy_win_shares(args.legacy_json)
    result = simulate_pool(
        SimulationConfig(
            input_dir=args.input_dir,
            n_sims=args.n_sims,
            seed=args.seed,
            rating_scale=10.0,
        )
    )
    rows = compare_top_entries(
        current_entries=result.entry_results,
        legacy_win_shares=legacy_win_shares,
        top_n=args.top_n,
    )

    print(
        "Compared "
        f"{len(rows)} entries with seed={args.seed}, "
        f"n_sims={args.n_sims}, max_delta={args.max_delta}"
    )
    print(format_table(rows))

    exceeding = [row for row in rows if row.abs_delta > args.max_delta]
    if exceeding:
        print(
            "\nFAIL: "
            f"{len(exceeding)} top entries exceeded max delta {args.max_delta:.6f}"
        )
        return 1

    print(f"\nPASS: all compared entries are within max delta {args.max_delta:.6f}")
    return 0


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    try:
        return run_parity_check(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR: failed reading parity inputs: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
