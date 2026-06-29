#!/usr/bin/env python3
"""Build consensus transcriptions (Option A format) from extraction directories.

Each output cell is:
  {"value": <float|null>, "correct": <bool>}

A cell is marked correct when either:
- at least `--agreement-threshold` models produce the same non-null value, or
- at least `--null-threshold` models produce null.

If both thresholds are satisfied, the non-null consensus takes precedence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

DAYS = [f"Day {i}" for i in range(1, 32)]
ALL_KEYS = DAYS + ["Totals"]


def _normalize_value(v: Any, precision: int) -> float | None:
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() == "null":
        return None
    try:
        return round(float(v), precision)
    except (TypeError, ValueError):
        return None


def _empty_consensus() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for key in ALL_KEYS:
        out[key] = [{"value": None, "correct": False} for _ in range(12)]
    return out


def _load_extraction(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if data.get("parse_failed"):
        return None
    grid = data.get("grid")
    if not isinstance(grid, dict):
        return None
    return grid


def _get_row(grid: dict[str, Any], key: str) -> list[Any] | None:
    """Return a 12-value row for ``key`` from either supported grid schema.

    Supports both:
    - flat schema: {"Day 1": [...], ..., "Totals": [...]}
    - nested schema: {"days": {"Day 1": [...]}, "totals": [...]}
    """
    # Flat schema support
    flat_row = grid.get(key)
    if isinstance(flat_row, list):
        return flat_row

    # Nested schema support used by extraction outputs
    if key == "Totals":
        totals = grid.get("totals")
        if isinstance(totals, list):
            return totals
    else:
        days = grid.get("days")
        if isinstance(days, dict):
            day_row = days.get(key)
            if isinstance(day_row, list):
                return day_row

    return None


def _collect_stems(input_dirs: list[Path]) -> set[str]:
    stems: set[str] = set()
    for d in input_dirs:
        for p in d.glob("*.json"):
            stems.add(p.stem)
    return stems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consensus transcriptions")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Path to consensus_config.json (reads extraction_dirs, threshold, precision from config)",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        dest="input_dirs",
        default=None,
        help="Model extraction directory (repeat 5 times). Required if --config-file not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for consensus transcription JSON files. "
        "If --config-file provided and --output-dir omitted, writes to {config-dir}/consensus_transcriptions/",
    )
    parser.add_argument(
        "--agreement-threshold",
        type=int,
        default=None,
        help="Minimum agreeing models to mark correct=true. "
        "If --config-file provided, defaults to config value; otherwise default is 2.",
    )
    parser.add_argument(
        "--null-threshold",
        type=int,
        default=None,
        help="Minimum agreeing models to mark correct=true when consensus value is null. "
        "Defaults to agreement threshold.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=None,
        help="Decimal precision for value normalization before voting. "
        "If --config-file provided, defaults to config value; otherwise default is 3.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional path to write summary JSON. "
        "If --config-file provided and --summary-file omitted, writes to {config-dir}/consensus_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config if provided
    config_data = None
    if args.config_file is not None:
        config_file = args.config_file.resolve()
        if not config_file.exists():
            raise SystemExit(f"Config file not found: {config_file}")
        try:
            config_data = json.loads(config_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SystemExit(f"Failed to parse config file: {config_file}: {e}")

    # Resolve input directories: from config or CLI
    if config_data is not None:
        input_dirs = [Path(d).resolve() for d in config_data.get("extraction_dirs", [])]
    elif args.input_dirs is not None:
        input_dirs = [Path(p).resolve() for p in args.input_dirs]
    else:
        raise SystemExit(
            "Must provide either --config-file or --input-dir (repeat 5 times)"
        )

    # Resolve output directory
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif config_data is not None:
        config_dir = args.config_file.resolve().parent
        output_dir = config_dir / "consensus_transcriptions"
    else:
        raise SystemExit("Must provide --output-dir or use --config-file")

    # Resolve agreement threshold
    threshold = args.agreement_threshold
    if threshold is None:
        if config_data is not None:
            threshold = config_data.get("agreement_threshold", 2)
        else:
            threshold = 2

    # Resolve null threshold
    null_threshold = args.null_threshold
    if null_threshold is None:
        if config_data is not None:
            null_threshold = config_data.get("null_threshold", threshold)
        else:
            null_threshold = threshold

    if threshold < 1:
        raise SystemExit("--agreement-threshold must be >= 1")
    if null_threshold < 1:
        raise SystemExit("--null-threshold must be >= 1")

    # Resolve precision
    precision = args.precision
    if precision is None:
        if config_data is not None:
            precision = config_data.get("precision", 3)
        else:
            precision = 3

    # Resolve summary file
    summary_file = args.summary_file
    if summary_file is None and config_data is not None:
        config_dir = args.config_file.resolve().parent
        summary_file = config_dir / "consensus_summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    for d in input_dirs:
        if not d.exists():
            raise SystemExit(f"Input dir not found: {d}")

    stems = sorted(_collect_stems(input_dirs))
    if not stems:
        raise SystemExit("No extraction JSON files found in input directories")

    stats = {
        "total_stems": len(stems),
        "total_cells": 0,
        "correct_cells": 0,
        "incorrect_cells": 0,
        "missing_model_files": 0,
        "parse_failed_or_invalid": 0,
        "input_dirs": [str(d) for d in input_dirs],
        "agreement_threshold": threshold,
        "null_threshold": null_threshold,
        "precision": precision,
    }

    for stem in stems:
        grids: list[dict[str, Any] | None] = []
        for d in input_dirs:
            p = d / f"{stem}.json"
            if not p.exists():
                stats["missing_model_files"] += 1
                grids.append(None)
                continue
            grid = _load_extraction(p)
            if grid is None:
                stats["parse_failed_or_invalid"] += 1
            grids.append(grid)

        consensus = _empty_consensus()

        for key in ALL_KEYS:
            for month_idx in range(12):
                votes: list[float | None] = []
                for grid in grids:
                    if grid is None:
                        continue
                    row = _get_row(grid, key)
                    if not isinstance(row, list) or len(row) <= month_idx:
                        continue
                    votes.append(_normalize_value(row[month_idx], precision))

                if not votes:
                    chosen = None
                    is_correct = False
                else:
                    counts = Counter(votes)
                    null_count = counts.get(None, 0)

                    non_null_counts = {v: c for v, c in counts.items() if v is not None}
                    if non_null_counts:
                        best_non_null_count = max(non_null_counts.values())
                        best_non_null_values = {
                            v
                            for v, c in non_null_counts.items()
                            if c == best_non_null_count
                        }
                        # Deterministic tie-break: first non-null value in encounter order.
                        best_non_null_value = next(
                            v
                            for v in votes
                            if v is not None and v in best_non_null_values
                        )
                    else:
                        best_non_null_count = 0
                        best_non_null_value = None

                    if best_non_null_count >= threshold:
                        chosen = best_non_null_value
                        is_correct = True
                    elif null_count >= null_threshold:
                        chosen = None
                        is_correct = True
                    else:
                        top_count = max(counts.values())
                        top_values = {v for v, c in counts.items() if c == top_count}
                        chosen = next(v for v in votes if v in top_values)
                        is_correct = False

                consensus[key][month_idx] = {"value": chosen, "correct": is_correct}

                stats["total_cells"] += 1
                if is_correct:
                    stats["correct_cells"] += 1
                else:
                    stats["incorrect_cells"] += 1

        out_file = output_dir / f"{stem}.json"
        out_file.write_text(json.dumps(consensus, indent=2), encoding="utf-8")

    coverage = 0.0
    if stats["total_cells"] > 0:
        coverage = stats["correct_cells"] / stats["total_cells"]
    stats["agreement_coverage"] = round(coverage, 6)

    print(json.dumps(stats, indent=2), flush=True)
    if summary_file is not None:
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
