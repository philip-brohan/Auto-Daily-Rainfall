#!/usr/bin/env python3
"""Build ensemble transcriptions from an extraction directory tree.

Recursively searches --input-dir for *.json extraction files, groups them by
stem (filename without extension), and writes one ensemble file per stem to
--output-dir.

Each output cell is:
  {"values": [<float|null|"missing">, ...]}

with one entry per extraction file found for that stem, in discovery order.
  - null means the extractor read the cell as blank (positive information)
  - "missing" means the file was found but could not be parsed

Typically, each stem appears in exactly N extraction files (one per model run),
so the values array has N entries without any bookkeeping about batches or models.

Usage:
    conda activate weather-doc-extractor
    python scripts/build_ensemble_transcriptions.py \
        --input-dir /path/to/individual_transcriptions \
        --output-dir /path/to/ensemble_transcriptions \
        [--precision 3] \
        [--summary-file /path/to/ensemble_summary.json]

    # Multiple input dirs are searched together:
    python scripts/build_ensemble_transcriptions.py \
        --input-dir outputs/extractions/run_a \
        --input-dir outputs/extractions/run_b \
        --output-dir outputs/ensemble/my_run

    # Or drive from a consensus_config.json (same format as
    # build_consensus_transcriptions.py):
    python scripts/build_ensemble_transcriptions.py \
        --config-file outputs/consensus_dataset_1000/consensus_config.json \
        --output-dir outputs/ensemble/my_run
"""

from __future__ import annotations

import argparse
import json
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


# Sentinel for a file that was found but could not be parsed.
# Distinct from None/null, which means the extractor read the cell as blank.
MISSING: str = "missing"


def _load_extraction(path: Path) -> dict[str, Any] | None:
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
    flat_row = grid.get(key)
    if isinstance(flat_row, list):
        return flat_row

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


def _discover_stem_files(input_dirs: list[Path]) -> dict[str, list[Path]]:
    """Recursively find all *.json files and group by stem."""
    stem_files: dict[str, list[Path]] = {}
    for d in input_dirs:
        for p in sorted(d.rglob("*.json")):
            stem_files.setdefault(p.stem, []).append(p)
    return stem_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ensemble transcriptions from an extraction directory tree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Path to consensus_config.json (reads extraction_dirs and precision from config)",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        dest="input_dirs",
        default=None,
        help="Directory to search recursively for extraction JSON files "
        "(repeat to include multiple directories). "
        "Required if --config-file not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for ensemble transcription JSON files. "
        "If --config-file provided and --output-dir omitted, writes to "
        "{config-dir}/ensemble_transcriptions/",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=None,
        help="Decimal precision for value normalization. "
        "If --config-file provided, defaults to config value; otherwise default is 3.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional path to write summary JSON. "
        "If --config-file provided and --summary-file omitted, writes to "
        "{config-dir}/ensemble_summary.json",
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

    # Resolve input directories
    if config_data is not None:
        input_dirs = [Path(d).resolve() for d in config_data.get("extraction_dirs", [])]
    elif args.input_dirs is not None:
        input_dirs = [Path(p).resolve() for p in args.input_dirs]
    else:
        raise SystemExit("Must provide either --config-file or --input-dir")

    if not input_dirs:
        raise SystemExit("No input directories specified")

    # Resolve output directory
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif config_data is not None:
        config_dir = args.config_file.resolve().parent
        output_dir = config_dir / "ensemble_transcriptions"
    else:
        raise SystemExit("Must provide --output-dir or use --config-file")

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
        summary_file = config_dir / "ensemble_summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    for d in input_dirs:
        if not d.exists():
            raise SystemExit(f"Input dir not found: {d}")

    print(
        f"Discovering extraction files in {[str(d) for d in input_dirs]} ...",
        flush=True,
    )
    stem_files = _discover_stem_files(input_dirs)
    if not stem_files:
        raise SystemExit("No extraction JSON files found in input directories")

    print(f"Found {len(stem_files)} stems.", flush=True)

    stats: dict[str, Any] = {
        "total_stems": len(stem_files),
        "total_cells": 0,
        "fully_present_cells": 0,
        "partial_cells": 0,
        "empty_cells": 0,
        "parse_failed_or_invalid": 0,
        "input_dirs": [str(d) for d in input_dirs],
        "precision": precision,
    }

    for stem, files in sorted(stem_files.items()):
        grids: list[dict[str, Any] | None] = []
        available: list[bool] = []
        for f in files:
            grid = _load_extraction(f)
            if grid is None:
                stats["parse_failed_or_invalid"] += 1
                available.append(False)
            else:
                available.append(True)
            grids.append(grid)

        n_found = len(grids)
        ensemble: dict[str, list[dict[str, Any]]] = {}

        for key in ALL_KEYS:
            month_entries = []
            for month_idx in range(12):
                values: list[float | None | str] = []
                for idx, grid in enumerate(grids):
                    if not available[idx]:
                        values.append(MISSING)
                        continue
                    row = _get_row(grid, key)
                    if not isinstance(row, list) or len(row) <= month_idx:
                        values.append(MISSING)
                        continue
                    values.append(_normalize_value(row[month_idx], precision))

                month_entries.append({"values": values})

                n_present = sum(1 for v in values if v != MISSING)
                stats["total_cells"] += 1
                if n_present == n_found:
                    stats["fully_present_cells"] += 1
                elif n_present > 0:
                    stats["partial_cells"] += 1
                else:
                    stats["empty_cells"] += 1

            ensemble[key] = month_entries

        out_file = output_dir / f"{stem}.json"
        out_file.write_text(json.dumps(ensemble, indent=2), encoding="utf-8")

    if stats["total_cells"] > 0:
        stats["fully_present_fraction"] = round(
            stats["fully_present_cells"] / stats["total_cells"], 6
        )
        stats["partial_fraction"] = round(
            stats["partial_cells"] / stats["total_cells"], 6
        )

    print(json.dumps(stats, indent=2), flush=True)
    if summary_file is not None:
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
