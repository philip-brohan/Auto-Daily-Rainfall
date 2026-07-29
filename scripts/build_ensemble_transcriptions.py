#!/usr/bin/env python3
"""Build ensemble transcriptions from extraction directories.

Each output cell is:
  {"values": [<float|null|"missing">, ...]}

with one entry per input directory (in the order given), preserving each
model's raw value (normalized to the requested precision).
  - null means the model read the cell as blank (positive information)
  - "missing" means the model's extraction file was absent or unparseable

Usage:
    conda activate weather-doc-extractor
    python scripts/build_ensemble_transcriptions.py \
        --input-dir outputs/extractions/run_a \
        --input-dir outputs/extractions/run_b \
        --input-dir outputs/extractions/run_c \
        --output-dir outputs/ensemble/my_run \
        [--precision 3] \
        [--summary-file outputs/ensemble/my_run/ensemble_summary.json]

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


# Sentinel for a model whose extraction file was absent or failed to parse.
# Distinct from None/null, which means the model read the cell as blank.
MISSING: str = "missing"


def _empty_ensemble(n_models: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for key in ALL_KEYS:
        out[key] = [{"values": [MISSING] * n_models} for _ in range(12)]
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


def _collect_stems(input_dirs: list[Path]) -> set[str]:
    stems: set[str] = set()
    for d in input_dirs:
        for p in d.glob("*.json"):
            stems.add(p.stem)
    return stems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ensemble transcriptions from multiple extraction directories",
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
        help="Model extraction directory (repeat for each model). "
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

    # Resolve input directories: from config or CLI
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

    n_models = len(input_dirs)
    stems = sorted(_collect_stems(input_dirs))
    if not stems:
        raise SystemExit("No extraction JSON files found in input directories")

    stats = {
        "total_stems": len(stems),
        "total_cells": 0,
        "fully_present_cells": 0,
        "partial_cells": 0,
        "empty_cells": 0,
        "missing_model_files": 0,
        "parse_failed_or_invalid": 0,
        "n_models": n_models,
        "input_dirs": [str(d) for d in input_dirs],
        "precision": precision,
    }

    for stem in stems:
        # True None means the file existed but the grid couldn't be read;
        # we use the MISSING sentinel in output to distinguish from null (blank cell).
        grids: list[dict[str, Any] | None] = []
        available: list[bool] = []
        for d in input_dirs:
            p = d / f"{stem}.json"
            if not p.exists():
                stats["missing_model_files"] += 1
                grids.append(None)
                available.append(False)
                continue
            grid = _load_extraction(p)
            if grid is None:
                stats["parse_failed_or_invalid"] += 1
                available.append(False)
            else:
                available.append(True)
            grids.append(grid)

        ensemble = _empty_ensemble(n_models)

        for key in ALL_KEYS:
            for month_idx in range(12):
                values: list[float | None | str] = []
                for model_idx, grid in enumerate(grids):
                    if not available[model_idx]:
                        values.append(MISSING)
                        continue
                    row = _get_row(grid, key)
                    if not isinstance(row, list) or len(row) <= month_idx:
                        values.append(MISSING)
                        continue
                    values.append(_normalize_value(row[month_idx], precision))

                ensemble[key][month_idx] = {"values": values}

                n_present = sum(1 for v in values if v != MISSING)
                stats["total_cells"] += 1
                if n_present == n_models:
                    stats["fully_present_cells"] += 1
                elif n_present > 0:
                    stats["partial_cells"] += 1
                else:
                    stats["empty_cells"] += 1

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
