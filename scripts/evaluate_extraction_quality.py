#!/usr/bin/env python3
"""Evaluate per-extraction cell accuracy against ground truth.

ENVIRONMENT: Run this script in the weather-doc-extractor conda environment:
  conda activate weather-doc-extractor
  python scripts/evaluate_extraction_quality.py \
    --ground-truth-dir test_data/fake/transcriptions \
    --extraction-dir outputs/extractions/20260618-161908 --label SmolVLM

The script compares extracted grids to ground truth on a cell-by-cell basis and
reports summary accuracy statistics for each extraction directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROW_LABELS = [f"Day {i}" for i in range(1, 32)] + ["Totals"]
MONTHS = 12
CELLS_PER_STEM = len(ROW_LABELS) * MONTHS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare extracted JSON outputs against ground truth and report per-extraction cell accuracy."
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        required=True,
        help="Directory containing ground-truth transcription JSON files",
    )
    parser.add_argument(
        "--extraction-dir",
        type=Path,
        action="append",
        required=True,
        help="Extraction directory containing one JSON per stem (repeatable)",
    )
    parser.add_argument(
        "--label",
        type=str,
        action="append",
        default=None,
        help="Optional label per extraction dir (repeatable, same order as --extraction-dir)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Numeric tolerance for float comparison (default: 0.005)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for JSON summary",
    )
    return parser.parse_args()


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_ground_truth(path: Path) -> dict[str, list[float | None]] | None:
    data = _load_json(path)
    if data is None:
        return None

    rows: dict[str, list[float | None]] = {}
    for row_label in ROW_LABELS:
        row = data.get(row_label, [])
        values: list[float | None] = []
        for i in range(MONTHS):
            raw = row[i] if isinstance(row, list) and i < len(row) else None
            values.append(_coerce_float(raw))
        rows[row_label] = values
    return rows


def _load_extraction(path: Path) -> dict[str, list[float | None]] | None:
    data = _load_json(path)
    if data is None:
        return None

    if bool(data.get("parse_failed", False)):
        return None

    grid = data.get("grid")
    if not isinstance(grid, dict):
        return None

    days = grid.get("days")
    totals = grid.get("totals")
    if not isinstance(days, dict) or not isinstance(totals, list):
        return None

    rows: dict[str, list[float | None]] = {}
    for row_label in ROW_LABELS[:-1]:
        row = days.get(row_label, [])
        values: list[float | None] = []
        for i in range(MONTHS):
            raw = row[i] if isinstance(row, list) and i < len(row) else None
            values.append(_coerce_float(raw))
        rows[row_label] = values

    totals_vals: list[float | None] = []
    for i in range(MONTHS):
        raw = totals[i] if i < len(totals) else None
        totals_vals.append(_coerce_float(raw))
    rows["Totals"] = totals_vals

    return rows


def _cell_equal(a: float | None, b: float | None, tolerance: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tolerance


def _score_extraction(
    extraction_dir: Path,
    label: str,
    ground_truth_dir: Path,
    tolerance: float,
) -> dict[str, Any]:
    gt_paths = sorted(ground_truth_dir.glob("*.json"))

    total_stems = len(gt_paths)
    total_gt_cells = total_stems * CELLS_PER_STEM

    missing_files = 0
    invalid_or_parse_failed = 0
    evaluated_stems = 0

    evaluated_cells = 0
    correct_cells = 0

    for gt_path in gt_paths:
        stem = gt_path.stem
        gt_rows = _load_ground_truth(gt_path)
        if gt_rows is None:
            # Ground truth should be valid; skip this stem entirely if corrupted.
            total_stems -= 1
            total_gt_cells -= CELLS_PER_STEM
            continue

        pred_path = extraction_dir / f"{stem}.json"
        if not pred_path.exists():
            missing_files += 1
            continue

        pred_rows = _load_extraction(pred_path)
        if pred_rows is None:
            invalid_or_parse_failed += 1
            continue

        evaluated_stems += 1
        for row_label in ROW_LABELS:
            gt_vals = gt_rows[row_label]
            pred_vals = pred_rows[row_label]
            for gv, pv in zip(gt_vals, pred_vals):
                evaluated_cells += 1
                if _cell_equal(pv, gv, tolerance):
                    correct_cells += 1

    incorrect_cells = evaluated_cells - correct_cells

    def _safe_ratio(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    return {
        "label": label,
        "extraction_dir": str(extraction_dir),
        "total_stems": total_stems,
        "evaluated_stems": evaluated_stems,
        "missing_files": missing_files,
        "invalid_or_parse_failed": invalid_or_parse_failed,
        "evaluated_cells": evaluated_cells,
        "correct_cells": correct_cells,
        "incorrect_cells": incorrect_cells,
        "accuracy_on_evaluated_cells": round(
            _safe_ratio(correct_cells, evaluated_cells), 6
        ),
        "accuracy_vs_all_ground_truth_cells": round(
            _safe_ratio(correct_cells, total_gt_cells), 6
        ),
        "coverage_of_ground_truth_cells": round(
            _safe_ratio(evaluated_cells, total_gt_cells), 6
        ),
    }


def main() -> int:
    args = _parse_args()

    gt_dir = args.ground_truth_dir.resolve()
    if not gt_dir.exists():
        raise SystemExit(f"Ground-truth directory not found: {gt_dir}")

    extraction_dirs = [p.resolve() for p in args.extraction_dir]
    for p in extraction_dirs:
        if not p.exists():
            raise SystemExit(f"Extraction directory not found: {p}")

    labels = args.label or []
    if labels and len(labels) != len(extraction_dirs):
        raise SystemExit(
            "If --label is provided, it must be repeated once for each --extraction-dir"
        )
    if not labels:
        labels = [p.name for p in extraction_dirs]

    results = [
        _score_extraction(
            extraction_dir=ext_dir,
            label=label,
            ground_truth_dir=gt_dir,
            tolerance=args.tolerance,
        )
        for ext_dir, label in zip(extraction_dirs, labels)
    ]

    results_sorted = sorted(
        results,
        key=lambda r: r["accuracy_vs_all_ground_truth_cells"],
        reverse=True,
    )

    print("\nEXTRACTION QUALITY SUMMARY")
    print("=" * 120)
    header = (
        f"{'Label':<12} {'Acc(all)':>9} {'Acc(eval)':>9} {'Coverage':>9} "
        f"{'Correct':>8} {'Incorrect':>10} {'Cells(eval)':>11} {'Miss':>6} {'Bad':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results_sorted:
        print(
            f"{r['label']:<12} "
            f"{r['accuracy_vs_all_ground_truth_cells']:>9.4f} "
            f"{r['accuracy_on_evaluated_cells']:>9.4f} "
            f"{r['coverage_of_ground_truth_cells']:>9.4f} "
            f"{r['correct_cells']:>8d} "
            f"{r['incorrect_cells']:>10d} "
            f"{r['evaluated_cells']:>11d} "
            f"{r['missing_files']:>6d} "
            f"{r['invalid_or_parse_failed']:>6d}"
        )

    payload = {
        "ground_truth_dir": str(gt_dir),
        "tolerance": args.tolerance,
        "results": results_sorted,
    }

    if args.output_json is not None:
        out = args.output_json.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote summary JSON: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
