#!/usr/bin/env python3
"""Generate validation plots for consensus transcriptions.

This script mirrors the original 2-panel visualize output by reusing
``weather_doc_extractor.visualize.make_figure``.

Panels are:
1. Source image.
2. Consensus output table.

The only intentional difference from the original is table text colour:
green for consensus-correct cells and red for consensus-incorrect cells.

Output files follow the existing validation naming convention:
``<stem>_comparison.png``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Make the src package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weather_doc_extractor.schemas import DailyRainfallGrid
from weather_doc_extractor.visualize import make_figure

_ROW_LABELS = [f"Day {i}" for i in range(1, 32)] + ["Totals"]
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate consensus validation figures (image + consensus table)"
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Path to consensus_config.json. If provided, derives dataset and consensus paths from config location.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/consensus_dataset_1000"),
        help="Root of consensus dataset (default: outputs/consensus_dataset_1000). "
        "Ignored if --config-file is provided.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing source images (default: <dataset-root>/images)",
    )
    parser.add_argument(
        "--consensus-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing consensus JSON files "
            "(default: <dataset-root>/consensus_transcriptions or <config-dir>/consensus_transcriptions)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output figures "
        "(default: <config-dir>/validation_figures if using config; "
        "otherwise outputs/validation/consensus/figures)",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Only process one stem (default: process all stems in consensus-dir)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of stems to process (after sorting)",
    )
    parser.add_argument(
        "--sample-denominator",
        type=int,
        default=1,
        help=(
            "Only process a deterministic 1/n sample of stems (default: 1, i.e. all stems). "
            "For a fixed n and input set, the selected subset is stable across runs."
        ),
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing ground-truth JSON transcriptions. "
            "When provided, figures use 4-category colouring (blue=consensus+agrees, "
            "red=consensus+disagrees, pale-blue=no-consensus+agrees, grey=no-consensus+disagrees) "
            "and summary stats are extended with per-category cell counts."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Numeric tolerance for comparing consensus and ground-truth values (default: 0.005)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG DPI (default: 150)",
    )
    return parser.parse_args()


def _load_consensus(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _consensus_to_grid(
    consensus: dict[str, Any],
) -> tuple[DailyRainfallGrid, list[list[bool]]]:
    days: dict[str, list[float | None]] = {}
    correct_mask: list[list[bool]] = []

    for row_label in _ROW_LABELS[:-1]:
        row_data = consensus.get(row_label, [])
        values: list[float | None] = []
        row_mask: list[bool] = []
        for month_idx in range(12):
            cell = (
                row_data[month_idx]
                if isinstance(row_data, list) and month_idx < len(row_data)
                else {}
            )
            if not isinstance(cell, dict):
                cell = {}
            values.append(_coerce_float(cell.get("value")))
            row_mask.append(bool(cell.get("correct", False)))
        days[row_label] = values
        correct_mask.append(row_mask)

    totals_data = consensus.get("Totals", [])
    totals: list[float | None] = []
    totals_mask: list[bool] = []
    for month_idx in range(12):
        cell = (
            totals_data[month_idx]
            if isinstance(totals_data, list) and month_idx < len(totals_data)
            else {}
        )
        if not isinstance(cell, dict):
            cell = {}
        totals.append(_coerce_float(cell.get("value")))
        totals_mask.append(bool(cell.get("correct", False)))
    correct_mask.append(totals_mask)

    return DailyRainfallGrid(days=days, totals=totals), correct_mask


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in _IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _get_table_artist(figure):
    # Axis 1 is the right-hand table panel in make_figure.
    if len(figure.axes) < 2:
        return None
    table_axis = figure.axes[1]
    for artist in table_axis.get_children():
        if hasattr(artist, "get_celld"):
            return artist
    return None


def _load_ground_truth_values(
    path: Path,
) -> dict[str, list[float | None]] | None:
    """Load a ground-truth transcription and return normalised float|None values.

    The ground-truth format uses string values ("null" or numeric strings) in a
    flat dict keyed by row label ("Day 1" … "Day 31", "Totals").  Missing rows
    are filled with 12 ``None`` values.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    result: dict[str, list[float | None]] = {}
    for key in _ROW_LABELS:
        row_data = data.get(key, [])
        values: list[float | None] = []
        for i in range(12):
            raw = (
                row_data[i]
                if isinstance(row_data, list) and i < len(row_data)
                else None
            )
            values.append(_coerce_float(raw))
        result[key] = values
    return result


def _build_agree_mask(
    grid: DailyRainfallGrid,
    gt_values: dict[str, list[float | None]],
    tolerance: float = 0.005,
) -> list[list[bool]]:
    """Return a 32×12 mask: True where the consensus value agrees with ground truth."""
    agree_mask: list[list[bool]] = []
    for row_label in _ROW_LABELS[:-1]:  # Day 1 … Day 31
        consensus_row = grid.days.get(row_label, [None] * 12)
        gt_row = gt_values.get(row_label, [None] * 12)
        row_agree: list[bool] = []
        for cv, gv in zip(consensus_row, gt_row):
            if cv is None and gv is None:
                row_agree.append(True)
            elif cv is None or gv is None:
                row_agree.append(False)
            else:
                row_agree.append(abs(cv - gv) <= tolerance)
        agree_mask.append(row_agree)
    # Totals row
    gt_totals = gt_values.get("Totals", [None] * 12)
    totals_agree: list[bool] = []
    for cv, gv in zip(grid.totals, gt_totals):
        if cv is None and gv is None:
            totals_agree.append(True)
        elif cv is None or gv is None:
            totals_agree.append(False)
        else:
            totals_agree.append(abs(cv - gv) <= tolerance)
    agree_mask.append(totals_agree)
    return agree_mask


def _colour_consensus_panel(figure, correct_mask: list[list[bool]]) -> bool:
    table_artist = _get_table_artist(figure)
    if table_artist is None:
        return False

    cells = table_artist.get_celld()
    for (row, col), cell in cells.items():
        if row <= 0 or col < 0:
            continue
        is_correct = correct_mask[row - 1][col]
        cell.get_text().set_color("blue" if is_correct else "#b2182b")
    return True


def _colour_consensus_panel_with_gt(
    figure,
    correct_mask: list[list[bool]],
    agree_mask: list[list[bool]],
) -> bool:
    """4-category colouring when ground truth is available.

    1. consensus + agrees with GT  → blue
    2. consensus + disagrees       → red (#b2182b)
    3. no consensus + agrees       → pale blue (#a6cee3)
    4. no consensus + disagrees    → grey (#aaaaaa)
    """
    table_artist = _get_table_artist(figure)
    if table_artist is None:
        return False
    cells = table_artist.get_celld()
    for (row, col), cell in cells.items():
        if row <= 0 or col < 0:
            continue
        has_consensus = correct_mask[row - 1][col]
        agrees = agree_mask[row - 1][col]
        if has_consensus and agrees:
            color = "blue"
        elif has_consensus and not agrees:
            color = "#b2182b"
        elif not has_consensus and agrees:
            color = "#a6cee3"
        else:
            color = "#aaaaaa"
        cell.get_text().set_color(color)
    return True


def _iter_stems(
    consensus_dir: Path,
    one_stem: str | None,
    limit: int | None,
    sample_denominator: int,
) -> list[str]:
    if one_stem is not None:
        stems = [one_stem]
    else:
        stems = sorted(path.stem for path in consensus_dir.glob("*.json"))

        if sample_denominator > 1:
            stems = [
                stem
                for stem in stems
                if int(hashlib.sha256(stem.encode("utf-8")).hexdigest(), 16)
                % sample_denominator
                == 0
            ]

            # Keep behavior predictable for very small datasets.
            if not stems:
                all_stems = sorted(path.stem for path in consensus_dir.glob("*.json"))
                if all_stems:
                    stems = [all_stems[0]]

    if limit is not None:
        stems = stems[:limit]
    return stems


def main() -> int:
    args = _parse_args()

    if args.sample_denominator < 1:
        print("Error: --sample-denominator must be >= 1", file=sys.stderr)
        return 1

    # Load config if provided
    config_data = None
    if args.config_file is not None:
        config_file = args.config_file.resolve()
        if not config_file.exists():
            print(f"Error: config file not found: {config_file}", file=sys.stderr)
            return 1
        try:
            config_data = json.loads(config_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(
                f"Error: failed to parse config file: {config_file}: {e}",
                file=sys.stderr,
            )
            return 1

    # Resolve paths
    if config_data is not None:
        config_dir = args.config_file.resolve().parent
        dataset_root = config_dir.parent
        images_dir = (
            args.images_dir if args.images_dir is not None else dataset_root / "images"
        ).resolve()
        consensus_dir = (
            args.consensus_dir
            if args.consensus_dir is not None
            else config_dir / "consensus_transcriptions"
        ).resolve()
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else config_dir / "validation_figures"
        ).resolve()
    else:
        dataset_root = args.dataset_root.resolve()
        images_dir = (
            args.images_dir if args.images_dir is not None else dataset_root / "images"
        ).resolve()
        consensus_dir = (
            args.consensus_dir
            if args.consensus_dir is not None
            else dataset_root / "consensus_transcriptions"
        ).resolve()
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else Path("outputs/validation/consensus/figures")
        ).resolve()

    if not images_dir.exists():
        print(f"Error: images directory not found: {images_dir}", file=sys.stderr)
        return 1
    if not consensus_dir.exists():
        print(f"Error: consensus directory not found: {consensus_dir}", file=sys.stderr)
        return 1

    stems = _iter_stems(
        consensus_dir,
        args.stem,
        args.limit,
        args.sample_denominator,
    )
    if not stems:
        print("No stems to process.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Resolve ground-truth directory
    gt_dir = (
        args.ground_truth_dir.resolve() if args.ground_truth_dir is not None else None
    )
    if gt_dir is not None and not gt_dir.exists():
        print(f"Error: ground-truth directory not found: {gt_dir}", file=sys.stderr)
        return 1

    # Stats for ground-truth comparison (only populated when gt_dir is set)
    gt_stats: dict[str, int] = {
        "stems_with_ground_truth": 0,
        "gt_consensus_agree_cells": 0,  # category 1: blue
        "gt_consensus_disagree_cells": 0,  # category 2: red
        "gt_no_consensus_agree_cells": 0,  # category 3: pale blue
        "gt_no_consensus_disagree_cells": 0,  # category 4: grey
    }

    saved = 0
    skipped = 0

    for stem in stems:
        consensus_path = consensus_dir / f"{stem}.json"
        consensus = _load_consensus(consensus_path)
        if consensus is None:
            print(f"Skipping {stem}: invalid consensus JSON ({consensus_path})")
            skipped += 1
            continue

        image_path = _find_image(images_dir, stem)
        if image_path is None:
            print(f"Skipping {stem}: image not found in {images_dir}")
            skipped += 1
            continue

        grid, correct_mask = _consensus_to_grid(consensus)
        title = stem

        fig = make_figure(
            image_path=image_path,
            grid=grid,
            title=title,
            model_name="Consensus",
        )

        # Apply colouring: 4-category when GT available, 2-category otherwise
        agree_mask: list[list[bool]] | None = None
        if gt_dir is not None:
            gt_path = gt_dir / f"{stem}.json"
            gt_values = _load_ground_truth_values(gt_path)
            if gt_values is not None:
                agree_mask = _build_agree_mask(grid, gt_values, args.tolerance)
                gt_stats["stems_with_ground_truth"] += 1
                for cor_row, agr_row in zip(correct_mask, agree_mask):
                    for has_c, agrees in zip(cor_row, agr_row):
                        if has_c and agrees:
                            gt_stats["gt_consensus_agree_cells"] += 1
                        elif has_c and not agrees:
                            gt_stats["gt_consensus_disagree_cells"] += 1
                        elif not has_c and agrees:
                            gt_stats["gt_no_consensus_agree_cells"] += 1
                        else:
                            gt_stats["gt_no_consensus_disagree_cells"] += 1

        if agree_mask is not None:
            if not _colour_consensus_panel_with_gt(fig, correct_mask, agree_mask):
                print(
                    f"Warning: could not locate consensus table artist for {stem}; using default colours"
                )
        else:
            if not _colour_consensus_panel(fig, correct_mask):
                print(
                    f"Warning: could not locate consensus table artist for {stem}; using default colours"
                )

        out_path = output_dir / f"{stem}_comparison.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        saved += 1
        print(f"Figure: {out_path}")

    print(f"Done. Saved {saved} figure(s); skipped {skipped}.")

    # Update consensus_summary.json with ground-truth comparison stats
    if gt_dir is not None and gt_stats["stems_with_ground_truth"] > 0:
        summary_path = consensus_dir.parent / "consensus_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = {}
        else:
            summary = {}
        summary.update(gt_stats)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Updated summary: {summary_path}")

    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
