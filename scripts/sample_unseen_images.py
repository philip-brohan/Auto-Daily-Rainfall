#!/usr/bin/env python3
"""Sample unseen rainfall images recursively and materialize a flat working set.

This script scans a nested source tree (e.g. 660k images), picks a deterministic
random subset, and writes a flat output directory suitable for existing tooling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

_STEM_RE = re.compile(r"^DRain_(\d{4}-\d{4})_(.+)-(\d+)$")


def _iter_images(root: Path) -> list[Path]:
    paths: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}:
            paths.append(p)
    return paths


def _safe_target_name(src: Path, rel: Path, seen: set[str]) -> str:
    name = src.name
    if name not in seen:
        seen.add(name)
        return name

    # Handle rare basename collisions across the nested tree.
    digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:8]
    candidate = f"{src.stem}__{digest}{src.suffix.lower()}"
    idx = 2
    while candidate in seen:
        candidate = f"{src.stem}__{digest}_{idx}{src.suffix.lower()}"
        idx += 1
    seen.add(candidate)
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample unseen rainfall images")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/data/scratch/philip.brohan/documents/Daily_Rainfall_UK/jpgs_25pc_filtered"
        ),
        help="Root of nested image tree",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/consensus_dataset_1000"),
        help="Output root containing images/, transcriptions/, and manifest files",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of images to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to materialize selected images in flat output images/",
    )
    parser.add_argument(
        "--strict-stem",
        action="store_true",
        help="Only include files whose stem matches DRain naming pattern",
    )
    return parser.parse_args()


def _materialize(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if not source_root.exists():
        raise SystemExit(f"Source root not found: {source_root}")

    print(f"Scanning recursively: {source_root}", flush=True)
    all_images = _iter_images(source_root)
    if args.strict_stem:
        all_images = [p for p in all_images if _STEM_RE.match(p.stem)]

    if len(all_images) < args.count:
        raise SystemExit(
            f"Requested {args.count} images but found only {len(all_images)}"
        )

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(all_images, args.count), key=lambda p: p.name)

    images_dir = output_root / "images"
    transcriptions_dir = output_root / "transcriptions"
    images_dir.mkdir(parents=True, exist_ok=True)
    transcriptions_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = output_root / "sample_manifest.csv"
    manifest_jsonl = output_root / "sample_manifest.jsonl"

    seen_names: set[str] = set()
    rows: list[dict[str, str]] = []

    for src in selected:
        rel = src.relative_to(source_root)
        target_name = _safe_target_name(src, rel, seen_names)
        dst = images_dir / target_name
        _materialize(src, dst, args.link_mode)

        rows.append(
            {
                "stem": Path(target_name).stem,
                "source_path": str(src),
                "source_relative_path": str(rel),
                "selected_image": str(dst),
            }
        )

    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "source_path",
                "source_relative_path",
                "selected_image",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Found images: {len(all_images)}", flush=True)
    print(f"Sampled images: {len(rows)}", flush=True)
    print(f"Flat images dir: {images_dir}", flush=True)
    print(f"Manifest CSV: {manifest_csv}", flush=True)
    print(f"Manifest JSONL: {manifest_jsonl}", flush=True)


if __name__ == "__main__":
    main()
