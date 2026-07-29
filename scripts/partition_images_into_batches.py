#!/usr/bin/env python3
"""Partition all images from a source tree into numbered flat batch directories.

Scans a nested source tree, sorts all images by name for determinism, then
distributes them across numbered subdirectories (batch_000, batch_001, ...)
with at most --batch-size images each.

Each batch directory has the same layout as sample_unseen_images.py output:
  <output-root>/batch_000/images/   ← flat images (symlinks/hardlinks/copies)
  <output-root>/batch_001/images/
  ...

A top-level manifest is written to <output-root>/batch_manifest.jsonl.

ENVIRONMENT: Run in the weather-doc-extractor conda environment:
  conda activate weather-doc-extractor
  python scripts/partition_images_into_batches.py [options]

Example:
  python scripts/partition_images_into_batches.py \\
      --source-root /data/scratch/philip.brohan/documents/Daily_Rainfall_UK/jpgs_25pc_filtered \\
      --output-root /data/scratch/philip.brohan/documents/Daily_Rainfall_UK/batches \\
      --batch-size 50000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


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


def _materialize(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition all images from a source tree into flat batch directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 2)[-1],
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/data/scratch/philip.brohan/documents/Daily_Rainfall_UK/jpgs_25pc_filtered"
        ),
        help="Root of nested image tree to scan (default: jpgs_25pc_filtered)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root; batch subdirectories are created here",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Maximum images per batch directory (default: 50000)",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to materialize images in flat output dirs (default: symlink)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if not source_root.exists():
        raise SystemExit(f"Source root not found: {source_root}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    print(f"Scanning: {source_root}", flush=True)
    all_images = sorted(_iter_images(source_root), key=lambda p: p.name)
    total = len(all_images)
    if total == 0:
        raise SystemExit("No images found in source root")

    n_batches = (total + args.batch_size - 1) // args.batch_size
    width = len(str(n_batches - 1))

    print(f"Total images : {total}", flush=True)
    print(f"Batch size   : {args.batch_size}", flush=True)
    print(f"Batches      : {n_batches}", flush=True)
    print(flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    for batch_idx in range(n_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, total)
        batch_images = all_images[start:end]

        batch_name = f"batch_{batch_idx:0{width}d}"
        images_dir = output_root / batch_name / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        seen_names: set[str] = set()
        for src in batch_images:
            rel = src.relative_to(source_root)
            target_name = _safe_target_name(src, rel, seen_names)
            dst = images_dir / target_name
            _materialize(src, dst, args.link_mode)

        manifest_rows.append(
            {
                "batch": batch_name,
                "images_dir": str(images_dir),
                "count": len(batch_images),
                "start_index": start,
                "end_index": end - 1,
                "first_image": batch_images[0].name,
                "last_image": batch_images[-1].name,
            }
        )
        print(
            f"  {batch_name}: {len(batch_images)} images "
            f"({batch_images[0].name} … {batch_images[-1].name})",
            flush=True,
        )

    manifest_path = output_root / "batch_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")

    print(flush=True)
    print(f"Batch manifest: {manifest_path}", flush=True)
    print(f"Done. {n_batches} batch(es) in {output_root}", flush=True)


if __name__ == "__main__":
    main()
