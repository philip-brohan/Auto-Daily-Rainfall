#!/usr/bin/env python
"""
Preprocess rainfall document images: trim whitespace and categorize by size.

Walks a directory tree, processes each image to remove padding, and saves
to output directories based on whether dimensions match the expected portrait
format.

Expected images: aspect ratio 0.6-0.85 (portrait orientation)
Irregular images: outside that range or too small

Supports cluster execution with parallelization and job sharding:
- Multi-process within a single job (--workers N)
- Distribute images across multiple cluster jobs (--shard INDEX TOTAL)

Usage (single machine):
    python scripts/preprocess_images.py \\
        --source /path/to/images \\
        --output-filtered /path/to/filtered \\
        --output-irregular /path/to/irregular \\
        [--workers 8] [--aspect-min 0.6] [--aspect-max 0.85] \\
        [--fuzz 5%] [--dry-run]

Usage (cluster - 6 jobs, 8 workers each):
    for i in {0..5}; do
        sbatch --job-name=preprocess_$i --time=6:00:00 \\
            scripts/preprocess_images.py \\
            --source /path/to/images \\
            --output-filtered /path/to/filtered \\
            --output-irregular /path/to/irregular \\
            --workers 8 --shard $i 6
    done
"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Optional, List
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_image_dimensions(image_path: Path) -> Optional[Tuple[int, int]]:
    """Get image dimensions using ImageMagick identify."""
    try:
        result = subprocess.run(
            ["identify", "-format", "%wx%h", str(image_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        dims = result.stdout.strip().split("x")
        return (int(dims[0]), int(dims[1]))
    except Exception as e:
        logger.error(f"Failed to get dimensions for {image_path}: {e}")
        return None


def trim_image(source_path: Path, dest_path: Path, fuzz: str = "5%") -> bool:
    """Trim whitespace from image and save to destination.

    Args:
        source_path: Input image path
        dest_path: Output image path
        fuzz: ImageMagick fuzz threshold (default "5%")

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Use convert to trim and save
        subprocess.run(
            [
                "convert",
                str(source_path),
                "-fuzz",
                fuzz,
                "-trim",
                "+repage",
                str(dest_path),
            ],
            check=True,
            capture_output=True,
        )
        return True
    except Exception as e:
        logger.error(f"Failed to trim {source_path}: {e}")
        return False


def is_expected_dimensions(
    width: int,
    height: int,
    aspect_min: float = 0.6,
    aspect_max: float = 0.85,
    min_width: int = 500,
    min_height: int = 800,
) -> bool:
    """Check if image dimensions match expected portrait format.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        aspect_min: Minimum acceptable aspect ratio (width/height)
        aspect_max: Maximum acceptable aspect ratio
        min_width: Minimum acceptable width in pixels
        min_height: Minimum acceptable height in pixels

    Returns:
        True if dimensions are within expected range
    """
    aspect = width / height
    return (
        aspect_min <= aspect <= aspect_max
        and width >= min_width
        and height >= min_height
    )


def process_image(
    source_path: Path,
    output_filtered: Path,
    output_irregular: Path,
    source_root: Path,
    aspect_min: float,
    aspect_max: float,
    min_width: int,
    min_height: int,
    fuzz: str,
    dry_run: bool = False,
) -> dict:
    """Process a single image.

    Args:
        source_path: Path to source image
        output_filtered: Root path for expected images
        output_irregular: Root path for irregular images
        source_root: Root path of source tree (for relative path calculation)
        aspect_min: Minimum acceptable aspect ratio
        aspect_max: Maximum acceptable aspect ratio
        min_width: Minimum acceptable width in pixels
        min_height: Minimum acceptable height in pixels
        fuzz: ImageMagick fuzz threshold
        dry_run: If True, don't actually write files

    Returns:
        Dictionary with processing results
    """
    result = {
        "path": str(source_path),
        "status": "unknown",
        "category": None,
        "dimensions": None,
        "aspect": None,
    }

    # Calculate relative path for output
    try:
        rel_path = source_path.relative_to(source_root)
    except ValueError:
        logger.error(f"Source {source_path} not under source root {source_root}")
        result["status"] = "error"
        return result

    # Determine output path
    if not dry_run:
        # Trim and save image
        dest_path = output_filtered / rel_path
        if not trim_image(source_path, dest_path, fuzz):
            result["status"] = "trim_failed"
            return result

        # Get dimensions of trimmed image
        dims = get_image_dimensions(dest_path)
        if not dims:
            result["status"] = "dimension_check_failed"
            return result

        width, height = dims
        aspect = width / height
        result["dimensions"] = dims
        result["aspect"] = aspect

        # Check if dimensions match expected
        is_expected = is_expected_dimensions(
            width,
            height,
            aspect_min,
            aspect_max,
            min_width,
            min_height,
        )

        if not is_expected:
            # Move to irregular directory
            irregular_dest = output_irregular / rel_path
            irregular_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest_path), str(irregular_dest))
            result["category"] = "irregular"
            logger.info(f"Irregular: {rel_path} ({width}x{height}, ar={aspect:.3f})")
        else:
            result["category"] = "filtered"
            logger.info(f"Filtered:  {rel_path} ({width}x{height}, ar={aspect:.3f})")

        result["status"] = "success"
    else:
        # Dry run: just report what would happen
        dims = get_image_dimensions(source_path)
        if dims:
            width, height = dims
            aspect = width / height
            result["dimensions"] = dims
            result["aspect"] = aspect
            is_expected = is_expected_dimensions(
                width,
                height,
                aspect_min,
                aspect_max,
                min_width,
                min_height,
            )
            result["category"] = "filtered" if is_expected else "irregular"
            result["status"] = "dry_run"
        else:
            result["status"] = "dimension_check_failed"

    return result


def _process_image_worker(args: Tuple) -> dict:
    """Wrapper for multiprocessing.Pool to unpack arguments."""
    return process_image(*args)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess rainfall document images: trim and categorize."
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Source directory tree containing images",
    )
    parser.add_argument(
        "--output-filtered",
        required=True,
        type=Path,
        help="Output directory for images with expected dimensions",
    )
    parser.add_argument(
        "--output-irregular",
        required=True,
        type=Path,
        help="Output directory for images with irregular dimensions",
    )
    parser.add_argument(
        "--aspect-min",
        type=float,
        default=0.6,
        help="Minimum acceptable aspect ratio (width/height) [default: 0.6]",
    )
    parser.add_argument(
        "--aspect-max",
        type=float,
        default=0.85,
        help="Maximum acceptable aspect ratio [default: 0.85]",
    )
    parser.add_argument(
        "--fuzz",
        type=str,
        default="5%",
        help="ImageMagick fuzz threshold for trim [default: 5%%]",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=500,
        help="Minimum acceptable image width in pixels [default: 500]",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=800,
        help="Minimum acceptable image height in pixels [default: 800]",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, use 0 for CPU count)",
    )
    parser.add_argument(
        "--shard",
        type=int,
        nargs=2,
        metavar=("INDEX", "TOTAL"),
        help="Process shard INDEX out of TOTAL shards (for cluster distribution)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze images without writing output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only N images (for testing)",
    )

    args = parser.parse_args()

    source = args.source.resolve()
    output_filtered = args.output_filtered.resolve()
    output_irregular = args.output_irregular.resolve()

    if not source.exists():
        logger.error(f"Source directory does not exist: {source}")
        return 1

    if not args.dry_run:
        output_filtered.mkdir(parents=True, exist_ok=True)
        output_irregular.mkdir(parents=True, exist_ok=True)

    # Determine number of workers
    num_workers = args.workers
    if num_workers <= 0:
        num_workers = cpu_count()

    # Parse shard info
    shard_index = None
    shard_total = None
    if args.shard:
        shard_index, shard_total = args.shard
        if not (0 <= shard_index < shard_total):
            logger.error(f"Invalid shard: INDEX must be in [0, {shard_total-1}]")
            return 1

    logger.info(f"Source:          {source}")
    logger.info(f"Output filtered: {output_filtered}")
    logger.info(f"Output irregular: {output_irregular}")
    logger.info(f"Aspect range:    {args.aspect_min:.2f} - {args.aspect_max:.2f}")
    logger.info(f"Min dimensions:  {args.min_width}x{args.min_height}")
    logger.info(f"Fuzz threshold:  {args.fuzz}")
    logger.info(f"Workers:         {num_workers}")
    if shard_index is not None:
        logger.info(f"Shard:           {shard_index}/{shard_total}")
    logger.info(f"Dry run:         {args.dry_run}\n")

    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = [
        p for ext in image_extensions for p in source.rglob(f"*{ext}") if p.is_file()
    ]
    image_files.sort()

    # Apply shard filtering
    if shard_index is not None:
        image_files = [
            f for i, f in enumerate(image_files) if i % shard_total == shard_index
        ]

    if args.limit:
        image_files = image_files[: args.limit]

    logger.info(f"Found {len(image_files)} images to process\n")

    # Prepare arguments for each image
    worker_args = [
        (
            image_path,
            output_filtered,
            output_irregular,
            source,
            args.aspect_min,
            args.aspect_max,
            args.min_width,
            args.min_height,
            args.fuzz,
            args.dry_run,
        )
        for image_path in image_files
    ]

    stats = {"success": 0, "filtered": 0, "irregular": 0, "failed": 0}

    # Process images in parallel
    if num_workers > 1:
        logger.info(f"Processing with {num_workers} workers...\n")
        with Pool(num_workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_process_image_worker, worker_args, chunksize=10),
                1,
            ):
                if result["status"] == "success" or result["status"] == "dry_run":
                    stats["success"] += 1
                    if result["category"] == "filtered":
                        stats["filtered"] += 1
                    else:
                        stats["irregular"] += 1
                else:
                    stats["failed"] += 1

                if i % max(1, len(worker_args) // 10) == 0 or i == len(worker_args):
                    logger.info(
                        f"  [{i}/{len(worker_args)}] "
                        f"Success: {stats['success']}, "
                        f"Failed: {stats['failed']}"
                    )
    else:
        # Single-threaded fallback
        logger.info("Processing with 1 worker...\n")
        for i, image_path in enumerate(image_files, 1):
            result = process_image(
                image_path,
                output_filtered,
                output_irregular,
                source,
                args.aspect_min,
                args.aspect_max,
                args.min_width,
                args.min_height,
                args.fuzz,
                args.dry_run,
            )

            if result["status"] == "success" or result["status"] == "dry_run":
                stats["success"] += 1
                if result["category"] == "filtered":
                    stats["filtered"] += 1
                else:
                    stats["irregular"] += 1
            else:
                stats["failed"] += 1

            if i % max(1, len(image_files) // 10) == 0 or i == len(image_files):
                logger.info(
                    f"  [{i}/{len(image_files)}] "
                    f"Success: {stats['success']}, "
                    f"Failed: {stats['failed']}"
                )

    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Total processed:  {stats['success']}")
    logger.info(f"  Filtered:       {stats['filtered']}")
    logger.info(f"  Irregular:      {stats['irregular']}")
    logger.info(f"Failed:           {stats['failed']}")

    return 0


if __name__ == "__main__":
    exit(main())
