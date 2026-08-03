#!/usr/bin/env python
"""
Split downloaded daily-rainfall document PDFs into per-page JPEG images.

The archive documents (see ``scripts/download_documents.py``) are multi-page
PDFs, one per county (or county part / box) per decade. The extraction pipeline
works on single-page JPEG images, one station-year table per image, named:

    <document-stem>-<page>.jpg      e.g. DRain_1871-1880_Cornwall-59.jpg

Page numbers are **zero-based** (the first page is ``-0``), matching the
existing sample and training data.

Rendering uses ``pdftoppm`` (poppler-utils). Each PDF is rendered to a
temporary prefix and the pages are renamed to the zero-based convention. The
step is idempotent: a PDF whose pages are all already present in the output
directory is skipped.

The JPEGs produced here still contain the scanned page margins. Run
``scripts/preprocess_images.py`` afterwards to trim whitespace and separate
regular from irregular pages.

ENVIRONMENT: Run this script in the weather-doc-extractor conda environment:
    conda activate weather-doc-extractor

Usage (single machine):
    python scripts/split_documents.py \\
        --source /path/to/documents \\
        --output /path/to/images \\
        [--dpi 150] [--workers 8] [--dry-run]

Usage (cluster - 6 jobs, 8 workers each):
    for i in {0..5}; do
        sbatch --job-name=split_$i --time=6:00:00 \\
            scripts/split_documents.py \\
            --source /path/to/documents \\
            --output /path/to/images \\
            --workers 8 --shard $i 6
    done
"""

import argparse
import logging
import subprocess
import sys
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def pdf_page_count(pdf_path: Path) -> Optional[int]:
    """Return the number of pages in a PDF using pdfinfo, or None on error."""
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        logger.error("pdfinfo failed for %s: %s", pdf_path.name, error)
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def already_split(stem: str, page_count: int, output_dir: Path) -> bool:
    """True if every zero-based page image for a document already exists."""
    return all((output_dir / f"{stem}-{i}.jpg").exists() for i in range(page_count))


def split_pdf(pdf_path: Path, output_dir: Path, dpi: int, dry_run: bool) -> str:
    """Split one PDF into zero-based per-page JPEGs.

    Returns one of ``"split"``, ``"skipped"`` or ``"failed"``.
    """
    stem = pdf_path.stem
    page_count = pdf_page_count(pdf_path)
    if page_count is None:
        return "failed"

    if already_split(stem, page_count, output_dir):
        logger.info("Skipping %s (%d pages already present)", stem, page_count)
        return "skipped"

    if dry_run:
        logger.info("Would split %s into %d pages", stem, page_count)
        return "split"

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir:
        prefix = Path(tmp_dir) / stem
        try:
            subprocess.run(
                ["pdftoppm", "-jpeg", "-r", str(dpi), str(pdf_path), str(prefix)],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            logger.error("pdftoppm failed for %s: %s", stem, error)
            return "failed"

        # pdftoppm emits one-based, zero-padded names (e.g. "<stem>-001.jpg").
        # Within a single PDF the padding width is fixed, so a lexical sort is
        # the correct page order. Rename to the zero-based convention.
        rendered = sorted(Path(tmp_dir).glob(f"{stem}-*.jpg"))
        for index, page in enumerate(rendered):
            page.replace(output_dir / f"{stem}-{index}.jpg")

    logger.info("Split %s into %d pages", stem, len(rendered))
    return "split"


def _worker(args: tuple) -> str:
    return split_pdf(*args)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split rainfall document PDFs into per-page JPEG images.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing the downloaded document PDFs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write per-page JPEG images into.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Rendering resolution in DPI (default: 150).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, cpu_count()),
        help="Number of parallel worker processes (default: min(8, CPUs)).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be split without writing any images.",
    )
    parser.add_argument(
        "--shard",
        nargs=2,
        type=int,
        metavar=("INDEX", "TOTAL"),
        default=None,
        help="Process only shard INDEX of TOTAL (round-robin over PDFs).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    pdfs = sorted(args.source.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDFs found in %s", args.source)
        return 2

    if args.shard is not None:
        index, total = args.shard
        if not 0 <= index < total:
            logger.error("Invalid --shard %d %d", index, total)
            return 2
        pdfs = pdfs[index::total]
        logger.info("Shard %d/%d: %d PDFs", index, total, len(pdfs))

    logger.info("Splitting %d PDFs with %d workers", len(pdfs), args.workers)
    tasks = [(pdf, args.output, args.dpi, args.dry_run) for pdf in pdfs]

    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [_worker(task) for task in tasks]

    counts = {"split": 0, "skipped": 0, "failed": 0}
    for status in results:
        counts[status] += 1
    logger.info(
        "Done: %d split, %d skipped, %d failed",
        counts["split"],
        counts["skipped"],
        counts["failed"],
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
