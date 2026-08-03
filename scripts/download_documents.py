#!/usr/bin/env python
"""
Download the historical daily-rainfall documents from the Met Office /
National Meteorological Library and Archive (NMLA) digital archive.

The archive is a Preservica "Universal Access" collection, surfaced through the
public NMLA portal at https://digital.nmla.metoffice.gov.uk. The "England &
Wales" daily-rainfall collection is a tree of folders:

    England & Wales (root folder)
      └─ DRain_1871-1880            (decade folder)
           ├─ DRain_1871-1880_Cornwall        (a document = one PDF)
           ├─ DRain_1871-1880_Devonshire_Part1
           └─ ...
      └─ DRain_1881-1890
           └─ ...

Each *folder* is a Preservica Structural Object (id ``SO_<uuid>``) and each
*document* is an Information Object (id ``IO_<uuid>``). A document downloads as
a single multi-page PDF (one county, or a part/box of a county, for a decade).

This script walks the folder tree from a root folder and downloads every
document PDF it finds. Downloads are idempotent and resumable: a file whose
size already matches the server's ``Content-Length`` is skipped, and each
download is written to a temporary file and atomically renamed on success.

The PDFs are the raw archive documents. To turn them into the per-page JPEG
images the extraction pipeline consumes, run ``scripts/split_documents.py``
afterwards (see the preparation notebook).

Data licence: the NMLA daily-rainfall collection is published under the Open
Government Licence. Please crawl politely (the default request delay is
deliberately conservative).

ENVIRONMENT: Run this script in the weather-doc-extractor conda environment:
    conda activate weather-doc-extractor

Usage (download the whole England & Wales collection):
    python scripts/download_documents.py --output /path/to/documents

Usage (enumerate only, write a manifest, download nothing):
    python scripts/download_documents.py --output /path/to/documents \\
        --dry-run --manifest /path/to/documents/manifest.json

Usage (one decade only):
    python scripts/download_documents.py --output /path/to/documents \\
        --root SO_7339c4e5-c3cf-4277-a1d2-a7e551f5aa59

Usage (cluster - split the document list across 6 jobs):
    for i in {0..5}; do
        sbatch --job-name=download_$i --time=12:00:00 \\
            scripts/download_documents.py --output /path/to/documents \\
            --shard $i 6
    done
"""

import argparse
import html
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Root folder of the "England & Wales" daily-rainfall collection.
DEFAULT_ROOT = "SO_51194883-b9dd-4e27-93db-958f8fbea38b"
DEFAULT_PORTAL = "https://digital.nmla.metoffice.gov.uk"
USER_AGENT = (
    "Auto-Daily-Rainfall-MO/1.0 "
    "(+https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO)"
)

# A result item in a folder listing looks like:
#   <div class="... result-item ..." ... onclick="window.location=
#       'https://metoffice.access.preservica.com/IO_<uuid>/'" title="DRain_...">
# This captures the child entity id (SO_ = folder, IO_ = document) and title.
_ITEM_RE = re.compile(
    r"result-item[^>]*?onclick=\"window\.location='"
    r"https://metoffice\.access\.preservica\.com/"
    r"((?:SO|IO)_[0-9a-f-]+)/'\"\s+title=\"([^\"]+)\"",
    re.IGNORECASE,
)


def _http_get(url: str, retries: int, backoff: float) -> bytes:
    """GET a URL with retries and exponential backoff, returning the body."""
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
            last_error = error
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "GET failed (attempt %d/%d) for %s: %s - retrying in %.1fs",
                attempt,
                retries,
                url,
                error,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last_error


def list_folder(
    portal: str, folder_id: str, retries: int, backoff: float, delay: float
) -> List[Tuple[str, str]]:
    """List the direct children of a folder.

    Returns a list of ``(entity_id, title)`` tuples, where ``entity_id`` starts
    with ``SO_`` (a sub-folder) or ``IO_`` (a document). Pages through the
    portal's paginated listing until an empty page is reached.
    """
    children: List[Tuple[str, str]] = []
    seen: set = set()
    page = 1
    while True:
        query = urllib.parse.urlencode({"pg": page, "name": folder_id})
        url = f"{portal}/?{query}"
        body = _http_get(url, retries, backoff).decode("utf-8", errors="replace")
        matches = _ITEM_RE.findall(body)
        if not matches:
            break
        new_on_page = 0
        for entity_id, title in matches:
            if entity_id in seen:
                continue
            seen.add(entity_id)
            children.append((entity_id, html.unescape(title)))
            new_on_page += 1
        # Defensive: if a page repeats the previous page's items, stop.
        if new_on_page == 0:
            break
        page += 1
        time.sleep(delay)
    return children


def enumerate_documents(
    portal: str, root_id: str, retries: int, backoff: float, delay: float
) -> List[Dict[str, str]]:
    """Recursively walk the folder tree and return all documents (PDFs).

    Each returned dict has ``id`` (the ``IO_<uuid>``), ``title`` (the document
    name, e.g. ``DRain_1871-1880_Cornwall``) and ``folder`` (the title of the
    containing folder, e.g. ``DRain_1871-1880``).
    """
    documents: List[Dict[str, str]] = []
    # Stack of (folder_id, folder_title) to visit.
    stack: List[Tuple[str, str]] = [(root_id, "")]
    visited: set = set()
    while stack:
        folder_id, folder_title = stack.pop()
        if folder_id in visited:
            continue
        visited.add(folder_id)
        logger.info("Listing folder %s (%s)", folder_title or "root", folder_id)
        for entity_id, title in list_folder(portal, folder_id, retries, backoff, delay):
            if entity_id.startswith("SO_"):
                stack.append((entity_id, title))
            elif entity_id.startswith("IO_"):
                documents.append(
                    {"id": entity_id, "title": title, "folder": folder_title}
                )
    # Stable ordering makes sharding deterministic across runs.
    documents.sort(key=lambda doc: doc["title"])
    return documents


def _remote_size(url: str, retries: int, backoff: float) -> Optional[int]:
    """Return the Content-Length of a URL via a HEAD request, or None."""
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}, method="HEAD"
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                length = response.headers.get("Content-Length")
                return int(length) if length is not None else None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            time.sleep(backoff * (2 ** (attempt - 1)))
    return None


def download_document(
    portal: str,
    document: Dict[str, str],
    output_dir: Path,
    retries: int,
    backoff: float,
) -> str:
    """Download a single document PDF. Idempotent and atomic.

    Returns one of ``"downloaded"``, ``"skipped"`` or ``"failed"``.
    """
    url = f"{portal}/download/file/{document['id']}"
    dest = output_dir / f"{document['title']}.pdf"
    tmp = dest.with_suffix(".pdf.part")

    # A file at the destination was written by a previous run's atomic rename,
    # so it is complete by construction. If the server reports a size, use it as
    # an extra integrity check; otherwise trust the existing file.
    remote_size = _remote_size(url, retries, backoff)
    if dest.exists() and (remote_size is None or dest.stat().st_size == remote_size):
        logger.info("Skipping %s (already complete)", dest.name)
        return "skipped"

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                open(tmp, "wb") as handle,
            ):
                while True:
                    chunk = response.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    handle.write(chunk)
            if remote_size is not None and tmp.stat().st_size != remote_size:
                raise IOError(
                    f"size mismatch: got {tmp.stat().st_size}, expected {remote_size}"
                )
            tmp.replace(dest)
            logger.info("Downloaded %s (%d bytes)", dest.name, dest.stat().st_size)
            return "downloaded"
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            IOError,
        ) as error:
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "Download failed (attempt %d/%d) for %s: %s - retrying in %.1fs",
                attempt,
                retries,
                dest.name,
                error,
                wait,
            )
            time.sleep(wait)
    logger.error("Giving up on %s after %d attempts", dest.name, retries)
    tmp.unlink(missing_ok=True)
    return "failed"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download NMLA daily-rainfall document PDFs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to save downloaded PDFs into.",
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help=f"Root folder entity id to crawl (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--portal",
        default=DEFAULT_PORTAL,
        help=f"Archive portal base URL (default: {DEFAULT_PORTAL}).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to write the enumerated document list as JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate documents (and write the manifest) but download nothing.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between HTTP requests (default: 1.0).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Number of attempts per request before giving up (default: 4).",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=2.0,
        help="Base seconds for exponential retry backoff (default: 2.0).",
    )
    parser.add_argument(
        "--shard",
        nargs=2,
        type=int,
        metavar=("INDEX", "TOTAL"),
        default=None,
        help="Download only shard INDEX of TOTAL (round-robin over documents).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("Enumerating documents under %s ...", args.root)
    documents = enumerate_documents(
        args.portal, args.root, args.retries, args.backoff, args.delay
    )
    logger.info("Found %d documents", len(documents))

    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(documents, indent=2))
        logger.info("Wrote manifest to %s", args.manifest)

    if args.shard is not None:
        index, total = args.shard
        if not 0 <= index < total:
            logger.error("Invalid --shard %d %d", index, total)
            return 2
        documents = documents[index::total]
        logger.info("Shard %d/%d: %d documents", index, total, len(documents))

    if args.dry_run:
        logger.info("Dry run - not downloading %d documents", len(documents))
        return 0

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    for document in documents:
        status = download_document(
            args.portal, document, args.output, args.retries, args.backoff
        )
        counts[status] += 1
        if status == "downloaded":
            time.sleep(args.delay)

    logger.info(
        "Done: %d downloaded, %d skipped, %d failed",
        counts["downloaded"],
        counts["skipped"],
        counts["failed"],
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
