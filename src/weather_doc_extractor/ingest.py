"""Ingest daily-rainfall document images and their transcriptions.

Public API
----------
parse_stem(stem)       Parse a filename stem into metadata fields.
load_grid(path)        Parse a transcription JSON into a DailyRainfallGrid.
scan_records(...)      Scan paired image/transcription directories.
to_hf_dataset(records) Convert records to a HuggingFace Dataset.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from weather_doc_extractor.schemas import DailyRainfallGrid, DailyRainfallRecord

# DRain_{decade}_{County}-{station_id}
_STEM_RE = re.compile(r"^DRain_(\d{4}-\d{4})_(.+)-(\d+)$")
_COLLISION_SUFFIX_RE = re.compile(r"__(?:[0-9a-f]{8})(?:_\d+)?$")


def _strip_collision_suffix(stem: str) -> str:
    """Remove sampler-added collision suffixes from stems.

    sample_unseen_images.py may append ``__<8hex>`` (optionally ``_<n>``)
    when two source files share the same basename. The metadata-bearing
    DRain prefix remains intact, so strip only this trailing collision tag.
    """
    return _COLLISION_SUFFIX_RE.sub("", stem)


def parse_stem(stem: str) -> dict[str, str]:
    """Return ``{"decade", "county", "station_id"}`` parsed from *stem*.

    Raises ``ValueError`` if the stem does not match the expected pattern.
    """
    m = _STEM_RE.match(_strip_collision_suffix(stem))
    if not m:
        raise ValueError(f"Stem does not match expected pattern: {stem!r}")
    decade, county, station_id = m.groups()
    return {"decade": decade, "county": county, "station_id": station_id}


def _coerce_value(raw: object) -> float | None:
    """Convert a raw JSON value to float or None."""
    if raw is None or raw == "null":
        return None
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def load_grid(path: Path) -> DailyRainfallGrid:
    """Load a transcription JSON file and return a :class:`DailyRainfallGrid`.

    The JSON is expected to have string keys ``"Day 1"`` … ``"Day 31"`` and
    ``"Totals"``, each mapped to a 12-element list of rainfall values (inches)
    given as strings or ``"null"``.
    """
    raw: dict[str, list[object]] = json.loads(path.read_text(encoding="utf-8"))
    days: dict[str, list[float | None]] = {}
    totals: list[float | None] = []
    for key, values in raw.items():
        parsed = [_coerce_value(v) for v in values]
        if key == "Totals":
            totals = parsed
        else:
            days[key] = parsed
    return DailyRainfallGrid(days=days, totals=totals)


def scan_records(
    images_dir: Path,
    transcriptions_dir: Path,
) -> list[DailyRainfallRecord]:
    """Scan *images_dir* for ``*.jpg`` files and pair each with a JSON.

    Images without a matching transcription are included with
    ``grid=None`` and ``transcription_path=None`` so they can be
    processed later.  Records are sorted by stem for reproducibility.
    """
    records: list[DailyRainfallRecord] = []
    image_paths = sorted(
        list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
    )
    for image_path in image_paths:
        stem = image_path.stem
        try:
            meta = parse_stem(stem)
        except ValueError:
            continue

        transcription_path = transcriptions_dir / f"{stem}.json"
        grid: DailyRainfallGrid | None = None
        if transcription_path.exists():
            grid = load_grid(transcription_path)
        else:
            transcription_path = None  # type: ignore[assignment]

        records.append(
            DailyRainfallRecord(
                stem=stem,
                county=meta["county"],
                station_id=meta["station_id"],
                decade=meta["decade"],
                image_path=image_path,
                transcription_path=transcription_path,
                grid=grid,
            )
        )
    return records


def to_hf_dataset(records: list[DailyRainfallRecord]):  # type: ignore[return]
    """Build a HuggingFace ``Dataset`` from *records*.

    Each row contains:

    - ``image`` — PIL image (via ``datasets.Image()``)
    - ``stem``, ``county``, ``station_id``, ``decade`` — string metadata
    - ``grid_json`` — serialised grid as a JSON string (empty for unpaired images)
    - ``has_transcription`` — bool flag

    Requires the ``train`` extras (``datasets``, ``pillow``).
    """
    try:
        import datasets  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Install the 'train' extras to build HuggingFace datasets: "
            "pip install -e '.[train]'"
        ) from exc

    rows: dict[str, list[object]] = {
        "stem": [],
        "county": [],
        "station_id": [],
        "decade": [],
        "image": [],
        "grid_json": [],
        "has_transcription": [],
    }
    for rec in records:
        rows["stem"].append(rec.stem)
        rows["county"].append(rec.county)
        rows["station_id"].append(rec.station_id)
        rows["decade"].append(rec.decade)
        rows["image"].append(str(rec.image_path))
        rows["grid_json"].append(json.dumps(asdict(rec.grid)) if rec.grid else "")
        rows["has_transcription"].append(rec.grid is not None)

    features = datasets.Features(
        {
            "stem": datasets.Value("string"),
            "county": datasets.Value("string"),
            "station_id": datasets.Value("string"),
            "decade": datasets.Value("string"),
            "image": datasets.Image(),
            "grid_json": datasets.Value("string"),
            "has_transcription": datasets.Value("bool"),
        }
    )
    return datasets.Dataset.from_dict(rows, features=features)
