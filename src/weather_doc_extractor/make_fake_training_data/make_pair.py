"""Generate one synthetic daily-rainfall image + JSON transcription pair.

Can be run directly::

    python -m weather_doc_extractor.make_fake_training_data.make_pair \\
        --stem DRain_1871-1880_Cornwall-59 \\
        --year 1875 \\
        --county Cornwall \\
        --station-id 59 \\
        --output-dir ./fake_daily_rainfall

Or call :func:`make_pair` programmatically.
"""

from __future__ import annotations

import argparse
import json
import random
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import geometry as geo
from .constants import COUNTIES, get_available_font_families
from .draw_grid import draw_document
from .make_data import generate_data, make_stem


@lru_cache(maxsize=1)
def _real_image_sizes() -> list[tuple[int, int]]:
    """Return empirical (width, height) sizes from real daily rainfall images."""
    repo_root = Path(__file__).resolve().parents[3]
    images_dir = repo_root / "Daily_rainfall_sample" / "images"
    sizes: list[tuple[int, int]] = []
    for p in images_dir.glob("*.jpg"):
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except OSError:
            continue
    # Fallback to the previous default canvas if sample images are unavailable.
    if not sizes:
        return [(int(geo.PAGE_WIDTH_IN * geo.DPI), int(geo.PAGE_HEIGHT_IN * geo.DPI))]
    return sizes


def make_pair(
    stem: str,
    output_dir: str | Path,
    year: int,
    county: str,
    station_id: int,
    *,
    seed: int | None = None,
    font_family: str | None = None,
    font_size: float = 20.0,
    jitter_grid_points: float = 0.0008,
    jpeg_quality: int = 85,
    right_day_label_probability: float = 0.9,
    post_dec_blank_column_probability: float = 0.2,
    line_intensity_sigma: float = 0.40,
    individual_line_intensity_sigma: float = 0.0,
) -> tuple[Path, Path]:
    """Generate one image + JSON pair and write them to *output_dir*.

    Parameters
    ----------
    stem:
        Filename stem (without extension), e.g.
        ``"DRain_1871-1880_Cornwall-59"``.  Both output files derive their
        names from this.
    output_dir:
        Root directory.  Two sub-directories are created/used:
        ``images/`` for the JPEG and ``transcriptions/`` for the JSON.
    year, county, station_id:
        Metadata written into the page header and used to generate the stem.
    seed:
        Optional integer seed for reproducibility.
    font_family:
        Matplotlib font family string; random if *None*.
    font_size:
        Base font size in points.
    jitter_grid_points:
        Position jitter std-dev (normalised page coordinates).
    jpeg_quality:
        JPEG compression quality (1–95).
    right_day_label_probability:
        Probability that the extra right-hand day-label column is rendered.
    post_dec_blank_column_probability:
        Probability that an additional blank column appears immediately to the
        right of Dec.
    line_intensity_sigma:
        Std-dev of per-page grid line intensity jitter.
    individual_line_intensity_sigma:
        Std-dev of per-line intensity variation (additive to per-page baseline).

    Returns
    -------
    (image_path, json_path)
        Paths to the files that were written.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    if font_family is None:
        font_family = py_rng.choice(get_available_font_families())

    data = generate_data(year, rng=rng)

    # Match real-corpus size/aspect distribution by sampling an observed image size.
    target_w, target_h = py_rng.choice(_real_image_sizes())
    base_w = int(geo.PAGE_WIDTH_IN * geo.DPI)
    base_h = int(geo.PAGE_HEIGHT_IN * geo.DPI)
    # Scale typography with canvas area so smaller sampled images do not look oversized.
    font_scale = ((target_w * target_h) / (base_w * base_h)) ** 0.5
    effective_font_size = max(6.0, font_size * font_scale)

    fig = draw_document(
        data,
        year=year,
        county=county,
        station_id=station_id,
        font_family=font_family,
        font_size=effective_font_size,
        jitter_grid_points=jitter_grid_points,
        right_day_label_probability=right_day_label_probability,
        post_dec_blank_column_probability=post_dec_blank_column_probability,
        line_intensity_sigma=line_intensity_sigma,
        individual_line_intensity_sigma=individual_line_intensity_sigma,
        rng=rng,
    )

    fig.set_size_inches(target_w / geo.DPI, target_h / geo.DPI)

    out = Path(output_dir)
    images_dir = out / "images"
    transcriptions_dir = out / "transcriptions"
    images_dir.mkdir(parents=True, exist_ok=True)
    transcriptions_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / f"{stem}.jpg"
    json_path = transcriptions_dir / f"{stem}.json"

    fig.savefig(
        image_path,
        format="jpeg",
        pil_kwargs={"quality": jpeg_quality},
        dpi=geo.DPI,
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4)

    return image_path, json_path


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate one fake daily-rainfall image + JSON pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir", default="fake_daily_rainfall", help="Root output directory."
    )
    p.add_argument(
        "--year",
        type=int,
        default=None,
        help="Calendar year (random 1860–1960 if omitted).",
    )
    p.add_argument("--county", default=None, help="County name (random if omitted).")
    p.add_argument(
        "--station-id",
        type=int,
        default=None,
        help="Station number (random 1–999 if omitted).",
    )
    p.add_argument(
        "--stem",
        default=None,
        help="Filename stem (derived from year/county/id if omitted).",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    p.add_argument("--font-family", default=None, help="Matplotlib font family.")
    p.add_argument(
        "--font-size", type=float, default=20.0, help="Base font size in points."
    )
    p.add_argument(
        "--jitter-grid-points",
        type=float,
        default=0.0008,
        help="Std-dev of grid-point position jitter.",
    )
    p.add_argument(
        "--jpeg-quality", type=int, default=85, help="JPEG compression quality (1–95)."
    )
    p.add_argument(
        "--right-day-label-probability",
        type=float,
        default=0.9,
        help="Probability of rendering the extra right-hand day-label column (0-1).",
    )
    p.add_argument(
        "--post-dec-blank-column-probability",
        type=float,
        default=0.2,
        help="Probability of rendering a blank column immediately to the right of Dec (0-1).",
    )
    p.add_argument(
        "--line-intensity-sigma",
        type=float,
        default=0.40,
        help="Std-dev of per-page grid line intensity jitter.",
    )
    p.add_argument(
        "--individual-line-intensity-sigma",
        type=float,
        default=0.0,
        help="Std-dev of per-line intensity variation (additive to per-page baseline).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if not 0.0 <= args.right_day_label_probability <= 1.0:
        raise ValueError("--right-day-label-probability must be between 0 and 1")
    if not 0.0 <= args.post_dec_blank_column_probability <= 1.0:
        raise ValueError("--post-dec-blank-column-probability must be between 0 and 1")
    if args.line_intensity_sigma < 0.0:
        raise ValueError("--line-intensity-sigma must be non-negative")
    if args.individual_line_intensity_sigma < 0.0:
        raise ValueError("--individual-line-intensity-sigma must be non-negative")

    py_rng = random.Random(args.seed)

    year = args.year if args.year is not None else py_rng.randint(1860, 1960)
    county = args.county if args.county is not None else py_rng.choice(COUNTIES)
    station_id = (
        args.station_id if args.station_id is not None else py_rng.randint(1, 999)
    )
    stem = args.stem if args.stem is not None else make_stem(year, county, station_id)

    img, jsn = make_pair(
        stem=stem,
        output_dir=args.output_dir,
        year=year,
        county=county,
        station_id=station_id,
        seed=args.seed,
        font_family=args.font_family,
        font_size=args.font_size,
        jitter_grid_points=args.jitter_grid_points,
        jpeg_quality=args.jpeg_quality,
        right_day_label_probability=args.right_day_label_probability,
        post_dec_blank_column_probability=args.post_dec_blank_column_probability,
        line_intensity_sigma=args.line_intensity_sigma,
        individual_line_intensity_sigma=args.individual_line_intensity_sigma,
    )
    print(f"image : {img}")
    print(f"json  : {jsn}")


if __name__ == "__main__":
    main()
