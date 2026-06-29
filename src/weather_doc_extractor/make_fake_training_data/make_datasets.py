"""Batch-generate a synthetic daily-rainfall training dataset.

Usage::

    python -m weather_doc_extractor.make_fake_training_data.make_datasets
    python -m weather_doc_extractor.make_fake_training_data.make_datasets \\
        --n-records 500 \\
        --output-dir /data/fake_daily_rainfall \\
        --seed 42

Each generated record is a unique (year, county, station_id) combination.
The outputs follow the same directory layout as the real sample data::

    <output_dir>/
        images/          DRain_1871-1880_Cornwall-59.jpg  …
        transcriptions/  DRain_1871-1880_Cornwall-59.json …
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from .make_pair import make_pair
from .make_data import make_stem
from .constants import COUNTIES


def main(
    n_records: int = 1000,
    output_dir: str = "fake_daily_rainfall",
    seed: int | None = None,
    font_size: float = 20.0,
    font_size_jitter: float = 0.22,
    jitter_grid_points: float = 0.0008,
    jpeg_quality: int = 85,
    right_day_label_probability: float = 0.9,
    post_dec_blank_column_probability: float = 0.2,
    line_intensity_sigma: float = 0.40,
    individual_line_intensity_sigma: float = 0.0,
) -> None:
    """Generate *n_records* image/JSON pairs into *output_dir*.

    Parameters
    ----------
    n_records:
        Number of records to generate.
    output_dir:
        Root output directory (created if absent).
    seed:
        Master random seed; per-record seeds are derived from it so the
        entire batch is reproducible.
    font_size:
        Base font size in points.
    font_size_jitter:
        Fractional per-record variation in base font size. For example,
        0.22 means each record uses a random size in
        ``[font_size * 0.78, font_size * 1.22]``.
    jitter_grid_points:
        Std-dev of grid-point position jitter (normalised coordinates).
    jpeg_quality:
        JPEG compression quality (1–95).
    right_day_label_probability:
        Probability that each rendered record includes the extra right-hand
        day-label column.
    post_dec_blank_column_probability:
        Probability that each rendered record includes a blank column
        immediately to the right of Dec.
    line_intensity_sigma:
        Std-dev of per-page grid line intensity jitter.
    individual_line_intensity_sigma:
        Std-dev of per-line intensity variation (additive to per-page baseline).
    """
    master_rng = random.Random(seed)

    # Pre-draw per-record seeds so each record is independently reproducible
    record_seeds = [master_rng.randint(0, 2**31 - 1) for _ in range(n_records)]

    for idx, rec_seed in enumerate(record_seeds, start=1):
        rec_rng = random.Random(rec_seed)

        year = rec_rng.randint(1860, 1960)
        county = rec_rng.choice(COUNTIES)
        station_id = rec_rng.randint(1, 999)
        stem = make_stem(year, county, station_id)
        min_font = font_size * (1.0 - font_size_jitter)
        max_font = font_size * (1.0 + font_size_jitter)
        record_font_size = rec_rng.uniform(min_font, max_font)

        img_path, json_path = make_pair(
            stem=stem,
            output_dir=output_dir,
            year=year,
            county=county,
            station_id=station_id,
            seed=rec_seed,
            font_size=record_font_size,
            jitter_grid_points=jitter_grid_points,
            jpeg_quality=jpeg_quality,
            right_day_label_probability=right_day_label_probability,
            post_dec_blank_column_probability=post_dec_blank_column_probability,
            line_intensity_sigma=line_intensity_sigma,
            individual_line_intensity_sigma=individual_line_intensity_sigma,
        )

        if idx % 50 == 0 or idx == n_records:
            print(f"[{idx:>{len(str(n_records))}}/{n_records}]  {img_path.name}")

    print(f"\nDone — {n_records} records written to {output_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-generate fake daily-rainfall training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n-records",
        type=int,
        default=1000,
        help="Number of image/JSON pairs to generate.",
    )
    p.add_argument(
        "--output-dir", default="fake_daily_rainfall", help="Root output directory."
    )
    p.add_argument("--seed", type=int, default=None, help="Master random seed.")
    p.add_argument(
        "--font-size", type=float, default=20.0, help="Base font size in points."
    )
    p.add_argument(
        "--font-size-jitter",
        type=float,
        default=0.22,
        help="Per-record fractional variation in base font size.",
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


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if not 0.0 <= args.right_day_label_probability <= 1.0:
        raise ValueError("--right-day-label-probability must be between 0 and 1")
    if not 0.0 <= args.post_dec_blank_column_probability <= 1.0:
        raise ValueError("--post-dec-blank-column-probability must be between 0 and 1")
    if args.line_intensity_sigma < 0.0:
        raise ValueError("--line-intensity-sigma must be non-negative")
    if args.individual_line_intensity_sigma < 0.0:
        raise ValueError("--individual-line-intensity-sigma must be non-negative")
    main(
        n_records=args.n_records,
        output_dir=args.output_dir,
        seed=args.seed,
        font_size=args.font_size,
        font_size_jitter=args.font_size_jitter,
        jitter_grid_points=args.jitter_grid_points,
        jpeg_quality=args.jpeg_quality,
        right_day_label_probability=args.right_day_label_probability,
        post_dec_blank_column_probability=args.post_dec_blank_column_probability,
        line_intensity_sigma=args.line_intensity_sigma,
        individual_line_intensity_sigma=args.individual_line_intensity_sigma,
    )
