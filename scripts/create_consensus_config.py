#!/usr/bin/env python3
"""Create a consensus variant configuration file.

ENVIRONMENT: Run this script in the weather-doc-extractor conda environment:
  conda activate weather-doc-extractor

Generates a JSON config specifying which extraction directories to use,
agreement threshold, precision, and other parameters for consensus building.

Config format:
{
  "variant_name": "consensus_1000",
  "description": "First consensus round with 5 base models",
  "extraction_dirs": ["path/to/extraction1", "path/to/extraction2", ...],
  "agreement_threshold": 4,
    "null_threshold": 4,
  "precision": 3,
  "timestamp": "2026-06-15T12:34:56",
  "notes": "Optional notes about this variant"
}
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a consensus variant configuration file"
    )
    parser.add_argument(
        "--variant-name",
        type=str,
        required=True,
        help="Name for this consensus variant (e.g., 'consensus_1000')",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root directory containing {variant-name}/ subdirectory",
    )
    parser.add_argument(
        "--extraction-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more extraction directories to use for consensus voting",
    )
    parser.add_argument(
        "--agreement-threshold",
        type=int,
        default=4,
        help="Minimum number of models that must agree for 'correct=true' (default: 4)",
    )
    parser.add_argument(
        "--null-threshold",
        type=int,
        default=None,
        help="Minimum number of models that must agree when the consensus value is null. "
        "Defaults to --agreement-threshold.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal precision for normalizing model values before voting (default: 3)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Human-readable description of this consensus variant",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes about this variant (e.g., fine-tuning details)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    variant_name = args.variant_name
    variant_dir = dataset_root / variant_name

    # Validate inputs
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    if args.agreement_threshold < 1:
        raise SystemExit("--agreement-threshold must be >= 1")

    null_threshold = args.null_threshold
    if null_threshold is None:
        null_threshold = args.agreement_threshold
    elif null_threshold < 1:
        raise SystemExit("--null-threshold must be >= 1")

    extraction_dirs = [d.resolve() for d in args.extraction_dirs]
    for ext_dir in extraction_dirs:
        if not ext_dir.exists():
            raise SystemExit(f"Extraction directory not found: {ext_dir}")

    # Create variant directory if needed
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Prepare config data
    config = {
        "variant_name": variant_name,
        "description": args.description,
        "extraction_dirs": [str(d) for d in extraction_dirs],
        "agreement_threshold": args.agreement_threshold,
        "null_threshold": null_threshold,
        "precision": args.precision,
        "timestamp": datetime.now().isoformat(),
        "notes": args.notes,
    }

    # Write config file
    config_file = variant_dir / "consensus_config.json"
    if config_file.exists() and not args.overwrite:
        raise SystemExit(
            f"Config file already exists: {config_file}. Use --overwrite to replace."
        )

    config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"✓ Created consensus config")
    print(f"  Variant:           {variant_name}")
    print(f"  Dataset root:      {dataset_root}")
    print(f"  Variant dir:       {variant_dir}")
    print(f"  Config file:       {config_file}")
    print(f"  Extraction dirs:   {len(extraction_dirs)} model(s)")
    for i, d in enumerate(extraction_dirs, 1):
        print(f"                     {i}. {d}")
    print(f"  Agreement thresh:  {args.agreement_threshold}")
    print(f"  Null thresh:       {null_threshold}")
    print(f"  Precision:         {args.precision}")
    if args.description:
        print(f"  Description:       {args.description}")
    if args.notes:
        print(f"  Notes:             {args.notes}")


if __name__ == "__main__":
    main()
