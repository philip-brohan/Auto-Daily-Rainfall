#!/usr/bin/env python3
"""List and inspect consensus variants in a dataset.

ENVIRONMENT: Run this script in the weather-doc-extractor conda environment:
  conda activate weather-doc-extractor

Shows all consensus variants found in a dataset root, including their
configuration and summary statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List consensus variants in a dataset")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root directory containing variant subdirectories",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    # Find all variant directories (those containing consensus_config.json)
    variants = []
    for variant_dir in sorted(dataset_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        config_file = variant_dir / "consensus_config.json"
        if not config_file.exists():
            continue

        try:
            config_data = json.loads(config_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        # Count consensus transcriptions
        consensus_dir = variant_dir / "consensus_transcriptions"
        num_consensus = (
            len(list(consensus_dir.glob("*.json"))) if consensus_dir.exists() else 0
        )

        # Try to load summary
        summary_file = variant_dir / "consensus_summary.json"
        summary = None
        if summary_file.exists():
            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        variants.append(
            {
                "name": variant_dir.name,
                "path": str(variant_dir),
                "config": config_data,
                "num_consensus": num_consensus,
                "summary": summary,
            }
        )

    if not variants:
        print(f"No consensus variants found in {dataset_root}")
        return

    print(f"\nConsensus variants in {dataset_root}:\n")
    for i, var in enumerate(variants, 1):
        print(f"{i}. {var['name']}")
        print(f"   Path: {var['path']}")

        config = var["config"]
        print(f"   Description: {config.get('description', '(no description)')}")
        print(f"   Extraction dirs: {len(config.get('extraction_dirs', []))} model(s)")
        for model_dir in config.get("extraction_dirs", []):
            print(f"                    - {model_dir}")
        print(f"   Agreement threshold: {config.get('agreement_threshold', 'N/A')}")
        print(
            "   Null threshold: "
            f"{config.get('null_threshold', config.get('agreement_threshold', 'N/A'))}"
        )
        print(f"   Precision: {config.get('precision', 'N/A')}")
        print(f"   Timestamp: {config.get('timestamp', 'N/A')}")

        if config.get("notes"):
            print(f"   Notes: {config['notes']}")

        print(f"   Consensus transcriptions: {var['num_consensus']}")

        if var["summary"] is not None:
            summary = var["summary"]
            print(
                f"   Summary: {summary.get('total_stems', 0)} stems, "
                f"{summary.get('correct_cells', 0)}/{summary.get('total_cells', 0)} cells correct "
                f"({summary.get('agreement_coverage', 0)*100:.1f}%)"
            )

        print()


if __name__ == "__main__":
    main()
