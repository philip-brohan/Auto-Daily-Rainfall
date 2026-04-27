from __future__ import annotations

import json
import sys
from pathlib import Path

from weather_doc_extractor.config import MODEL_PRESETS, AppConfig
from weather_doc_extractor.pipeline import (
    build_project_summary,
    describe_inference_stage,
    describe_ingest_stage,
    describe_training_stage,
    extract_from_image,
    run_batch_extract,
    run_evaluation,
    run_finetune,
    run_ingest,
)


def _parse_model_flag(args: list[str], config: AppConfig) -> list[str]:
    """Pop ``--model <name|id>`` from *args* and update *config* in place.

    Returns the remaining args with the flag and its value removed.
    Exits with an error message if the value is unrecognised.
    """
    remaining: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            value = args[i + 1]
            # Accept either a short preset name or a full HF model ID
            model_id = MODEL_PRESETS.get(value, value)
            config.model.model_name = model_id
            i += 2
        else:
            remaining.append(args[i])
            i += 1
    return remaining


def run(argv: list[str] | None = None) -> int:
    args = argv or ["info"]
    command = args[0]
    config = AppConfig()

    if command == "info":
        print("Weather document extraction project")
        print(json.dumps(build_project_summary(config), indent=2, default=str))
        return 0

    if command == "stages":
        print(describe_ingest_stage())
        print(describe_inference_stage())
        print(describe_training_stage())
        return 0

    if command == "ingest":
        summary = run_ingest(config)
        print(
            f"Scanned {summary['total']} records "
            f"({summary['paired']} paired, {summary['unpaired']} unpaired)"
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if command == "extract":
        if len(args) < 2:
            print(
                "Usage: extract [--model smolvlm|granite|<hf-id>] <image_path>",
                file=sys.stderr,
            )
            return 1
        remaining = _parse_model_flag(list(args[1:]), config)
        if not remaining:
            print(
                "Usage: extract [--model smolvlm|granite|<hf-id>] <image_path>",
                file=sys.stderr,
            )
            return 1
        image_path = Path(remaining[0])
        if not image_path.exists():
            print(f"Image not found: {image_path}", file=sys.stderr)
            return 1
        print(f"Extracting from {image_path} using {config.model.model_name} …")
        grid, raw = extract_from_image(image_path, config)
        if grid is None:
            print(
                "Extraction failed — could not parse model response.", file=sys.stderr
            )
            print("Raw model output:", file=sys.stderr)
            print(raw, file=sys.stderr)
            return 1
        print(json.dumps(grid.to_dict(), indent=2, default=str))
        return 0

    if command == "evaluate":
        # Optional args: --model <name>  --limit N  --tolerance F
        #                --shard N  --total-shards M  --output-file PATH
        limit: int | None = None
        tolerance = 0.005
        shard: int | None = None
        total_shards: int | None = None
        output_file: Path | None = None
        remaining = _parse_model_flag(list(args[1:]), config)
        while remaining:
            flag = remaining.pop(0)
            if flag == "--limit" and remaining:
                limit = int(remaining.pop(0))
            elif flag == "--tolerance" and remaining:
                tolerance = float(remaining.pop(0))
            elif flag == "--shard" and remaining:
                shard = int(remaining.pop(0))
            elif flag == "--total-shards" and remaining:
                total_shards = int(remaining.pop(0))
            elif flag == "--output-file" and remaining:
                output_file = Path(remaining.pop(0))
        if (shard is None) != (total_shards is None):
            print("--shard and --total-shards must be used together", file=sys.stderr)
            return 1
        print(f"Evaluating with model: {config.model.model_name}")
        if limit:
            print(f"Limiting to {limit} images")
        if shard is not None:
            print(f"Shard {shard}/{total_shards}")
        report = run_evaluation(
            config,
            limit=limit,
            tolerance=tolerance,
            shard=shard,
            total_shards=total_shards,
        )
        summary = report.summary()
        print(json.dumps(summary, indent=2, default=str))
        if report.comparisons:
            print("\nPer-image results:")
            for c in report.comparisons:
                print(json.dumps(c.summary(), indent=2, default=str))
        if output_file is not None:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "summary": summary,
                "comparisons": [c.summary() for c in report.comparisons],
            }
            if shard is not None:
                payload["shard"] = shard
                payload["total_shards"] = total_shards
            output_file.write_text(json.dumps(payload, indent=2, default=str))
            print(f"\nReport written to: {output_file}")
        return 0

    if command == "finetune":
        # Optional args: --model <name>  --epochs N  --output-dir PATH  --eval-split F
        #                --lora-r N  --report-to wandb|tensorboard|none
        remaining = _parse_model_flag(list(args[1:]), config)
        while remaining:
            flag = remaining.pop(0)
            if flag == "--epochs" and remaining:
                config.training.epochs = int(remaining.pop(0))
            elif flag == "--output-dir" and remaining:
                from pathlib import Path as _Path

                config.training.output_dir = _Path(remaining.pop(0))
            elif flag == "--eval-split" and remaining:
                config.training.eval_split = float(remaining.pop(0))
            elif flag == "--lora-r" and remaining:
                config.training.lora_r = int(remaining.pop(0))
            elif flag == "--report-to" and remaining:
                config.training.report_to = remaining.pop(0)
        print(f"Fine-tuning model: {config.model.model_name}")
        print(f"Epochs: {config.training.epochs}  |  LoRA r={config.training.lora_r}")
        print(f"Reporting to: {config.training.report_to}")
        output_dir = run_finetune(config)
        print(f"Adapter saved to: {output_dir}")
        return 0

    if command == "visualize":
        # Usage: visualize [--model X] [--output PATH] [--ground-truth PATH] <image_path>
        output_path: Path | None = None
        gt_path: Path | None = None
        use_model = False
        remaining = list(args[1:])
        i = 0
        filtered: list[str] = []
        while i < len(remaining):
            if remaining[i] == "--model" and i + 1 < len(remaining):
                value = remaining[i + 1]
                config.model.model_name = MODEL_PRESETS.get(value, value)
                use_model = True
                i += 2
            elif remaining[i] == "--output" and i + 1 < len(remaining):
                output_path = Path(remaining[i + 1])
                i += 2
            elif remaining[i] == "--ground-truth" and i + 1 < len(remaining):
                gt_path = Path(remaining[i + 1])
                i += 2
            else:
                filtered.append(remaining[i])
                i += 1

        if not filtered:
            print(
                "Usage: visualize [--model X] [--output PATH] [--ground-truth PATH] <image_path>",
                file=sys.stderr,
            )
            return 1

        image_path = Path(filtered[0])
        if not image_path.exists():
            print(f"Image not found: {image_path}", file=sys.stderr)
            return 1

        # Derive output path from image stem if not given
        if output_path is None:
            output_path = (
                Path("outputs") / "figures" / (image_path.stem + "_figure.png")
            )

        # Load ground truth if provided, or look for a sibling JSON
        from weather_doc_extractor.ingest import load_grid

        ground_truth = None
        if gt_path is not None:
            ground_truth = load_grid(gt_path)
        else:
            sibling = (
                image_path.with_suffix(".json").parent.parent
                / "transcriptions"
                / image_path.with_suffix(".json").name
            )
            if sibling.exists():
                ground_truth = load_grid(sibling)

        # Run extraction or use ground truth as the displayed grid
        if use_model:
            print(f"Extracting from {image_path} using {config.model.model_name} …")
            grid, _ = extract_from_image(image_path, config)
            if grid is None:
                print("Extraction failed — cannot build figure.", file=sys.stderr)
                return 1
        elif ground_truth is not None:
            grid = ground_truth
            ground_truth = None  # nothing to compare against
            print("No --model given; displaying ground-truth data.")
        else:
            print(
                "Provide --model to run extraction, or --ground-truth to display known data.",
                file=sys.stderr,
            )
            return 1

        from weather_doc_extractor.visualize import save_figure

        saved = save_figure(
            image_path,
            grid,
            output_path,
            ground_truth=ground_truth,
            model_name=config.model.model_name if use_model else None,
        )
        print(f"Figure saved to: {saved}")
        return 0

    if command == "batch-extract":
        # Usage: batch-extract [--model X] [--output-dir PATH]
        #                      [--shard N --total-shards M]
        output_dir: Path = config.paths.outputs_dir / "extractions"
        shard: int | None = None
        total_shards: int | None = None
        remaining = _parse_model_flag(list(args[1:]), config)
        while remaining:
            flag = remaining.pop(0)
            if flag == "--output-dir" and remaining:
                output_dir = Path(remaining.pop(0))
            elif flag == "--shard" and remaining:
                shard = int(remaining.pop(0))
            elif flag == "--total-shards" and remaining:
                total_shards = int(remaining.pop(0))
        if (shard is None) != (total_shards is None):
            print("--shard and --total-shards must be used together", file=sys.stderr)
            return 1
        print(f"Batch extraction with model: {config.model.model_name}")
        print(f"Writing results to: {output_dir}")
        if shard is not None:
            print(f"Shard {shard}/{total_shards}")
        summary = run_batch_extract(
            config,
            output_dir=output_dir,
            shard=shard,
            total_shards=total_shards,
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0

    print(f"Unknown command: {command}")
    print(
        "Available commands: info, stages, ingest, extract, batch-extract, "
        "evaluate, finetune, visualize"
    )
    return 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))


def main() -> None:
    """Console-script entry point — reads sys.argv."""
    sys.exit(run(sys.argv[1:]))
