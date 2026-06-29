from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
    run_finetune_consensus,
    run_ingest,
)

_ROW_LABELS = [f"Day {i}" for i in range(1, 32)] + ["Totals"]


def _load_consensus_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _consensus_to_grid_and_mask(
    consensus: dict[str, Any],
) -> tuple["DailyRainfallGrid", list[list[bool]]]:
    from weather_doc_extractor.schemas import DailyRainfallGrid

    days: dict[str, list[float | None]] = {}
    correct_mask: list[list[bool]] = []

    for row_label in _ROW_LABELS[:-1]:
        row_data = consensus.get(row_label, [])
        values: list[float | None] = []
        mask_row: list[bool] = []
        for month_idx in range(12):
            cell = (
                row_data[month_idx]
                if isinstance(row_data, list) and month_idx < len(row_data)
                else {}
            )
            if not isinstance(cell, dict):
                cell = {}
            values.append(_coerce_float(cell.get("value")))
            mask_row.append(bool(cell.get("correct", False)))
        days[row_label] = values
        correct_mask.append(mask_row)

    totals_data = consensus.get("Totals", [])
    totals: list[float | None] = []
    totals_mask: list[bool] = []
    for month_idx in range(12):
        cell = (
            totals_data[month_idx]
            if isinstance(totals_data, list) and month_idx < len(totals_data)
            else {}
        )
        if not isinstance(cell, dict):
            cell = {}
        totals.append(_coerce_float(cell.get("value")))
        totals_mask.append(bool(cell.get("correct", False)))
    correct_mask.append(totals_mask)

    return DailyRainfallGrid(days=days, totals=totals), correct_mask


def _colour_consensus_table(fig, correct_mask: list[list[bool]]) -> bool:
    if len(fig.axes) < 2:
        return False
    table_axis = fig.axes[1]
    table_artist = None
    for artist in table_axis.get_children():
        if hasattr(artist, "get_celld"):
            table_artist = artist
            break
    if table_artist is None:
        return False

    for (row, col), cell in table_artist.get_celld().items():
        if row <= 0 or col < 0:
            continue
        cell.get_text().set_color(
            "#1b7837" if correct_mask[row - 1][col] else "#b2182b"
        )
    return True


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


def _resolve_validation_run_name(
    extractions_dir: Path,
    registry_path: Path,
) -> str:
    """Resolve validation output run name from extraction registry when possible."""
    candidate = extractions_dir.name.strip()
    if not candidate:
        candidate = "validation-run"

    if not registry_path.exists():
        return candidate

    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return candidate

    entries = payload.get("extractions", [])
    if not isinstance(entries, list):
        return candidate

    # Most robust path: direct run_name match.
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        run_name = str(entry.get("run_name", "")).strip()
        if run_name and run_name == candidate:
            return run_name

    ext_norm = extractions_dir.as_posix().strip("/")
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        reg_path = str(entry.get("extractions_path", "")).replace("\\", "/").strip("/")
        if not reg_path:
            continue
        if (
            reg_path == ext_norm
            or reg_path.endswith(ext_norm)
            or ext_norm.endswith(reg_path)
        ):
            run_name = str(entry.get("run_name", "")).strip()
            if run_name:
                return run_name

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        reg_path = str(entry.get("extractions_path", "")).replace("\\", "/").strip("/")
        if not reg_path:
            continue
        if Path(reg_path).name == candidate:
            run_name = str(entry.get("run_name", "")).strip()
            if run_name:
                return run_name

    return candidate


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
                "Usage: extract [--model smolvlm|smolvlm2|granite|granite4|<hf-id>] <image_path>",
                file=sys.stderr,
            )
            return 1
        remaining = _parse_model_flag(list(args[1:]), config)
        if not remaining:
            print(
                "Usage: extract [--model smolvlm|smolvlm2|granite|granite4|<hf-id>] <image_path>",
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

    if command == "finetune-consensus":
        # Optional args: --model <checkpoint|name> --epochs N --output-dir PATH
        #                --consensus-dir PATH --eval-split F --lora-r N --report-to X
        remaining = _parse_model_flag(list(args[1:]), config)
        while remaining:
            flag = remaining.pop(0)
            if flag == "--epochs" and remaining:
                config.training.epochs = int(remaining.pop(0))
            elif flag == "--output-dir" and remaining:
                from pathlib import Path as _Path

                config.training.output_dir = _Path(remaining.pop(0))
            elif flag == "--consensus-dir" and remaining:
                from pathlib import Path as _Path

                path = _Path(remaining.pop(0))
                config.ingest.transcriptions_dir = path
                config.training.consensus_transcriptions_dir = path
            elif flag == "--eval-split" and remaining:
                config.training.eval_split = float(remaining.pop(0))
            elif flag == "--lora-r" and remaining:
                config.training.lora_r = int(remaining.pop(0))
            elif flag == "--report-to" and remaining:
                config.training.report_to = remaining.pop(0)

        print(f"Consensus fine-tuning model: {config.model.model_name}")
        print(f"Epochs: {config.training.epochs}  |  LoRA r={config.training.lora_r}")
        print(f"Consensus transcriptions: {config.ingest.transcriptions_dir}")
        print("Loss masking: strict (correct=true consensus cells only)")
        output_dir = run_finetune_consensus(config)
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

    if command == "visualize-consensus":
        # Usage: visualize-consensus [--output PATH] [--consensus PATH]
        #                           [--consensus-dir PATH] <image_path>
        output_path: Path | None = None
        consensus_path: Path | None = None
        consensus_dir: Path | None = None
        remaining = list(args[1:])
        i = 0
        filtered: list[str] = []
        while i < len(remaining):
            if remaining[i] == "--output" and i + 1 < len(remaining):
                output_path = Path(remaining[i + 1])
                i += 2
            elif remaining[i] == "--consensus" and i + 1 < len(remaining):
                consensus_path = Path(remaining[i + 1])
                i += 2
            elif remaining[i] == "--consensus-dir" and i + 1 < len(remaining):
                consensus_dir = Path(remaining[i + 1])
                i += 2
            else:
                filtered.append(remaining[i])
                i += 1

        if not filtered:
            print(
                "Usage: visualize-consensus [--output PATH] [--consensus PATH] "
                "[--consensus-dir PATH] <image_path>",
                file=sys.stderr,
            )
            return 1

        image_path = Path(filtered[0])
        if not image_path.exists():
            print(f"Image not found: {image_path}", file=sys.stderr)
            return 1

        if output_path is None:
            output_path = (
                Path("outputs") / "figures" / (image_path.stem + "_figure.png")
            )

        if consensus_path is None:
            if consensus_dir is not None:
                consensus_path = consensus_dir / f"{image_path.stem}.json"
            else:
                sibling = (
                    image_path.with_suffix(".json").parent.parent
                    / "consensus_transcriptions"
                    / image_path.with_suffix(".json").name
                )
                default_consensus = (
                    Path("outputs")
                    / "consensus_dataset_1000"
                    / "consensus_transcriptions"
                    / image_path.with_suffix(".json").name
                )
                if sibling.exists():
                    consensus_path = sibling
                else:
                    consensus_path = default_consensus

        consensus = _load_consensus_json(consensus_path)
        if consensus is None:
            print(
                f"Consensus file not found or invalid JSON: {consensus_path}",
                file=sys.stderr,
            )
            return 1

        grid, correct_mask = _consensus_to_grid_and_mask(consensus)

        from weather_doc_extractor.visualize import make_figure

        fig = make_figure(
            image_path=image_path,
            grid=grid,
            title=image_path.stem,
            model_name="Consensus",
        )
        if not _colour_consensus_table(fig, correct_mask):
            print(
                "Warning: could not locate table artist; using default text colours",
                file=sys.stderr,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
        print(f"Figure saved to: {output_path}")
        return 0

    if (
        command == "batch-extract"
    ):  # Usage: batch-extract [--model X] [--output-dir PATH]
        #                      [--shard N --total-shards M] [--limit N]
        # Default output: outputs/extractions/<model-slug>/<YYYYMMDD-HHMMSS>/
        # Override with --output-dir to write to a specific path.
        from datetime import datetime, timezone

        output_dir: Path | None = None
        shard: int | None = None
        total_shards: int | None = None
        limit: int | None = None
        batch_size: int = 10
        retry_batch_size: int | None = None
        remaining = _parse_model_flag(list(args[1:]), config)
        while remaining:
            flag = remaining.pop(0)
            if flag == "--output-dir" and remaining:
                output_dir = Path(remaining.pop(0))
            elif flag == "--shard" and remaining:
                shard = int(remaining.pop(0))
            elif flag == "--total-shards" and remaining:
                total_shards = int(remaining.pop(0))
            elif flag == "--limit" and remaining:
                limit = int(remaining.pop(0))
            elif flag == "--batch-size" and remaining:
                batch_size = int(remaining.pop(0))
            elif flag == "--retry-batch-size" and remaining:
                retry_batch_size = int(remaining.pop(0))
        if output_dir is None:
            model_slug = config.model.model_name.split("/")[-1]
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            output_dir = (
                config.paths.outputs_dir / "extractions" / model_slug / timestamp
            )
        if (shard is None) != (total_shards is None):
            print("--shard and --total-shards must be used together", file=sys.stderr)
            return 1
        print(f"Batch extraction with model: {config.model.model_name}")
        print(f"Writing results to: {output_dir}")
        if limit:
            print(f"Limiting to {limit} images")
        if shard is not None:
            print(f"Shard {shard}/{total_shards}")
        summary = run_batch_extract(
            config,
            output_dir=output_dir,
            shard=shard,
            total_shards=total_shards,
            limit=limit,
            batch_size=batch_size,
            retry_batch_size=retry_batch_size,
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if command == "validate":
        # Usage: validate --extractions <dir> --test-data <dir>
        #                 [--output-dir <dir>] [--report-file <path>]
        #                 [--model <name>] [--tolerance F] [--no-figures]
        extractions_dir: Path | None = None
        test_data_dir: Path | None = None
        output_dir: Path | None = None
        report_file: Path | None = None
        tolerance = 0.005
        no_figures = False
        explicit_model: bool = False
        _default_model = config.model.model_name
        remaining = _parse_model_flag(list(args[1:]), config)
        explicit_model = config.model.model_name != _default_model
        while remaining:
            flag = remaining.pop(0)
            if flag == "--extractions" and remaining:
                extractions_dir = Path(remaining.pop(0))
            elif flag == "--test-data" and remaining:
                test_data_dir = Path(remaining.pop(0))
            elif flag == "--output-dir" and remaining:
                output_dir = Path(remaining.pop(0))
            elif flag == "--report-file" and remaining:
                report_file = Path(remaining.pop(0))
            elif flag == "--tolerance" and remaining:
                tolerance = float(remaining.pop(0))
            elif flag == "--no-figures":
                no_figures = True

        if extractions_dir is None or test_data_dir is None:
            print(
                "Usage: validate --extractions <dir> --test-data <dir> "
                "[--output-dir <dir>] [--report-file <path>] "
                "[--model <name>] [--tolerance F] [--no-figures]",
                file=sys.stderr,
            )
            return 1
        if not extractions_dir.exists():
            print(
                f"Extractions directory not found: {extractions_dir}", file=sys.stderr
            )
            return 1
        if not test_data_dir.exists():
            print(f"Test data directory not found: {test_data_dir}", file=sys.stderr)
            return 1

        if output_dir is None:
            run_name = _resolve_validation_run_name(
                extractions_dir,
                Path("outputs") / "extraction_registry.json",
            )
            output_dir = Path("outputs") / "validation" / run_name

        from weather_doc_extractor.validate import (
            generate_figures,
            load_extraction_results,
            print_summary,
            run_validation,
            save_report,
        )

        print(f"Loading extraction results from: {extractions_dir}")
        print(f"Test data directory:             {test_data_dir}")
        print(f"Validation output directory:     {output_dir}")
        records = load_extraction_results(extractions_dir, test_data_dir)
        if not records:
            print("No paired records found — nothing to validate.", file=sys.stderr)
            return 1
        print(f"Loaded {len(records)} validation records.")

        report = run_validation(records, tolerance=tolerance)
        print_summary(report)

        if report_file is None:
            report_file = output_dir / "report.json"
        saved = save_report(report, report_file)
        print(f"\nReport written to: {saved}")

        if not no_figures:
            figures_dir = output_dir / "figures"
            print(f"\nGenerating comparison figures → {figures_dir}")
            if explicit_model:
                model_label = config.model.model_name.split("/")[-1]
            else:
                # Infer from extractions path: .../extractions/<model>/<timestamp>/
                model_label = (
                    extractions_dir.parent.name
                    or config.model.model_name.split("/")[-1]
                )
            generate_figures(
                records,
                report,
                figures_dir,
                model_name=model_label,
                tolerance=tolerance,
            )
        return 0

    print(f"Unknown command: {command}")
    print(
        "Available commands: info, stages, ingest, extract, batch-extract, "
        "evaluate, validate, finetune, visualize, visualize-consensus"
    )
    return 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))


def main() -> None:
    """Console-script entry point — reads sys.argv."""
    sys.exit(run(sys.argv[1:]))
