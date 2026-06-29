from dataclasses import asdict
from pathlib import Path
from typing import Any

from weather_doc_extractor.config import AppConfig
from weather_doc_extractor.evaluate import EvaluationReport, evaluate_dataset
from weather_doc_extractor.ingest import scan_records
from weather_doc_extractor.schemas import DailyRainfallGrid


def build_project_summary(config: AppConfig) -> dict[str, object]:
    """Return a serializable summary of the current project configuration."""
    return {
        "paths": asdict(config.paths),
        "ingest": asdict(config.ingest),
        "model": asdict(config.model),
        "training": asdict(config.training),
    }


def run_ingest(config: AppConfig) -> dict[str, object]:
    """Scan the configured directories and return an ingest summary."""
    records = scan_records(config.ingest.images_dir, config.ingest.transcriptions_dir)
    paired = sum(1 for r in records if r.grid is not None)
    counties = sorted({r.county for r in records})
    decades = sorted({r.decade for r in records})
    return {
        "total": len(records),
        "paired": paired,
        "unpaired": len(records) - paired,
        "counties": counties,
        "decades": decades,
    }


def run_evaluation(
    config: AppConfig,
    limit: int | None = None,
    tolerance: float = 0.005,
    shard: int | None = None,
    total_shards: int | None = None,
) -> EvaluationReport:
    """Run the model over paired records and return an EvaluationReport.

    Parameters
    ----------
    shard:
        1-based index of this shard (e.g. 1 of 4).  When provided together
        with *total_shards* the paired records are evenly partitioned and
        only this shard's slice is evaluated.  Useful for Azure Batch job
        arrays where each task sets ``--shard $AZ_BATCH_TASK_ID`` and
        ``--total-shards <pool_size>``.
    total_shards:
        Total number of shards.  Must be provided when *shard* is set.
    """
    records = scan_records(config.ingest.images_dir, config.ingest.transcriptions_dir)
    paired = [r for r in records if r.grid is not None]
    if shard is not None and total_shards is not None:
        paired = _shard_list(paired, shard, total_shards)
        records = paired  # evaluate_dataset will re-filter, so pass paired directly
    return evaluate_dataset(records, config, tolerance=tolerance, limit=limit)


def _shard_list(items: list, shard: int, total_shards: int) -> list:
    """Return the slice of *items* belonging to 1-based *shard* of *total_shards*."""
    if total_shards < 1:
        raise ValueError(f"total_shards must be >= 1, got {total_shards}")
    if not (1 <= shard <= total_shards):
        raise ValueError(f"shard must be in 1..{total_shards}, got {shard}")
    indices = range(shard - 1, len(items), total_shards)
    return [items[i] for i in indices]


def extract_from_image(
    image_path: Path,
    config: AppConfig,
) -> tuple[DailyRainfallGrid | None, str]:
    """Run the configured VLM over *image_path* and return ``(grid, raw_text)``.

    Delegates to :func:`~weather_doc_extractor.inference.extract_grid`.
    """
    from weather_doc_extractor.inference import extract_grid

    return extract_grid(image_path, config.model)


def describe_ingest_stage() -> str:
    return (
        "Ingest document images, align annotations, and normalize them into "
        "training-ready records."
    )


def describe_inference_stage() -> str:
    return (
        "Run a vision-language mini-LLM over each document image and map the "
        "response into the weather extraction schema."
    )


def run_finetune(config: AppConfig) -> Path:
    """Scan paired records and fine-tune the configured model via LoRA + SFT.

    Returns the directory where the LoRA adapter was saved.
    """
    from weather_doc_extractor.finetune import run_finetune as _run_finetune

    records = scan_records(config.ingest.images_dir, config.ingest.transcriptions_dir)
    return _run_finetune(records, config.model, config.training)


def run_finetune_consensus(config: AppConfig) -> Path:
    """Run additive consensus-only fine-tuning pathway.

    This path is separate from :func:`run_finetune` and is intended for
    consensus transcriptions with strict token masking.
    """
    from weather_doc_extractor.finetune import (
        run_finetune_consensus as _run_finetune_consensus,
    )

    records = scan_records(config.ingest.images_dir, config.ingest.transcriptions_dir)
    return _run_finetune_consensus(records, config.model, config.training)


def describe_training_stage() -> str:
    return (
        "Fine-tune the selected model with TRL using labeled extraction "
        "examples and evaluate field-level accuracy."
    )


def run_batch_extract(
    config: AppConfig,
    output_dir: Path,
    shard: int | None = None,
    total_shards: int | None = None,
    limit: int | None = None,
    batch_size: int = 10,
    retry_batch_size: int | None = None,
) -> dict[str, object]:
    """Run inference on every image in the configured images directory.

    Writes one ``<stem>.json`` per image to *output_dir*.  Images where the
    model response cannot be parsed are recorded with ``"parse_failed": true``
    and no grid data.

    Parameters
    ----------
    output_dir:
        Directory to write per-image JSON result files.
    shard / total_shards:
        When both are provided, only the ``shard``-th (1-based) of
        ``total_shards`` equal slices of the image list is processed.
        Allows parallel execution via an Azure Batch job array.
    limit:
        If set, process only the first *limit* images (after sharding).
        Useful for quick smoke tests.
    batch_size:
        Number of images to process in each forward pass.  Larger batches
        improve GPU utilisation but require more VRAM.  Default: 10.
    retry_batch_size:
        Number of images to process together during stage-2 retry for parse
        failures. If unset, defaults to ``min(batch_size, 4)``.

    Returns
    -------
    dict
        Summary with keys ``total``, ``succeeded``, ``failed``, ``output_dir``.
    """
    from weather_doc_extractor.inference import (
        _load_model_and_processor,
        detect_model_family,
        extract_grid_with_model,
        extract_grid_batch_with_model,
    )

    import json

    records = scan_records(config.ingest.images_dir, config.ingest.transcriptions_dir)
    if shard is not None and total_shards is not None:
        records = _shard_list(records, shard, total_shards)
    if limit is not None:
        records = records[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out records that already have output files (supports resumption after preemption)
    skipped = 0
    records_to_process = []
    for record in records:
        out_path = output_dir / f"{record.stem}.json"
        if out_path.exists():
            skipped += 1
        else:
            records_to_process.append(record)

    if skipped > 0:
        print(
            f"[resumption] Skipping {skipped} images with existing output files",
            flush=True,
        )

    succeeded = 0
    failed = 0
    recovered_by_batch_retry = 0
    recovered_by_single_retry = 0

    print(f"Loading model {config.model.model_name} …", flush=True)
    processor, model = _load_model_and_processor(config.model)
    family = detect_model_family(config.model.model_name)
    print(
        f"Processing {len(records_to_process)} images in batches of {batch_size} …",
        flush=True,
    )

    first_pass_failures: list[tuple[Any, str]] = []

    for batch_start in range(0, len(records_to_process), batch_size):
        batch = records_to_process[batch_start : batch_start + batch_size]
        batch_end = batch_start + len(batch)
        print(
            f"  [{batch_end}/{len(records_to_process)}] "
            f"images {batch_start + 1}–{batch_end} …",
            flush=True,
        )
        results = extract_grid_batch_with_model(
            [r.image_path for r in batch], config.model, processor, model, family
        )
        for record, (grid, raw_text) in zip(batch, results):
            if grid is not None:
                result = {
                    "stem": record.stem,
                    "parse_failed": False,
                    "grid": grid.to_dict(),
                }
                succeeded += 1
                out_path = output_dir / f"{record.stem}.json"
                out_path.write_text(json.dumps(result, indent=2, default=str))
            else:
                first_pass_failures.append((record, raw_text))

    second_pass_failures: list[tuple[Any, str, str]] = []
    retry_batch_size = (
        max(1, retry_batch_size)
        if retry_batch_size is not None
        else max(1, min(batch_size, 4))
    )
    if first_pass_failures:
        print(
            f"[retry-2] Reprocessing {len(first_pass_failures)} parse failures "
            f"in batches of {retry_batch_size} …",
            flush=True,
        )
    for batch_start in range(0, len(first_pass_failures), retry_batch_size):
        chunk = first_pass_failures[batch_start : batch_start + retry_batch_size]
        chunk_records = [item[0] for item in chunk]
        chunk_first_raw = [item[1] for item in chunk]
        results = extract_grid_batch_with_model(
            [r.image_path for r in chunk_records],
            config.model,
            processor,
            model,
            family,
        )
        for record, first_raw, (grid, retry_batch_raw) in zip(
            chunk_records, chunk_first_raw, results
        ):
            if grid is not None:
                result = {
                    "stem": record.stem,
                    "parse_failed": False,
                    "grid": grid.to_dict(),
                    "recovered_by_batch_retry": True,
                }
                succeeded += 1
                recovered_by_batch_retry += 1
                out_path = output_dir / f"{record.stem}.json"
                out_path.write_text(json.dumps(result, indent=2, default=str))
                print(f"    INFO: recovered parse for {record.stem} via batched retry")
            else:
                second_pass_failures.append((record, first_raw, retry_batch_raw))

    if second_pass_failures:
        print(
            f"[retry-3] Reprocessing {len(second_pass_failures)} remaining parse failures "
            f"one-by-one …",
            flush=True,
        )
    for record, first_raw, retry_batch_raw in second_pass_failures:
        retry_grid, retry_raw = extract_grid_with_model(
            record.image_path,
            config.model,
            processor,
            model,
            family,
        )
        if retry_grid is not None:
            result = {
                "stem": record.stem,
                "parse_failed": False,
                "grid": retry_grid.to_dict(),
                "recovered_by_single_retry": True,
            }
            succeeded += 1
            recovered_by_single_retry += 1
            print(f"    INFO: recovered parse for {record.stem} via single-image retry")
        else:
            result = {
                "stem": record.stem,
                "parse_failed": True,
                "raw_text": first_raw,
                "retry_batch_raw_text": retry_batch_raw,
                "retry_raw_text": retry_raw,
            }
            failed += 1
            print(f"    WARNING: parse failed for {record.stem}")
        out_path = output_dir / f"{record.stem}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))

    recovered_by_retry = recovered_by_batch_retry + recovered_by_single_retry

    return {
        "total": len(records),
        "skipped": skipped,
        "processed": len(records_to_process),
        "recovered_by_batch_retry": recovered_by_batch_retry,
        "recovered_by_single_retry": recovered_by_single_retry,
        "recovered_by_retry": recovered_by_retry,
        "succeeded": succeeded,
        "failed": failed,
        "output_dir": str(output_dir),
    }
