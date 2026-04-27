# HPC and Azure Batch

This guide covers running the weather document extraction pipeline at scale on
Microsoft Azure, using Azure Batch for parallel evaluation and GPU-accelerated
fine-tuning.

## Overview

The pipeline has three compute-intensive stages suited to HPC:

| Stage | Parallelism | Recommended resource |
|-------|-------------|----------------------|
| `batch-extract` | Embarrassingly parallel — each image is independent | GPU job array |
| `evaluate` | Embarrassingly parallel — each image is independent | CPU or GPU job array |
| `finetune` | Single training run, benefits from multi-GPU | GPU node (multi-GPU) |

---

## Environment variables

All paths and key hyper-parameters can be overridden via environment variables.
This lets Azure Batch tasks pick up the right mounted storage paths without
modifying source code.

| Variable | Overrides | Default |
|---|---|---|
| `WEATHER_DATA_DIR` | `paths.data_dir` | `data` |
| `WEATHER_OUTPUT_DIR` | `paths.outputs_dir` | `outputs` |
| `WEATHER_MODELS_DIR` | `paths.models_dir` | `models` |
| `WEATHER_IMAGES_DIR` | `ingest.images_dir` | `Daily_rainfall_sample/images` |
| `WEATHER_TRANSCRIPTIONS_DIR` | `ingest.transcriptions_dir` | `Daily_rainfall_sample/transcriptions` |
| `WEATHER_INGEST_OUTPUT_DIR` | `ingest.output_dir` | `data/dataset` |
| `WEATHER_TRAINING_OUTPUT_DIR` | `training.output_dir` | `outputs/checkpoints` |
| `WEATHER_MODEL` | `model.model_name` | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| `WEATHER_DEVICE` | `model.device` | `auto` |
| `WEATHER_EPOCHS` | `training.epochs` | `3` |
| `WEATHER_BATCH_SIZE` | `training.batch_size` | `1` |
| `WEATHER_GRAD_ACCUM_STEPS` | `training.gradient_accumulation_steps` | `8` |
| `WEATHER_LEARNING_RATE` | `training.learning_rate` | `0.0002` |
| `WEATHER_REPORT_TO` | `training.report_to` | `none` |
| `HF_HOME` | HuggHuggingFaceingFace model cache | (HuggingFace default) |

These are read at `AppConfig` construction time, so they apply to every CLI
command without any extra flags.

---

## Mounting Azure Blob Storage

Use [blobfuse2](https://github.com/Azure/azure-storage-fuse) to mount a Blob
Storage container as a local filesystem on each Batch node.

```bash
# Install blobfuse2 on Ubuntu 22.04
wget https://packages.microsoft.com/repos/azure-cli/pool/main/b/blobfuse2/blobfuse2_2.3.0_ubuntu22.04_amd64.deb
sudo dpkg -i blobfuse2_2.3.0_ubuntu22.04_amd64.deb

# Mount the container
sudo mkdir -p /mnt/blob
blobfuse2 mount /mnt/blob \
    --container-name <container> \
    --account-name <storage-account> \
    --use-adls=false \
    --tmp-path=/mnt/blobfuse-cache
```

Then point the pipeline at the mount:

```bash
export WEATHER_IMAGES_DIR=/mnt/blob/Daily_rainfall_sample/images
export WEATHER_TRANSCRIPTIONS_DIR=/mnt/blob/Daily_rainfall_sample/transcriptions
export WEATHER_OUTPUT_DIR=/mnt/blob/outputs
export HF_HOME=/mnt/blob/hf_cache
```

---

## Bulk extraction with job arrays

`batch-extract` runs the model over every image in `WEATHER_IMAGES_DIR` and
writes one `<stem>.json` per image to an output directory.  Ground-truth
transcriptions are **not** required — this is the right command when you want
to extract data from new, unannotated images.

### Output format

Each file contains:

```json
{
  "stem": "DRain_1871-1880_Cornwall-59",
  "parse_failed": false,
  "grid": {
    "days": {"Day 1": [0.12, null, ...], ...},
    "totals": [1.5, ...]
  }
}
```

If the model response could not be parsed, `parse_failed` is `true` and a
`raw_text` field contains the raw model output for debugging.

### Local test

```bash
weather-extract batch-extract \
    --model smolvlm \
    --output-dir outputs/extractions
```

### Sharded job array (Azure CLI)

```bash
POOL_ID=gpu-pool
JOB_ID=extract-$(date +%Y%m%d-%H%M%S)
TOTAL_SHARDS=8

az batch job create \
    --id "$JOB_ID" \
    --pool-id "$POOL_ID"

for i in $(seq 1 "$TOTAL_SHARDS"); do
    az batch task create \
        --job-id "$JOB_ID" \
        --task-id "extract-shard-$i" \
        --command-line "/bin/bash \$AZ_BATCH_NODE_SHARED_DIR/scripts/azure_extract_array.sh" \
        --environment-settings \
            "AZ_BATCH_TASK_ID=$((i - 1))" \
            "TOTAL_SHARDS=$TOTAL_SHARDS" \
            "WEATHER_IMAGES_DIR=/mnt/blob/Daily_rainfall_sample/images" \
            "WEATHER_TRANSCRIPTIONS_DIR=/mnt/blob/Daily_rainfall_sample/transcriptions" \
            "WEATHER_OUTPUT_DIR=/mnt/blob/outputs" \
            "WEATHER_MODEL=smolvlm" \
            "HF_HOME=/mnt/blob/hf_cache"
done
```

The script `scripts/azure_extract_array.sh` wraps this automatically when
used with native Azure Batch task arrays (where `$AZ_BATCH_TASK_ID` is
injected by the service).

### Collecting results

After all tasks finish, the output directory contains one JSON per image.
Load them all:

```python
import json, pathlib

results = [
    json.loads(p.read_text())
    for p in sorted(pathlib.Path("outputs/extractions").glob("*.json"))
]
succeeded = [r for r in results if not r["parse_failed"]]
failed    = [r for r in results if r["parse_failed"]]
print(f"Extracted: {len(succeeded)}  Failed: {len(failed)}")
```

---

## Parallel evaluation with job arrays

The `evaluate` command accepts `--shard N --total-shards M` to process the
`N`-th of `M` equal-sized slices of the paired records.  This maps directly
to an Azure Batch job array where each task sets `--shard` to its (1-based)
task index.

### Local test

```bash
# Split 100 records across 4 shards — run locally to verify
weather-extract evaluate \
    --shard 1 --total-shards 4 \
    --output-file outputs/eval/shard_1_of_4.json
```

### Submit a job array (Azure CLI)

```bash
POOL_ID=cpu-pool
JOB_ID=eval-$(date +%Y%m%d-%H%M%S)
TOTAL_SHARDS=8

az batch job create \
    --id "$JOB_ID" \
    --pool-id "$POOL_ID"

for i in $(seq 1 "$TOTAL_SHARDS"); do
    az batch task create \
        --job-id "$JOB_ID" \
        --task-id "eval-shard-$i" \
        --command-line "/bin/bash \$AZ_BATCH_NODE_SHARED_DIR/scripts/azure_evaluate_array.sh" \
        --environment-settings \
            "AZ_BATCH_TASK_ID=$((i - 1))" \
            "TOTAL_SHARDS=$TOTAL_SHARDS" \
            "WEATHER_IMAGES_DIR=/mnt/blob/Daily_rainfall_sample/images" \
            "WEATHER_TRANSCRIPTIONS_DIR=/mnt/blob/Daily_rainfall_sample/transcriptions" \
            "WEATHER_OUTPUT_DIR=/mnt/blob/outputs/eval" \
            "WEATHER_MODEL=smolvlm" \
            "HF_HOME=/mnt/blob/hf_cache"
done
```

The script `scripts/azure_evaluate_array.sh` wraps this automatically when
used with native Azure Batch task arrays (where `$AZ_BATCH_TASK_ID` is
injected by the service).

### Aggregating shard results

Each shard writes a JSON file containing its `summary` and `comparisons`.
Combine them with a simple script:

```python
import json, pathlib, statistics

shards = [json.loads(p.read_text()) for p in pathlib.Path("outputs/eval").glob("shard_*.json")]
all_comparisons = [c for s in shards for c in s["comparisons"]]

accuracies = [c["accuracy"] for c in all_comparisons if not c["parse_failed"]]
print(f"Images evaluated: {len(all_comparisons)}")
print(f"Macro accuracy:   {statistics.mean(accuracies):.1%}")
```

---

## Fine-tuning on a GPU node

### Single GPU

```bash
export WEATHER_IMAGES_DIR=/mnt/blob/Daily_rainfall_sample/images
export WEATHER_TRANSCRIPTIONS_DIR=/mnt/blob/Daily_rainfall_sample/transcriptions
export WEATHER_TRAINING_OUTPUT_DIR=/mnt/blob/outputs/checkpoints
export HF_HOME=/mnt/blob/hf_cache
export WEATHER_REPORT_TO=wandb   # or tensorboard, or none

weather-extract finetune --model smolvlm --epochs 5
```

### Multi-GPU with Accelerate

`scripts/accelerate_config.yaml` provides a ready-to-use configuration for
single-node multi-GPU training (4× GPU by default — adjust `num_processes`).

```bash
export ACCELERATE_CONFIG=$PWD/scripts/accelerate_config.yaml
bash scripts/azure_finetune.sh
```

Or directly:

```bash
accelerate launch \
    --config_file scripts/accelerate_config.yaml \
    -m weather_doc_extractor.cli \
    finetune --model smolvlm --epochs 5 --report-to tensorboard
```

### Recommended Azure VM SKUs

| SKU | GPUs | Use case |
|-----|------|----------|
| `Standard_NC6s_v3` | 1× V100 16 GB | Development / small runs |
| `Standard_NC24s_v3` | 4× V100 16 GB | Multi-GPU fine-tuning |
| `Standard_NC96ads_A100_v4` | 4× A100 80 GB | Large model fine-tuning |
| `Standard_ND96asr_v4` | 8× A100 40 GB | Largest scale |

---

## Node setup

`scripts/setup_env.sh` installs the Conda environment on a fresh Batch node.
Call it as the pool start task or at the top of each job script:

```bash
bash scripts/setup_env.sh
```

It reads `CONDA_HOME`, `REPO_DIR`, `CONDA_ENV_NAME`, and `HF_HOME` from the
environment, with sensible defaults.
