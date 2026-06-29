# HPC and Azure ML

This guide covers running the weather document extraction pipeline at scale on
Microsoft Azure, using Azure ML workspaces for parallel evaluation and
GPU-accelerated fine-tuning.

## Overview

The pipeline has three compute-intensive stages suited to HPC:

| Stage | Parallelism | Recommended resource |
|-------|-------------|----------------------|
| `batch-extract` | Embarrassingly parallel — each image is independent | GPU job array |
| `evaluate` | Embarrassingly parallel — each image is independent | CPU or GPU job array |
| `finetune` | Single training run on one GPU | Single A100 80 GB node |

---

## Prerequisites

1. **Azure CLI** with the **ML extension**:

    ```bash
    az extension add -n ml
    az login
    ```

2. **Workspace identifiers** — you need three values:

    | Value | Where to find it |
    |-------|-----------------|
    | Subscription ID | `az account list --query "[].{name:name,id:id}" -o table` |
    | Resource group | Azure ML Studio → workspace overview |
    | Workspace name | Azure ML Studio → workspace overview |

3. **Compute cluster** — create a GPU cluster in the workspace if you don't already have one:

    ```bash
    az ml compute create \
        --name gpu-cluster \
        --type amlcompute \
        --min-instances 0 \
        --max-instances 8 \
        --size Standard_NC6s_v3 \
        --workspace-name <workspace> \
        --resource-group <resource-group> \
        --subscription <subscription>
    ```

---

## Configuration file

Copy the template and fill in your values — you only need to do this once:

```bash
cp azureml/config.env.example azureml/config.env
```

Then edit `azureml/config.env`:

```bash
# Workspace coordinates
AML_SUBSCRIPTION=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
AML_RESOURCE_GROUP=my-resource-group
AML_WORKSPACE=my-aml-workspace

# Compute cluster name
AML_COMPUTE=gpu-cluster

# Base datastore URI — the default store is created automatically with the workspace
AML_DATASTORE_BASE=azureml://datastores/workspaceblobstore/paths

# Input paths (relative to AML_DATASTORE_BASE)
AML_IMAGES_PATH=Daily_rainfall_sample/images
AML_TRANSCRIPTIONS_PATH=Daily_rainfall_sample/transcriptions

# Output root (relative to AML_DATASTORE_BASE)
# Jobs write to sub-paths: outputs/extractions, outputs/eval, outputs/checkpoints
AML_OUTPUTS_PATH=outputs
```

`azureml/config.env` is gitignored so your subscription ID is never committed.
CLI flags (`--subscription`, `--workspace`, etc.) override any value in the file.

---

## Register the Azure ML environment (once)

Register the environment before submitting jobs for the first time:

```bash
bash scripts/aml_submit.sh env
```

Two environment variants are available; set `AML_ENV_VARIANT` in `azureml/config.env`:

| Variant | PyTorch | CUDA | Use case |
|---------|---------|------|----------|
| `v100` | 2.4 | 12.1 | SmolVLM, Granite |
| `a100` | 2.8 | 12.6 | Gemma 3/4, Ministral (requires `torch>=2.6`) |

Edit `azureml/environment-<variant>.yml` and `azureml/conda-<variant>.yml` to
change the base image or add dependencies.

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
| `HF_HOME` | HuggingFace model cache | (HuggingFace default) |

These are read at `AppConfig` construction time, so they apply to every CLI
command without any extra flags.  In Azure ML jobs they are set via the
`environment_variables` section of the job YAML and can be overridden per
submission with `--set environment_variables.VAR=value`.

---

## Data paths and outputs

Each job YAML in `azureml/` declares typed `inputs` and `outputs`:

- **Inputs** (`type: uri_folder`, `mode: ro_mount`) — Azure ML mounts your
  datastore paths read-only on the compute node.  The default paths point at
  `workspaceblobstore` (the storage account that every workspace creates
  automatically).  Override them with `--set`:

  ```bash
  az ml job create --file azureml/extract_job.yml ... \
      --set inputs.images_dir.path="azureml://datastores/mystore/paths/images"
  ```

- **Outputs** (`type: uri_folder`, `mode: rw_mount`) — Azure ML mounts a
  writable directory backed by `workspaceblobstore`.  When the job finishes,
  the files are **automatically uploaded and persisted**.  You can access them
  via:
  - **Studio UI**: job → **Outputs + logs** tab
  - **Azure CLI**: `az ml job download --name <job-id> --output-name <name> -w <ws> -g <rg>`
  - As inputs to a subsequent job

So you don't need to configure blobfuse2 or worry about ephemeral compute disk
— results are safely stored in the workspace datastore automatically.

### Changing the datastore

The default `workspaceblobstore` is created automatically with every workspace.
To use a different registered datastore:

```bash
--set inputs.images_dir.path="azureml://datastores/<datastore-name>/paths/<path>"
```

### Mounting via blobfuse2 (advanced)

If you need to access data that is not in a registered datastore, you can mount
a Blob Storage container directly using blobfuse2 and pass the mount path as
an environment variable override.  See the
[blobfuse2 documentation](https://github.com/Azure/azure-storage-fuse) for
installation and mount instructions.

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

### Sharded job array (Azure ML)

Use `scripts/aml_submit.sh` — it reads workspace coordinates and paths from
`azureml/config.env` automatically:

```bash
bash scripts/aml_submit.sh --total-shards 8 extract
```

The script prints the resolved workspace, compute, and data paths before
submitting so you can verify them.  Override any value on the command line:

```bash
bash scripts/aml_submit.sh --total-shards 8 --compute big-gpu-cluster extract
```

For multi-GPU nodes (for example `Standard_ND96amsr_A100_v4` with 8 GPUs),
you can run one extraction worker per GPU inside each AML job:

```bash
bash scripts/aml_submit.sh --total-shards 1 --node-gpu-workers 8 extract
```

This keeps a single AML job per model while saturating all 8 GPUs on the node.

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

### Submit a job array (Azure ML)

```bash
bash scripts/aml_submit.sh --total-shards 8 evaluate
```

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

### Submit the fine-tuning job (Azure ML)

```bash
bash scripts/aml_submit.sh finetune
```

Override training hyper-parameters with `--set`:

```bash
az ml job create \
    --file azureml/finetune_job.yml \
    --workspace-name "$AML_WORKSPACE" \
    --resource-group "$AML_RESOURCE_GROUP" \
    --subscription "$AML_SUBSCRIPTION" \
    --set environment_variables.WEATHER_EPOCHS=5 \
    --set environment_variables.WEATHER_REPORT_TO=wandb
```

Edit `azureml/finetune_job.yml` to choose your compute cluster name and data
paths.  The job now auto-launches with `accelerate` when
`WEATHER_NUM_PROCESSES > 1`.

For `Standard_ND96amsr_A100_v4`, set 8 processes to use all GPUs:

```bash
bash scripts/aml_submit.sh --compute A100x8 --finetune-gpu-workers 8 finetune
```

Use `--finetune-gpu-workers 1` if you want single-GPU fine-tuning.

By default, gradient accumulation is auto-scaled by world size to keep global
batch size approximately stable across 1-GPU and multi-GPU runs.
Override if needed:

```bash
bash scripts/aml_submit.sh \
    --compute A100x8 \
    --finetune-gpu-workers 8 \
    --grad-accum-steps 8 \
    --auto-scale-grad-accum true \
    finetune
```

### Local / single-GPU run

```bash
export WEATHER_IMAGES_DIR=Daily_rainfall_sample/images
export WEATHER_TRANSCRIPTIONS_DIR=Daily_rainfall_sample/transcriptions
export WEATHER_TRAINING_OUTPUT_DIR=outputs/checkpoints
export HF_TOKEN=<your-token>   # required for Gemma/Ministral

weather-extract finetune --model smolvlm --epochs 5
```

### Recommended Azure VM SKUs

| SKU | GPUs | Use case |
|-----|------|----------|
| `Standard_NC6s_v3` | 1× V100 16 GB | SmolVLM / Granite development runs |
| `Standard_NC96ads_A100_v4` | 4× A100 80 GB | Gemma / Ministral; use single GPU per job |
| `Standard_ND96asr_v4` | 8× A100 40 GB | Largest scale |

---

## Granite4 validation gate (Azure)

When local testing is constrained by RAM or missing GPUs, run the Granite 4.1
validation gate in Azure before merging model-related changes.

This gate submits both required jobs:

1. Granite4 extraction smoke test
2. Granite4 fine-tuning smoke test

```bash
bash scripts/azure_validate_granite4.sh --limit 2 --env-variant a100
```

The helper script wraps `scripts/aml_submit.sh` and enforces `--model granite4`
for both submissions.

Required checks after submission:

1. Extraction job completes and produces parseable JSON output.
2. Fine-tune job completes and writes adapter checkpoints.
3. Post-finetune extraction succeeds when the adapter path is passed to `--model`.
4. One regression extraction with `--model granite` still succeeds (3.2 compatibility).

---

## Node setup (Azure Batch)

`scripts/setup_env.sh` installs the Conda environment on a fresh Azure Batch node.
Call it as the pool start task or at the top of each job script:

```bash
bash scripts/setup_env.sh
```

It reads `CONDA_HOME`, `REPO_DIR`, `CONDA_ENV_NAME`, and `HF_HOME` from the
environment, with sensible defaults.  This is not needed for Azure ML jobs —
the environment is managed by the `azureml/environment.yml` definition.
