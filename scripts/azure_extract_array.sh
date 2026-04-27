#!/usr/bin/env bash
# azure_extract_array.sh — Run bulk extraction as an Azure Batch job array.
#
# Each task processes one shard of the images directory and writes one
# <stem>.json per image to a shared output directory.  Run after ingesting
# images; ground-truth transcriptions are not required.
#
# Azure Batch injects $AZ_BATCH_TASK_ID (0-based) automatically.
# This script converts it to the 1-based shard index expected by the CLI.
#
# ── Required environment variables ───────────────────────────────────────────
#   TOTAL_SHARDS              Total number of array tasks (= job array size)
#   WEATHER_IMAGES_DIR        Path to mounted images directory
#   WEATHER_OUTPUT_DIR        Parent output directory (results go to
#                             $WEATHER_OUTPUT_DIR/extractions/)
#
# ── Optional environment variables ───────────────────────────────────────────
#   WEATHER_TRANSCRIPTIONS_DIR  Path to transcriptions (used only for stem
#                               metadata; not required for extraction)
#   WEATHER_MODEL               HuggingFace model ID or preset (default: smolvlm)
#   WEATHER_DEVICE              torch device_map value        (default: auto)
#   HF_HOME                     HuggingFace model cache root
#   EXTRACT_OUTPUT_DIR          Override output directory for extracted JSON files
#                               (default: $WEATHER_OUTPUT_DIR/extractions)
#   CONDA_HOME                  Conda installation root       (default: ~/miniconda3)
#   CONDA_ENV_NAME              Conda environment name        (default: weather-doc-extractor)
#   REPO_DIR                    Repository root               (default: ~/weather-doc-extractor)
#
# ── Example job submission (Azure CLI) ───────────────────────────────────────
#   az batch job create --id extract-job --pool-id gpu-pool
#   az batch task create \
#     --job-id extract-job \
#     --task-id "extract-{0..7}" \
#     --command-line "/bin/bash scripts/azure_extract_array.sh" \
#     --environment-settings \
#         TOTAL_SHARDS=8 \
#         WEATHER_IMAGES_DIR=/mnt/blob/Daily_rainfall_sample/images \
#         WEATHER_TRANSCRIPTIONS_DIR=/mnt/blob/Daily_rainfall_sample/transcriptions \
#         WEATHER_OUTPUT_DIR=/mnt/blob/outputs \
#         WEATHER_MODEL=smolvlm \
#         HF_HOME=/mnt/blob/hf_cache

set -euo pipefail

CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-weather-doc-extractor}"
REPO_DIR="${REPO_DIR:-$HOME/weather-doc-extractor}"
TOTAL_SHARDS="${TOTAL_SHARDS:?TOTAL_SHARDS must be set}"

# Azure Batch task IDs are 0-based; our CLI uses 1-based shard indices.
SHARD=$(( AZ_BATCH_TASK_ID + 1 ))

WEATHER_OUTPUT_DIR="${WEATHER_OUTPUT_DIR:-outputs}"
EXTRACT_OUTPUT_DIR="${EXTRACT_OUTPUT_DIR:-$WEATHER_OUTPUT_DIR/extractions}"

echo "[extract_array] Task $AZ_BATCH_TASK_ID → shard $SHARD/$TOTAL_SHARDS"
echo "[extract_array] Output directory: $EXTRACT_OUTPUT_DIR"

# ── Activate environment ──────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

cd "$REPO_DIR"

# ── Run batch extraction for this shard ───────────────────────────────────────
weather-extract batch-extract \
    --shard "$SHARD" \
    --total-shards "$TOTAL_SHARDS" \
    --output-dir "$EXTRACT_OUTPUT_DIR" \
    ${WEATHER_MODEL:+--model "$WEATHER_MODEL"}

echo "[extract_array] Shard $SHARD complete. Results in: $EXTRACT_OUTPUT_DIR"
