#!/usr/bin/env bash
# azure_evaluate_array.sh — Run evaluation as an Azure Batch job array.
#
# Each task in the array evaluates one shard of the paired records and writes
# its results to a per-shard JSON file.  After all tasks complete, run
# aggregate_eval.py (or inspect the files manually) to combine results.
#
# Azure Batch injects $AZ_BATCH_TASK_ID (0-based) automatically.
# This script converts it to the 1-based shard index expected by the CLI.
#
# ── Required environment variables ───────────────────────────────────────────
#   TOTAL_SHARDS          Total number of array tasks (= job array size)
#   WEATHER_IMAGES_DIR    Path to mounted images directory
#   WEATHER_TRANSCRIPTIONS_DIR  Path to mounted transcriptions directory
#   WEATHER_OUTPUT_DIR    Path to write evaluation reports
#
# ── Optional environment variables ───────────────────────────────────────────
#   WEATHER_MODEL         HuggingFace model ID or preset name (default: smolvlm)
#   WEATHER_DEVICE        torch device_map value          (default: auto)
#   HF_HOME               HuggingFace model cache root
#   CONDA_HOME            Conda installation root         (default: ~/miniconda3)
#   CONDA_ENV_NAME        Conda environment name          (default: weather-doc-extractor)
#   REPO_DIR              Repository root                 (default: ~/weather-doc-extractor)
#
# ── Example job submission (Azure CLI) ───────────────────────────────────────
#   az batch job create --id eval-job --pool-id gpu-pool
#   az batch task create \
#     --job-id eval-job \
#     --task-id "eval-{0..7}" \
#     --command-line "/bin/bash scripts/azure_evaluate_array.sh" \
#     --environment-settings \
#         TOTAL_SHARDS=8 \
#         WEATHER_IMAGES_DIR=/mnt/blob/Daily_rainfall_sample/images \
#         WEATHER_TRANSCRIPTIONS_DIR=/mnt/blob/Daily_rainfall_sample/transcriptions \
#         WEATHER_OUTPUT_DIR=/mnt/blob/outputs/eval \
#         WEATHER_MODEL=smolvlm

set -euo pipefail

CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-weather-doc-extractor}"
REPO_DIR="${REPO_DIR:-$HOME/weather-doc-extractor}"
TOTAL_SHARDS="${TOTAL_SHARDS:?TOTAL_SHARDS must be set}"

# Azure Batch task IDs are 0-based; our CLI uses 1-based shard indices.
SHARD=$(( AZ_BATCH_TASK_ID + 1 ))

OUTPUT_DIR="${WEATHER_OUTPUT_DIR:-outputs/eval}"
OUTPUT_FILE="$OUTPUT_DIR/shard_${SHARD}_of_${TOTAL_SHARDS}.json"

echo "[evaluate_array] Task $AZ_BATCH_TASK_ID → shard $SHARD/$TOTAL_SHARDS"
echo "[evaluate_array] Output: $OUTPUT_FILE"

# ── Activate environment ──────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

cd "$REPO_DIR"

# ── Run evaluation for this shard ─────────────────────────────────────────────
weather-extract evaluate \
    --shard "$SHARD" \
    --total-shards "$TOTAL_SHARDS" \
    --output-file "$OUTPUT_FILE" \
    ${WEATHER_MODEL:+--model "$WEATHER_MODEL"}

echo "[evaluate_array] Shard $SHARD complete. Report: $OUTPUT_FILE"
