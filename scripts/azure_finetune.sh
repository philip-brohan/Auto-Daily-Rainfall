#!/usr/bin/env bash
# azure_finetune.sh — Run fine-tuning on an Azure Batch GPU node.
#
# Supports single-node multi-GPU via `accelerate launch` when
# ACCELERATE_CONFIG is set, or falls back to direct `weather-extract finetune`
# for single-GPU / CPU runs.
#
# ── Required environment variables ───────────────────────────────────────────
#   WEATHER_IMAGES_DIR          Path to mounted images directory
#   WEATHER_TRANSCRIPTIONS_DIR  Path to mounted transcriptions directory
#   WEATHER_TRAINING_OUTPUT_DIR Path to write LoRA adapter checkpoints
#
# ── Optional environment variables ───────────────────────────────────────────
#   WEATHER_MODEL         HuggingFace model ID or preset name (default: smolvlm)
#   WEATHER_EPOCHS        Number of training epochs           (default: 3)
#   WEATHER_BATCH_SIZE    Per-device batch size               (default: 1)
#   WEATHER_GRAD_ACCUM_STEPS  Gradient accumulation steps     (default: 8)
#   WEATHER_LEARNING_RATE Learning rate                       (default: 0.0002)
#   WEATHER_REPORT_TO     Tracking backend: none|wandb|tensorboard (default: none)
#   ACCELERATE_CONFIG     Path to accelerate config YAML for multi-GPU launch
#   HF_HOME               HuggingFace model cache root
#   CONDA_HOME            Conda installation root             (default: ~/miniconda3)
#   CONDA_ENV_NAME        Conda environment name              (default: weather-doc-extractor)
#   REPO_DIR              Repository root                     (default: ~/weather-doc-extractor)
#
# ── Example (single GPU) ─────────────────────────────────────────────────────
#   WEATHER_IMAGES_DIR=/mnt/blob/images \
#   WEATHER_TRANSCRIPTIONS_DIR=/mnt/blob/transcriptions \
#   WEATHER_TRAINING_OUTPUT_DIR=/mnt/blob/outputs/checkpoints \
#   WEATHER_MODEL=smolvlm \
#   WEATHER_EPOCHS=5 \
#   bash scripts/azure_finetune.sh
#
# ── Example (multi-GPU with accelerate) ──────────────────────────────────────
#   ACCELERATE_CONFIG=scripts/accelerate_config.yaml \
#   WEATHER_IMAGES_DIR=/mnt/blob/images \
#   ... \
#   bash scripts/azure_finetune.sh

set -euo pipefail

CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-weather-doc-extractor}"
REPO_DIR="${REPO_DIR:-$HOME/weather-doc-extractor}"

# ── Activate environment ──────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

cd "$REPO_DIR"

# ── Build CLI arguments ───────────────────────────────────────────────────────
FINETUNE_ARGS=()
[[ -n "${WEATHER_MODEL:-}" ]]         && FINETUNE_ARGS+=(--model "$WEATHER_MODEL")
[[ -n "${WEATHER_EPOCHS:-}" ]]        && FINETUNE_ARGS+=(--epochs "$WEATHER_EPOCHS")
[[ -n "${WEATHER_REPORT_TO:-}" ]]     && FINETUNE_ARGS+=(--report-to "$WEATHER_REPORT_TO")
[[ -n "${WEATHER_TRAINING_OUTPUT_DIR:-}" ]] && FINETUNE_ARGS+=(--output-dir "$WEATHER_TRAINING_OUTPUT_DIR")

echo "[finetune] Starting fine-tuning"
echo "[finetune] Images:          ${WEATHER_IMAGES_DIR:-Daily_rainfall_sample/images}"
echo "[finetune] Transcriptions:  ${WEATHER_TRANSCRIPTIONS_DIR:-Daily_rainfall_sample/transcriptions}"
echo "[finetune] Output dir:      ${WEATHER_TRAINING_OUTPUT_DIR:-outputs/checkpoints}"
echo "[finetune] Args: ${FINETUNE_ARGS[*]:-<none>}"

# ── Launch ────────────────────────────────────────────────────────────────────
if [[ -n "${ACCELERATE_CONFIG:-}" ]]; then
    echo "[finetune] Multi-GPU launch via accelerate (config: $ACCELERATE_CONFIG)"
    accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        -m weather_doc_extractor.cli \
        finetune "${FINETUNE_ARGS[@]}"
else
    echo "[finetune] Single-device launch"
    weather-extract finetune "${FINETUNE_ARGS[@]}"
fi

echo "[finetune] Done."
