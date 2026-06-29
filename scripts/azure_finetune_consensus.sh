#!/usr/bin/env bash
# azure_finetune_consensus.sh — Run consensus-masked fine-tuning on Azure.
#
# This is additive to scripts/azure_finetune.sh and does not alter the
# standard fine-tuning path.

set -euo pipefail

CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-weather-doc-extractor}"
REPO_DIR="${REPO_DIR:-$HOME/weather-doc-extractor}"

# shellcheck source=/dev/null
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

cd "$REPO_DIR"

FINETUNE_ARGS=(finetune-consensus)
[[ -n "${WEATHER_MODEL:-}" ]] && FINETUNE_ARGS+=(--model "$WEATHER_MODEL")
[[ -n "${WEATHER_EPOCHS:-}" ]] && FINETUNE_ARGS+=(--epochs "$WEATHER_EPOCHS")
[[ -n "${WEATHER_REPORT_TO:-}" ]] && FINETUNE_ARGS+=(--report-to "$WEATHER_REPORT_TO")
[[ -n "${WEATHER_TRAINING_OUTPUT_DIR:-}" ]] && FINETUNE_ARGS+=(--output-dir "$WEATHER_TRAINING_OUTPUT_DIR")
[[ -n "${WEATHER_CONSENSUS_TRANSCRIPTIONS_DIR:-}" ]] && FINETUNE_ARGS+=(--consensus-dir "$WEATHER_CONSENSUS_TRANSCRIPTIONS_DIR")

echo "[finetune-consensus] Starting strict consensus fine-tuning"
echo "[finetune-consensus] Images: ${WEATHER_IMAGES_DIR:-Daily_rainfall_sample/images}"
echo "[finetune-consensus] Consensus transcriptions: ${WEATHER_CONSENSUS_TRANSCRIPTIONS_DIR:-$WEATHER_TRANSCRIPTIONS_DIR}"
echo "[finetune-consensus] Output dir: ${WEATHER_TRAINING_OUTPUT_DIR:-outputs/checkpoints}"
echo "[finetune-consensus] Args: ${FINETUNE_ARGS[*]}"

if [[ -n "${ACCELERATE_CONFIG:-}" ]]; then
    accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        -m weather_doc_extractor.cli \
        "${FINETUNE_ARGS[@]}"
else
    weather-extract "${FINETUNE_ARGS[@]}"
fi

echo "[finetune-consensus] Done."
