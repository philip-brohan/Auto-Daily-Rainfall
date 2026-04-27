#!/usr/bin/env bash
# setup_env.sh — Install the weather-doc-extractor environment on an Azure Batch node.
#
# Run once per node at pool start-up (set as the pool start task) or as the
# first step of a task that needs a fresh environment.
#
# Prerequisites on the node image:
#   - Miniconda / Miniforge already installed at $CONDA_HOME (default: ~/miniconda3)
#   - Git (to clone the repo if needed)
#   - CUDA drivers and toolkit pre-installed for GPU nodes
#
# Environment variables consumed (all have defaults):
#   CONDA_HOME        Path to Conda installation   (default: ~/miniconda3)
#   REPO_DIR          Local clone of the repository (default: ~/weather-doc-extractor)
#   CONDA_ENV_NAME    Name of the Conda environment (default: weather-doc-extractor)
#   HF_HOME           HuggingFace cache root        (default: ~/hf_cache)

set -euo pipefail

CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
REPO_DIR="${REPO_DIR:-$HOME/weather-doc-extractor}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-weather-doc-extractor}"
HF_HOME="${HF_HOME:-$HOME/hf_cache}"

# ── Activate Conda ────────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$CONDA_HOME/etc/profile.d/conda.sh"

# ── Create or update environment ─────────────────────────────────────────────
if conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo "[setup_env] Updating existing environment: $CONDA_ENV_NAME"
    conda env update -n "$CONDA_ENV_NAME" -f "$REPO_DIR/environment.yml" --prune
else
    echo "[setup_env] Creating environment: $CONDA_ENV_NAME"
    conda env create -n "$CONDA_ENV_NAME" -f "$REPO_DIR/environment.yml"
fi

conda activate "$CONDA_ENV_NAME"

# ── Install HuggingFace cache dir ─────────────────────────────────────────────
mkdir -p "$HF_HOME"
export HF_HOME

echo "[setup_env] Environment ready: $CONDA_ENV_NAME"
echo "[setup_env] HuggingFace cache: $HF_HOME"
