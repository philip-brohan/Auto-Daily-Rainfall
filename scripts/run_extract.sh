#!/usr/bin/env bash
# run_extract.sh — entrypoint called by azureml/extract_job.yml
# Environment variables set by the AML job:
#   WEATHER_IMAGES_DIR, SHARD, TOTAL_SHARDS, EXTRACT_LIMIT (optional)
#   WEATHER_MODEL (default model), WEATHER_CHECKPOINT (optional — overrides WEATHER_MODEL)
#   NODE_GPU_WORKERS (optional — one extraction worker per GPU on the node)
#   CLEAR_HF_MODULES (optional — set to "1" to delete stale transformers_modules cache)
set -euo pipefail

detect_gpu_count() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' '
    else
        echo "1"
    fi
}

# Clear stale HF custom model code if requested (fixes create_causal_mask errors after
# transformers version upgrades — the cached modules/ dir gets rebuilt on next download)
if [[ "${CLEAR_HF_MODULES:-0}" == "1" ]]; then
    # Clear both the modules cache and the granite model snapshots to force re-download
    MODULES_DIR="${HF_HOME:-}/modules/transformers_modules"
    GRANITE_DIR="${HF_HOME:-}/hub/models--ibm-granite--granite-vision-4.1-4b"
    
    if [[ -d "$MODULES_DIR" ]]; then
        echo "[cache] Clearing stale HF modules cache: $MODULES_DIR"
        rm -rf "$MODULES_DIR"
    fi
    if [[ -d "$GRANITE_DIR" ]]; then
        echo "[cache] Clearing Granite 4.1 model snapshots (will re-download): $GRANITE_DIR"
        rm -rf "$GRANITE_DIR"
    fi
    echo "[cache] Done — will re-download fresh model and custom code."
fi

LIMIT_FLAG=""
if [[ -n "${EXTRACT_LIMIT:-}" ]]; then
    LIMIT_FLAG="--limit $EXTRACT_LIMIT"
fi

BATCH_SIZE_FLAG=""
if [[ -n "${BATCH_SIZE:-}" ]]; then
    BATCH_SIZE_FLAG="--batch-size $BATCH_SIZE"
fi

RETRY_BATCH_SIZE_FLAG=""
if [[ -n "${RETRY_BATCH_SIZE:-}" ]]; then
    RETRY_BATCH_SIZE_FLAG="--retry-batch-size $RETRY_BATCH_SIZE"
fi

# Use checkpoint when adapter files exist at the mounted path (directly or one
# level deeper in a model-slug subdirectory).  Fail hard if a checkpoint is
# specified but no adapter_config.json can be found — never fall back silently.
MODEL=""
CHECKPOINT_REQUIRED="${WEATHER_CHECKPOINT_REQUIRED:-0}"
check_adapter_recursive() {
    local root="$1"
    # Check top level first
    if [[ -f "$root/adapter_config.json" ]]; then
        echo "[run_extract] Found adapter at top level: $root/adapter_config.json" >&2
        echo "$root"
        return 0
    fi
    # Then check one level deeper (in case it's in a model-slug subdirectory)
    if [[ -d "$root" ]]; then
        echo "[run_extract] Searching subdirectories of: $root" >&2
        # Use find to search more reliably (handles symlinks, special names)
        local found
        found=$(find "$root" -maxdepth 2 -name "adapter_config.json" -type f | head -1)
        if [[ -n "$found" ]]; then
            # Return the directory containing adapter_config.json
            local adapter_dir
            adapter_dir=$(dirname "$found")
            echo "[run_extract] Found adapter at: $found" >&2
            echo "$adapter_dir"
            return 0
        fi
    fi
    echo "[run_extract] No adapter_config.json found in $root (searched top level and 2 levels deep)" >&2
    return 1
}
if [[ -n "${WEATHER_CHECKPOINT:-}" ]] && [[ "$CHECKPOINT_REQUIRED" == "1" ]]; then
    echo "[run_extract] Looking for adapter in: $WEATHER_CHECKPOINT" >&2
    MODEL="$(check_adapter_recursive "$WEATHER_CHECKPOINT")" || true
    if [[ -z "$MODEL" ]]; then
        echo "[run_extract] ERROR: no adapter_config.json found in $WEATHER_CHECKPOINT (top level or subdirectories)." >&2
        exit 1
    fi
    echo "[run_extract] Using checkpoint: $MODEL" >&2
elif [[ -n "${WEATHER_CHECKPOINT:-}" ]] && [[ "$CHECKPOINT_REQUIRED" != "1" ]]; then
    echo "[run_extract] Ignoring checkpoint input mount (not explicitly requested)." >&2
    MODEL="$(check_adapter_recursive "$WEATHER_CHECKPOINT")" || true
    if [[ -n "$MODEL" ]]; then
        echo "[run_extract] Found adapter in checkpoint mount; using: $MODEL" >&2
    elif [[ -n "${WEATHER_MODEL:-}" ]]; then
        MODEL="$WEATHER_MODEL"
        echo "[run_extract] No adapter found in checkpoint mount; falling back to WEATHER_MODEL: $MODEL" >&2
    fi
elif [[ -n "${WEATHER_MODEL:-}" ]]; then
    MODEL="$WEATHER_MODEL"
else
    echo "[run_extract] ERROR: neither WEATHER_CHECKPOINT nor WEATHER_MODEL is set — refusing to use a default model." >&2
    exit 1
fi

if [[ -z "$MODEL" ]]; then
    echo "[run_extract] ERROR: resolved model is empty; refusing to continue." >&2
    exit 1
fi

echo "[run_extract] shard=$SHARD/$TOTAL_SHARDS images=$WEATHER_IMAGES_DIR output=$OUTPUT_DIR model=$MODEL ${LIMIT_FLAG:+limit=$EXTRACT_LIMIT} ${BATCH_SIZE_FLAG:+batch=$BATCH_SIZE} ${RETRY_BATCH_SIZE_FLAG:+retry_batch=$RETRY_BATCH_SIZE}"
python -c "import torch, transformers; print(f'[env] torch={torch.__version__} transformers={transformers.__version__} cuda={torch.version.cuda}')"

cmd=(
    python -m weather_doc_extractor.cli batch-extract
    --shard "$SHARD"
    --total-shards "$TOTAL_SHARDS"
    --output-dir "$OUTPUT_DIR"
    --model "$MODEL"
)

if [[ -n "${EXTRACT_LIMIT:-}" ]]; then
    cmd+=(--limit "$EXTRACT_LIMIT")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
    cmd+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "${RETRY_BATCH_SIZE:-}" ]]; then
    cmd+=(--retry-batch-size "$RETRY_BATCH_SIZE")
fi

AVAILABLE_GPUS="$(detect_gpu_count)"
NODE_GPU_WORKERS="${NODE_GPU_WORKERS:-1}"

if ! [[ "$NODE_GPU_WORKERS" =~ ^[0-9]+$ ]] || [[ "$NODE_GPU_WORKERS" -lt 1 ]]; then
    echo "[run_extract] ERROR: NODE_GPU_WORKERS must be a positive integer (got: $NODE_GPU_WORKERS)" >&2
    exit 1
fi

if [[ "$NODE_GPU_WORKERS" -gt "$AVAILABLE_GPUS" ]]; then
    echo "[run_extract] WARNING: NODE_GPU_WORKERS=$NODE_GPU_WORKERS exceeds available GPUs=$AVAILABLE_GPUS; capping to $AVAILABLE_GPUS" >&2
    NODE_GPU_WORKERS="$AVAILABLE_GPUS"
fi

if [[ "$NODE_GPU_WORKERS" -eq 1 ]]; then
    "${cmd[@]}"
    exit 0
fi

echo "[run_extract] Launching $NODE_GPU_WORKERS extraction workers on this node (available GPUs: $AVAILABLE_GPUS)."
echo "[run_extract] Effective global shards per job: $((TOTAL_SHARDS * NODE_GPU_WORKERS))"

pids=()
for worker_idx in $(seq 1 "$NODE_GPU_WORKERS"); do
    gpu_idx=$((worker_idx - 1))
    worker_shard=$(((SHARD - 1) * NODE_GPU_WORKERS + worker_idx))
    worker_total_shards=$((TOTAL_SHARDS * NODE_GPU_WORKERS))

    (
        export CUDA_VISIBLE_DEVICES="$gpu_idx"
        worker_cmd=(
            python -m weather_doc_extractor.cli batch-extract
            --shard "$worker_shard"
            --total-shards "$worker_total_shards"
            --output-dir "$OUTPUT_DIR"
            --model "$MODEL"
        )

        if [[ -n "${EXTRACT_LIMIT:-}" ]]; then
            worker_cmd+=(--limit "$EXTRACT_LIMIT")
        fi
        if [[ -n "${BATCH_SIZE:-}" ]]; then
            worker_cmd+=(--batch-size "$BATCH_SIZE")
        fi
        if [[ -n "${RETRY_BATCH_SIZE:-}" ]]; then
            worker_cmd+=(--retry-batch-size "$RETRY_BATCH_SIZE")
        fi

        echo "[run_extract] Worker $worker_idx/$NODE_GPU_WORKERS on GPU $gpu_idx => shard $worker_shard/$worker_total_shards"
        "${worker_cmd[@]}"
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    echo "[run_extract] ERROR: One or more GPU workers failed." >&2
    exit 1
fi

echo "[run_extract] All GPU workers completed successfully."
