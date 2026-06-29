#!/usr/bin/env bash
# aml_submit.sh — Submit weather-extract jobs to an Azure ML workspace.
#
# ── Configuration file ────────────────────────────────────────────────────────
#   Copy azureml/config.env.example to azureml/config.env and fill in your
#   workspace coordinates and data paths.  The script sources it automatically.
#   CLI flags and environment variables override any value in config.env.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/aml_submit.sh {extract|evaluate|finetune|finetune-consensus|env} [options]
#
# ── Options ───────────────────────────────────────────────────────────────────
#   --subscription SUB      Azure subscription ID or name
#   --resource-group RG     Azure resource group containing the workspace
#   --workspace WS          Azure ML workspace name
#   --compute CLUSTER       Compute cluster name (default: A100x8)
#   --total-shards N        Parallel shards for extract/evaluate (default: 8)
#   --limit N               Process only N images per shard (for smoke tests);
#                           also defaults --total-shards to 1
#   --batch-size N          Batch size per extraction job (default: 10)
#   --node-gpu-workers N    Parallel extraction workers per AML node/GPU (default: 1)
#   --finetune-gpu-workers N  Number of GPU processes for finetune jobs (default: 1)
#   --epochs N              Number of epochs for finetune jobs (default: 3)
#   --grad-accum-steps N    Baseline gradient accumulation steps (default: 8)
#   --auto-scale-grad-accum true|false
#                           Auto-scale grad accumulation by world size for DDP (default: true)
#   --retry-batch-size N    Stage-2 retry batch size for parse failures
#                           (default: min(batch-size, 4))
#   --model MODEL           Model preset (smolvlm, smolvlm2, granite, granite4) or HF ID
#   --env-variant v100|a100 Azure ML environment to use (default: v100).
#                           v100 uses weather-doc-extractor  (PyTorch 2.4, CUDA 12.1)
#                           a100 uses weather-doc-extractor-a100 (PyTorch 2.8, CUDA 12.6)
#                           Auto-selected for granite4, gemma3, gemma4, ministral which need torch>=2.6.
#   --dataset real|fake|test_real|test_fake
#                           Quick dataset selector:
#                             real      → Daily_rainfall_sample
#                             fake      → fake_daily_rainfall
#                             test_real → test_data/real  (committed test split)
#                             test_fake → test_data/fake  (committed test split)
#   --dataset-dir PATH      Custom dataset root path containing:
#                             PATH/images
#                             PATH/transcriptions
#   --images-path PATH      Override AML_IMAGES_PATH
#   --transcriptions-path PATH  Override AML_TRANSCRIPTIONS_PATH
#   --consensus-transcriptions-path PATH
#                           Override consensus transcription path used by
#                           finetune-consensus
#   --checkpoint PATH       Use a saved fine-tuned checkpoint for extraction
#                           (overrides --model); path relative to AML_DATASTORE_BASE
#   --extraction-registry FILE
#                           Local JSON registry file for submitted extraction runs
#                           (default: outputs/extraction_registry.json)
#   --model-registry FILE   Local JSON registry file for fine-tuned checkpoints
#                           (default: outputs/model_registry.json)
#   --clear-modules         Delete stale HF transformers_modules cache before running.
#                           Use after upgrading transformers to fix custom-model errors.
#   --help                  Show this message
#
# ── Examples ──────────────────────────────────────────────────────────────────
#   # First-time setup: copy and edit the config, then register the environment:
#   cp azureml/config.env.example azureml/config.env
#   # edit azureml/config.env
#   bash scripts/aml_submit.sh env
#
#   # Extract from real data (default, 8 shards):
#   bash scripts/aml_submit.sh --total-shards 8 extract
#
#   # Extract from fake data (quick test):
#   bash scripts/aml_submit.sh --dataset fake --limit 10 extract
#
#   # Extract from fake data for full fine-tuning:
#   bash scripts/aml_submit.sh --dataset fake --model granite4 finetune
#
#   # Fine-tune from a custom synthetic dataset directory:
#   bash scripts/aml_submit.sh --dataset-dir fake_daily_rainfall_v2 --model smolvlm2 finetune
#
#   # Extract using a fine-tuned checkpoint:
#   bash scripts/aml_submit.sh --checkpoint outputs/checkpoints/granite-fake-20260526-143000 --limit 50 extract
#
#   # Quick smoke test — real data, 1 shard, 5 images:
#   bash scripts/aml_submit.sh --limit 5 extract

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$REPO_DIR/azureml/config.env"

# ── Load config.env (if present) before applying CLI overrides ───────────────
if [[ -f "$CONFIG_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$CONFIG_FILE"
fi

# Apply defaults *after* sourcing so config.env values become the baseline.
COMMAND=""
TOTAL_SHARDS="${TOTAL_SHARDS:-1}"
EXTRACT_LIMIT=""
BATCH_SIZE="${BATCH_SIZE:-}"
NODE_GPU_WORKERS="${NODE_GPU_WORKERS:-1}"
FINETUNE_GPU_WORKERS="${FINETUNE_GPU_WORKERS:-1}"
WEATHER_EPOCHS="${WEATHER_EPOCHS:-}"
WEATHER_GRAD_ACCUM_STEPS="${WEATHER_GRAD_ACCUM_STEPS:-8}"
WEATHER_AUTO_SCALE_GRAD_ACCUM="${WEATHER_AUTO_SCALE_GRAD_ACCUM:-true}"
RETRY_BATCH_SIZE="${RETRY_BATCH_SIZE:-}"
WEATHER_MODEL="${WEATHER_MODEL:-}"
WEATHER_CHECKPOINT=""
EXTRACTION_REGISTRY_FILE="${EXTRACTION_REGISTRY_FILE:-outputs/extraction_registry.json}"
MODEL_REGISTRY_FILE="${MODEL_REGISTRY_FILE:-outputs/model_registry.json}"
CLEAR_HF_MODULES="0"
HF_TOKEN="${HF_TOKEN:-}"
AML_ENV_VARIANT="${AML_ENV_VARIANT:-v100}"
AML_COMPUTE="${AML_COMPUTE:-A100x8}"
AML_SUBSCRIPTION="${AML_SUBSCRIPTION:-}"
AML_RESOURCE_GROUP="${AML_RESOURCE_GROUP:-}"
AML_WORKSPACE="${AML_WORKSPACE:-}"
AML_DATASTORE_BASE="${AML_DATASTORE_BASE:-azureml://datastores/workspaceblobstore/paths}"
AML_IMAGES_PATH="${AML_IMAGES_PATH:-Daily_rainfall_sample/images}"
AML_TRANSCRIPTIONS_PATH="${AML_TRANSCRIPTIONS_PATH:-Daily_rainfall_sample/transcriptions}"
AML_CONSENSUS_TRANSCRIPTIONS_PATH="${AML_CONSENSUS_TRANSCRIPTIONS_PATH:-$AML_TRANSCRIPTIONS_PATH}"
AML_OUTPUTS_PATH="${AML_OUTPUTS_PATH:-outputs}"
AML_HF_CACHE_PATH="${AML_HF_CACHE_PATH:-hf_cache}"

join_datastore_uri() {
    local base="${1%/}"
    local rel="${2#/}"
    echo "${base}/${rel}"
}

usage() {
    sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subscription)      AML_SUBSCRIPTION="$2";    shift 2 ;;
        --resource-group)    AML_RESOURCE_GROUP="$2";  shift 2 ;;
        --workspace)         AML_WORKSPACE="$2";        shift 2 ;;
        --total-shards)      TOTAL_SHARDS="$2";         shift 2 ;;
        --epochs)            WEATHER_EPOCHS="$2";       shift 2 ;;
        --limit)             EXTRACT_LIMIT="$2";        shift 2 ;;
        --batch-size)        BATCH_SIZE="$2";            shift 2 ;;
        --node-gpu-workers)  NODE_GPU_WORKERS="$2";      shift 2 ;;
        --finetune-gpu-workers) FINETUNE_GPU_WORKERS="$2"; shift 2 ;;
        --retry-batch-size)  RETRY_BATCH_SIZE="$2";      shift 2 ;;
        --grad-accum-steps)  WEATHER_GRAD_ACCUM_STEPS="$2"; shift 2 ;;
        --auto-scale-grad-accum) WEATHER_AUTO_SCALE_GRAD_ACCUM="$2"; shift 2 ;;
        --model)             WEATHER_MODEL="$2";        shift 2 ;;
        --compute)           AML_COMPUTE="$2";          shift 2 ;;
        --env-variant)       AML_ENV_VARIANT="$2";      shift 2 ;;
        --dataset)
            case "$2" in
                real)      AML_IMAGES_PATH="Daily_rainfall_sample/images"; AML_TRANSCRIPTIONS_PATH="Daily_rainfall_sample/transcriptions" ;;
                fake)      AML_IMAGES_PATH="fake_daily_rainfall/images"; AML_TRANSCRIPTIONS_PATH="fake_daily_rainfall/transcriptions" ;;
                test_real) AML_IMAGES_PATH="test_data/real/images"; AML_TRANSCRIPTIONS_PATH="test_data/real/transcriptions" ;;
                test_fake) AML_IMAGES_PATH="test_data/fake/images"; AML_TRANSCRIPTIONS_PATH="test_data/fake/transcriptions" ;;
                *) echo "Unknown dataset: $2 (use 'real', 'fake', 'test_real', or 'test_fake')" >&2; exit 1 ;;
            esac
            AML_CONSENSUS_TRANSCRIPTIONS_PATH="$AML_TRANSCRIPTIONS_PATH"
            shift 2 ;;
        --dataset-dir)
            DATASET_DIR="${2%/}"
            AML_IMAGES_PATH="$DATASET_DIR/images"
            AML_TRANSCRIPTIONS_PATH="$DATASET_DIR/transcriptions"
            AML_CONSENSUS_TRANSCRIPTIONS_PATH="$AML_TRANSCRIPTIONS_PATH"
            shift 2 ;;
        --images-path)       AML_IMAGES_PATH="$2";       shift 2 ;;
        --transcriptions-path) AML_TRANSCRIPTIONS_PATH="$2"; shift 2 ;;
        --consensus-transcriptions-path) AML_CONSENSUS_TRANSCRIPTIONS_PATH="$2"; shift 2 ;;
        --checkpoint)        WEATHER_CHECKPOINT="$2";    shift 2 ;;
        --extraction-registry) EXTRACTION_REGISTRY_FILE="$2"; shift 2 ;;
        --model-registry)    MODEL_REGISTRY_FILE="$2"; shift 2 ;;
        --clear-modules)     CLEAR_HF_MODULES="1";         shift ;;
        --help|-h)           usage 0 ;;
        extract|evaluate|finetune|finetune-consensus|env) COMMAND="$1"; shift ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

    [[ -z "$COMMAND" ]] && { echo "Error: command required (extract|evaluate|finetune|finetune-consensus|env)" >&2; usage 1; }
[[ "$COMMAND" == "extract" && -z "$WEATHER_MODEL" && -z "$WEATHER_CHECKPOINT" ]] && { echo "Error: --model or --checkpoint required for extract" >&2; exit 1; }
[[ -z "$AML_SUBSCRIPTION" ]]   && { echo "Error: AML_SUBSCRIPTION not set (use --subscription or azureml/config.env)"   >&2; exit 1; }
[[ -z "$AML_RESOURCE_GROUP" ]] && { echo "Error: AML_RESOURCE_GROUP not set (use --resource-group or azureml/config.env)" >&2; exit 1; }
[[ -z "$AML_WORKSPACE" ]]      && { echo "Error: AML_WORKSPACE not set (use --workspace or azureml/config.env)"           >&2; exit 1; }

# Select environment file and registered name based on variant.
# Auto-upgrade to a100 if the model requires torch >= 2.6 (v100 has 2.4)
case "$WEATHER_MODEL" in
    granite4|granite-vision-4.1*|gemma3|gemma4|ministral|mistral*)
        if [[ "$AML_ENV_VARIANT" == "v100" ]]; then
            echo "⚠️  Model $WEATHER_MODEL requires torch >= 2.6. Auto-switching to a100 environment."
            AML_ENV_VARIANT="a100"
        fi
        ;;
esac

if [[ "$AML_ENV_VARIANT" == "a100" ]]; then
    AML_ENV_FILE="$REPO_DIR/azureml/environment-a100.yml"
    AML_ENV_OVERRIDE="--set environment=azureml:weather-doc-extractor-a100@latest"
else
    AML_ENV_FILE="$REPO_DIR/azureml/environment.yml"
    AML_ENV_OVERRIDE=""
fi

# When --limit is set and --total-shards was not explicitly specified, default to 1 shard.
[[ -n "$EXTRACT_LIMIT" && "$TOTAL_SHARDS" -eq 8 ]] && TOTAL_SHARDS=1

# Command-specific defaults:
# - extract/evaluate use extraction batch defaults
# - finetune modes use training defaults
if [[ "$COMMAND" == "extract" || "$COMMAND" == "evaluate" ]]; then
    [[ -z "$BATCH_SIZE" ]] && BATCH_SIZE=10
fi

if [[ "$COMMAND" == "finetune" || "$COMMAND" == "finetune-consensus" ]]; then
    [[ -z "$BATCH_SIZE" ]] && BATCH_SIZE=1
    [[ -z "$WEATHER_EPOCHS" ]] && WEATHER_EPOCHS=3
fi

# Output subdirectory: extractions/<model-slug>/<YYYYMMDD-HHMMSS>
# This ensures runs from different models and repeated runs never overwrite each other.
# If checkpoint is specified, use it for extraction; otherwise use the model preset.
if [[ -n "$WEATHER_CHECKPOINT" ]]; then
    # Checkpoint overrides model; extract the checkpoint name for the output directory
    CHECKPOINT_NAME="${WEATHER_CHECKPOINT##*/}"
    CHECKPOINT_URI="$(join_datastore_uri "$AML_DATASTORE_BASE" "$WEATHER_CHECKPOINT")"
    MODEL_SLUG="$CHECKPOINT_NAME"
    CHECKPOINT_REQUIRED="1"
else
    CHECKPOINT_URI=""
    MODEL_SLUG="${WEATHER_MODEL##*/}"
    CHECKPOINT_REQUIRED="0"
fi
RUN_TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
EXTRACTIONS_REL_PATH="$AML_OUTPUTS_PATH/extractions/$MODEL_SLUG/$RUN_TIMESTAMP"
CHECKPOINT_RUN_SLUG="${MODEL_SLUG}-${RUN_TIMESTAMP}"
CHECKPOINTS_REL_PATH="$AML_OUTPUTS_PATH/checkpoints/$CHECKPOINT_RUN_SLUG"
# Fine-tune saves under the resolved full model ID slug ("/" -> "--"), not
# the short preset name. Resolve the effective model name so registry paths
# always match the actual folder written by run_finetune().
case "$WEATHER_MODEL" in
    smolvlm)   RESOLVED_TRAINING_MODEL_NAME="HuggingFaceTB/SmolVLM-500M-Instruct" ;;
    smolvlm2)  RESOLVED_TRAINING_MODEL_NAME="HuggingFaceTB/SmolVLM2-2.2B-Instruct" ;;
    granite)   RESOLVED_TRAINING_MODEL_NAME="ibm-granite/granite-vision-3.2-2b" ;;
    granite4)  RESOLVED_TRAINING_MODEL_NAME="ibm-granite/granite-vision-4.1-4b" ;;
    gemma3)    RESOLVED_TRAINING_MODEL_NAME="google/gemma-3-4b-it" ;;
    gemma4)    RESOLVED_TRAINING_MODEL_NAME="google/gemma-4-E4B-it" ;;
    ministral) RESOLVED_TRAINING_MODEL_NAME="mistralai/Mistral-Small-3.1-24B-Instruct-2503" ;;
    *)         RESOLVED_TRAINING_MODEL_NAME="$WEATHER_MODEL" ;;
esac
TRAINING_MODEL_SLUG="${RESOLVED_TRAINING_MODEL_NAME//\//--}"
CHECKPOINT_PATH="$CHECKPOINTS_REL_PATH/$TRAINING_MODEL_SLUG"

# Resolved datastore URIs
IMAGES_URI="$(join_datastore_uri "$AML_DATASTORE_BASE" "$AML_IMAGES_PATH")"
TRANSCRIPTIONS_URI="$(join_datastore_uri "$AML_DATASTORE_BASE" "$AML_TRANSCRIPTIONS_PATH")"
CONSENSUS_TRANSCRIPTIONS_URI="$(join_datastore_uri "$AML_DATASTORE_BASE" "$AML_CONSENSUS_TRANSCRIPTIONS_PATH")"
OUTPUTS_URI="$(join_datastore_uri "$AML_DATASTORE_BASE" "$AML_OUTPUTS_PATH")"
EXTRACTIONS_URI="$(join_datastore_uri "$AML_DATASTORE_BASE" "$EXTRACTIONS_REL_PATH")"
HF_CACHE_URI="$(join_datastore_uri "$AML_DATASTORE_BASE" "$AML_HF_CACHE_PATH")"

AML_ARGS=(
    --workspace-name "$AML_WORKSPACE"
    --resource-group "$AML_RESOURCE_GROUP"
    --subscription   "$AML_SUBSCRIPTION"
)

echo "Workspace:  $AML_WORKSPACE  ($AML_RESOURCE_GROUP / $AML_SUBSCRIPTION)"
echo "Compute:    $AML_COMPUTE"
echo "GPU workers per node: $NODE_GPU_WORKERS"
echo "Finetune GPU processes: $FINETUNE_GPU_WORKERS"
echo "Grad accum steps (base): $WEATHER_GRAD_ACCUM_STEPS"
echo "Auto-scale grad accum:   $WEATHER_AUTO_SCALE_GRAD_ACCUM"
echo "Model:      $WEATHER_MODEL"
echo "Images:     $IMAGES_URI"
[[ "$COMMAND" != "extract" ]] && echo "Transcript: $TRANSCRIPTIONS_URI"
echo "Outputs:    $OUTPUTS_URI"
echo

case "$COMMAND" in
    env)
        # Auto-increment the version so re-registration never fails on duplicate.
        ENV_NAME=$(grep '^name:' "$AML_ENV_FILE" | awk '{print $2}')
        CURRENT_VERSION=$(az ml environment list \
            --name "$ENV_NAME" "${AML_ARGS[@]}" \
            --query "[].version" --output tsv 2>/dev/null \
            | sort -n | tail -1) || true
        [[ -z "$CURRENT_VERSION" ]] && CURRENT_VERSION=0
        NEXT_VERSION=$(( CURRENT_VERSION + 1 ))
        echo "Registering environment '$ENV_NAME' version $NEXT_VERSION (variant: $AML_ENV_VARIANT)..."
        PATCHED_YML="$REPO_DIR/azureml/.environment_tmp.yml"
        sed "s/^version:.*/version: $NEXT_VERSION/" "$AML_ENV_FILE" > "$PATCHED_YML"
        az ml environment create --file "$PATCHED_YML" "${AML_ARGS[@]}"
        rm -f "$PATCHED_YML"
        sed -i "s/^version:.*/version: $NEXT_VERSION/" "$AML_ENV_FILE"
        echo "Environment registered as version $NEXT_VERSION."
        echo "$AML_ENV_FILE updated — commit the version bump if desired."
        ;;

    extract)
        echo "Submitting $TOTAL_SHARDS extract shard(s)${EXTRACT_LIMIT:+ (limit: $EXTRACT_LIMIT images each)}..."
        [[ -n "$WEATHER_CHECKPOINT" ]] && echo "  Checkpoint: $CHECKPOINT_URI"
        echo "  Output:   $EXTRACTIONS_URI"
        echo "  HF cache: $HF_CACHE_URI"
        JOB_IDS=()
        for i in $(seq 1 "$TOTAL_SHARDS"); do
            echo "  Shard $i / $TOTAL_SHARDS ..."
            JOB_ID=$(az ml job create \
                --file "$REPO_DIR/azureml/extract_job.yml" \
                "${AML_ARGS[@]}" \
                --set compute="azureml:$AML_COMPUTE" \
                ${AML_ENV_OVERRIDE:+$AML_ENV_OVERRIDE} \
                --set inputs.images_dir.path="$IMAGES_URI" \
                --set outputs.extractions.path="$EXTRACTIONS_URI" \
                --set outputs.hf_cache.path="$HF_CACHE_URI" \
                --set inputs.checkpoint_dir.path="${CHECKPOINT_URI:-$IMAGES_URI}" \
                --set environment_variables.SHARD="$i" \
                --set environment_variables.TOTAL_SHARDS="$TOTAL_SHARDS" \
                --set environment_variables.WEATHER_MODEL="$WEATHER_MODEL" \
                --set environment_variables.WEATHER_CHECKPOINT_REQUIRED="$CHECKPOINT_REQUIRED" \
                --set environment_variables.BATCH_SIZE="$BATCH_SIZE" \
                --set environment_variables.NODE_GPU_WORKERS="$NODE_GPU_WORKERS" \
                ${RETRY_BATCH_SIZE:+--set environment_variables.RETRY_BATCH_SIZE="$RETRY_BATCH_SIZE"} \
                ${EXTRACT_LIMIT:+--set environment_variables.EXTRACT_LIMIT="$EXTRACT_LIMIT"} \
                --set environment_variables.CLEAR_HF_MODULES="$CLEAR_HF_MODULES" \
                ${HF_TOKEN:+--set environment_variables.HF_TOKEN="$HF_TOKEN"} \
                --set display_name="batch-extract-${MODEL_SLUG}-${i}-of-${TOTAL_SHARDS}" \
                --query name --output tsv)
            echo "$JOB_ID"
            JOB_IDS+=("$JOB_ID")
        done

        JOB_IDS_CSV=""
        if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
            JOB_IDS_CSV="$(IFS=,; echo "${JOB_IDS[*]}")"
        fi

        python3 "$REPO_DIR/scripts/create_extraction_registry_entry.py" \
            --extractions-path "$EXTRACTIONS_REL_PATH" \
            --model "$WEATHER_MODEL" \
            --model-slug "$MODEL_SLUG" \
            --dataset "$AML_IMAGES_PATH" \
            --images-path "$AML_IMAGES_PATH" \
            --transcriptions-path "$AML_TRANSCRIPTIONS_PATH" \
            --total-shards "$TOTAL_SHARDS" \
            --registry-file "$EXTRACTION_REGISTRY_FILE" \
            ${WEATHER_CHECKPOINT:+--checkpoint-path "$WEATHER_CHECKPOINT"} \
            ${EXTRACT_LIMIT:+--limit "$EXTRACT_LIMIT"} \
            ${JOB_IDS_CSV:+--job-ids "$JOB_IDS_CSV"}

        echo "Submitted $TOTAL_SHARDS extract job(s)."
        echo "Extraction registry: $EXTRACTION_REGISTRY_FILE"
        ;;

    evaluate)
        echo "Submitting $TOTAL_SHARDS evaluate shards..."
        for i in $(seq 1 "$TOTAL_SHARDS"); do
            echo "  Shard $i / $TOTAL_SHARDS ..."
            az ml job create \
                --file "$REPO_DIR/azureml/evaluate_job.yml" \
                "${AML_ARGS[@]}" \
                --set compute="azureml:$AML_COMPUTE" \
                ${AML_ENV_OVERRIDE:+$AML_ENV_OVERRIDE} \
                --set inputs.images_dir.path="$IMAGES_URI" \
                --set inputs.transcriptions_dir.path="$TRANSCRIPTIONS_URI" \
                --set outputs.results.path="$OUTPUTS_URI/eval" \
                --set environment_variables.SHARD="$i" \
                --set environment_variables.TOTAL_SHARDS="$TOTAL_SHARDS" \
                --set display_name="evaluate-${i}-of-${TOTAL_SHARDS}" \
                --query name --output tsv
        done
        echo "Submitted $TOTAL_SHARDS evaluate jobs."
        ;;

    finetune)
        echo "Submitting finetune job..."
        FINETUNE_CHECKPOINT_URI="$(join_datastore_uri "$AML_DATASTORE_BASE" "$AML_OUTPUTS_PATH/checkpoints")"
        if [[ -n "$WEATHER_CHECKPOINT" ]]; then
            FINETUNE_CHECKPOINT_URI="$(join_datastore_uri "$AML_DATASTORE_BASE" "$WEATHER_CHECKPOINT")"
            FINETUNE_CHECKPOINT_SLUG="${WEATHER_CHECKPOINT##*/}"
            CHECKPOINT_RUN_SLUG="${FINETUNE_CHECKPOINT_SLUG}-${RUN_TIMESTAMP}"
            CHECKPOINTS_REL_PATH="$AML_OUTPUTS_PATH/checkpoints/$CHECKPOINT_RUN_SLUG"
            CHECKPOINT_PATH="$CHECKPOINTS_REL_PATH/$FINETUNE_CHECKPOINT_SLUG"
        fi
        echo "  Checkpoint output: $(join_datastore_uri "$AML_DATASTORE_BASE" "$CHECKPOINTS_REL_PATH")"
        echo "  Checkpoint input mount: $FINETUNE_CHECKPOINT_URI"
        [[ -n "$WEATHER_CHECKPOINT" ]] && echo "  Input checkpoint override: $FINETUNE_CHECKPOINT_URI"
        JOB_ID=$(az ml job create \
            --file "$REPO_DIR/azureml/finetune_job.yml" \
            "${AML_ARGS[@]}" \
            --set compute="azureml:$AML_COMPUTE" \
            ${AML_ENV_OVERRIDE:+$AML_ENV_OVERRIDE} \
            --set inputs.images_dir.path="$IMAGES_URI" \
            --set inputs.transcriptions_dir.path="$TRANSCRIPTIONS_URI" \
            --set inputs.checkpoint_dir.path="$FINETUNE_CHECKPOINT_URI" \
            --set outputs.checkpoints.path="$(join_datastore_uri "$AML_DATASTORE_BASE" "$CHECKPOINTS_REL_PATH")" \
            --set outputs.hf_cache.path="$(join_datastore_uri "$AML_DATASTORE_BASE" "$AML_HF_CACHE_PATH")" \
            --set environment_variables.WEATHER_MODEL="$WEATHER_MODEL" \
            --set environment_variables.WEATHER_EPOCHS="$WEATHER_EPOCHS" \
            --set environment_variables.WEATHER_BATCH_SIZE="$BATCH_SIZE" \
            --set environment_variables.WEATHER_NUM_PROCESSES="$FINETUNE_GPU_WORKERS" \
            --set environment_variables.WEATHER_GRAD_ACCUM_STEPS="$WEATHER_GRAD_ACCUM_STEPS" \
            --set environment_variables.WEATHER_AUTO_SCALE_GRAD_ACCUM="$WEATHER_AUTO_SCALE_GRAD_ACCUM" \
            ${EXTRACT_LIMIT:+--set environment_variables.EXTRACT_LIMIT="$EXTRACT_LIMIT"} \
            ${HF_TOKEN:+--set environment_variables.HF_TOKEN="$HF_TOKEN"} \
            --set display_name="finetune-${MODEL_SLUG}" \
            --query name --output tsv)
        echo "Finetune job submitted: $JOB_ID"

        python3 "$REPO_DIR/scripts/create_model_registry_entry.py" \
            --checkpoint-path "$CHECKPOINT_PATH" \
            --base-model "$WEATHER_MODEL" \
            --dataset "$AML_IMAGES_PATH" \
            --registry-file "$MODEL_REGISTRY_FILE" \
            --job-id "$JOB_ID" \
            --status submitted \
            --training-mode standard \
                        --base-model-name-or-path "$RESOLVED_TRAINING_MODEL_NAME" \
            --notes "Auto-registered on finetune submit"

        echo "Model registry: $MODEL_REGISTRY_FILE"
        echo
        echo "Checkpoint path for extraction (once job completes):"
        echo "  bash scripts/aml_submit.sh --checkpoint $CHECKPOINT_PATH extract"
        ;;

    finetune-consensus)
        echo "Submitting consensus finetune job..."
        CONSENSUS_CHECKPOINT_PATH="$WEATHER_CHECKPOINT"
        if [[ -z "$CONSENSUS_CHECKPOINT_PATH" && "$WEATHER_MODEL" == *"/outputs/checkpoints/"* ]]; then
            CONSENSUS_CHECKPOINT_PATH="$WEATHER_MODEL"
        fi
        if [[ -z "$CONSENSUS_CHECKPOINT_PATH" ]]; then
            echo "Error: finetune-consensus requires a checkpoint path via --model <checkpoint-path> or --checkpoint <checkpoint-path>" >&2
            exit 1
        fi
        CONSENSUS_CHECKPOINT_URI="$(join_datastore_uri "$AML_DATASTORE_BASE" "$CONSENSUS_CHECKPOINT_PATH")"
        CONSENSUS_CHECKPOINT_SLUG="${CONSENSUS_CHECKPOINT_PATH##*/}"
        MODEL_SLUG="$CONSENSUS_CHECKPOINT_SLUG"
        CHECKPOINT_RUN_SLUG="${MODEL_SLUG}-${RUN_TIMESTAMP}"
        CHECKPOINTS_REL_PATH="$AML_OUTPUTS_PATH/checkpoints/$CHECKPOINT_RUN_SLUG"
        CHECKPOINT_PATH="$CHECKPOINTS_REL_PATH/$MODEL_SLUG"
        echo "  Checkpoint output: $(join_datastore_uri "$AML_DATASTORE_BASE" "$CHECKPOINTS_REL_PATH")"
        echo "  Input checkpoint: $CONSENSUS_CHECKPOINT_URI"
        echo "  Consensus transcriptions: $CONSENSUS_TRANSCRIPTIONS_URI"
        JOB_ID=$(az ml job create \
            --file "$REPO_DIR/azureml/finetune_job.yml" \
            "${AML_ARGS[@]}" \
            --set compute="azureml:$AML_COMPUTE" \
            ${AML_ENV_OVERRIDE:+$AML_ENV_OVERRIDE} \
            --set inputs.images_dir.path="$IMAGES_URI" \
            --set inputs.transcriptions_dir.path="$CONSENSUS_TRANSCRIPTIONS_URI" \
            --set inputs.checkpoint_dir.path="$CONSENSUS_CHECKPOINT_URI" \
            --set outputs.checkpoints.path="$(join_datastore_uri "$AML_DATASTORE_BASE" "$CHECKPOINTS_REL_PATH")" \
            --set outputs.hf_cache.path="$(join_datastore_uri "$AML_DATASTORE_BASE" "$AML_HF_CACHE_PATH")" \
            --set environment_variables.WEATHER_MODEL="$WEATHER_MODEL" \
            --set environment_variables.WEATHER_EPOCHS="$WEATHER_EPOCHS" \
            --set environment_variables.WEATHER_BATCH_SIZE="$BATCH_SIZE" \
            --set environment_variables.WEATHER_NUM_PROCESSES="$FINETUNE_GPU_WORKERS" \
            --set environment_variables.WEATHER_GRAD_ACCUM_STEPS="$WEATHER_GRAD_ACCUM_STEPS" \
            --set environment_variables.WEATHER_AUTO_SCALE_GRAD_ACCUM="$WEATHER_AUTO_SCALE_GRAD_ACCUM" \
            --set environment_variables.WEATHER_FINETUNE_MODE="finetune-consensus" \
            ${EXTRACT_LIMIT:+--set environment_variables.EXTRACT_LIMIT="$EXTRACT_LIMIT"} \
            ${HF_TOKEN:+--set environment_variables.HF_TOKEN="$HF_TOKEN"} \
            --set display_name="finetune-consensus-${MODEL_SLUG}" \
            --query name --output tsv)
        echo "Consensus finetune job submitted: $JOB_ID"

        python3 "$REPO_DIR/scripts/create_model_registry_entry.py" \
            --checkpoint-path "$CHECKPOINT_PATH" \
            --base-model "$WEATHER_MODEL" \
            --dataset "$AML_IMAGES_PATH" \
            --registry-file "$MODEL_REGISTRY_FILE" \
            --job-id "$JOB_ID" \
            --status submitted \
            --training-mode consensus-masked \
            --consensus-transcriptions-path "$AML_CONSENSUS_TRANSCRIPTIONS_PATH" \
                        --base-model-name-or-path "$RESOLVED_TRAINING_MODEL_NAME" \
            --notes "Consensus finetune submit"

        echo "Model registry: $MODEL_REGISTRY_FILE"
        ;;
esac
