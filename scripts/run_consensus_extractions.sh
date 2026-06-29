#!/usr/bin/env bash
# run_consensus_extractions.sh — submit extraction jobs for fixed 5 checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

usage() {
    cat <<EOF
Usage:
  bash scripts/run_consensus_extractions.sh [options]

Options:
  --images-path PATH            Datastore-relative images path (required)
  --transcriptions-path PATH    Datastore-relative transcriptions path for registry metadata
                                (default: test_data/real/transcriptions)
  --limit N                     Optional per-shard image limit
  --total-shards N              Number of extraction shards (default: 1)
    --batch-size N                Batch size per job (default: 10)
    --retry-batch-size N          Stage-2 retry batch size for parse failures
                                                                (default: min(batch-size, 4))
  --compute NAME                AML compute override
  --extraction-registry FILE    Local extraction registry path
                                (default: outputs/extraction_registry.json)
  --dry-run                     Print commands without running
  --help                        Show this message

Notes:
  - Uses fixed checkpoint set:
      1) Daily_rainfall_sample/outputs/checkpoints/smolvlm2-20260601-154723/HuggingFaceTB--SmolVLM2-2.2B-Instruct
      2) Daily_rainfall_sample/outputs/checkpoints/granite4-20260601-121821/ibm-granite--granite-vision-4.1-4b
      3) Daily_rainfall_sample/outputs/checkpoints/gemma3-20260601-121832/google--gemma-3-4b-it
      4) Daily_rainfall_sample/outputs/checkpoints/gemma4-20260601-121840/google--gemma-4-E4B-it
      5) Daily_rainfall_sample/outputs/checkpoints/ministral-20260601-121858/mistralai--Mistral-Small-3.1-24B-Instruct-2503
EOF
}

IMAGES_PATH=""
TRANSCRIPTIONS_PATH="test_data/real/transcriptions"
LIMIT=""
TOTAL_SHARDS="1"
BATCH_SIZE="10"
RETRY_BATCH_SIZE=""
COMPUTE=""
EXTRACTION_REGISTRY="outputs/extraction_registry.json"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --images-path) IMAGES_PATH="$2"; shift 2 ;;
        --transcriptions-path) TRANSCRIPTIONS_PATH="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --total-shards) TOTAL_SHARDS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --retry-batch-size) RETRY_BATCH_SIZE="$2"; shift 2 ;;
        --compute) COMPUTE="$2"; shift 2 ;;
        --extraction-registry) EXTRACTION_REGISTRY="$2"; shift 2 ;;
        --dry-run) DRY_RUN="1"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$IMAGES_PATH" ]]; then
    echo "Error: --images-path is required" >&2
    usage
    exit 1
fi

CHECKPOINTS=(
    "Daily_rainfall_sample/outputs/checkpoints/smolvlm2-20260601-154723/HuggingFaceTB--SmolVLM2-2.2B-Instruct"
    "Daily_rainfall_sample/outputs/checkpoints/granite4-20260601-121821/ibm-granite--granite-vision-4.1-4b"
    "Daily_rainfall_sample/outputs/checkpoints/gemma3-20260601-121832/google--gemma-3-4b-it"
    "Daily_rainfall_sample/outputs/checkpoints/gemma4-20260601-121840/google--gemma-4-E4B-it"
    "Daily_rainfall_sample/outputs/checkpoints/ministral-20260601-121858/mistralai--Mistral-Small-3.1-24B-Instruct-2503"
)

for checkpoint in "${CHECKPOINTS[@]}"; do
    cmd=(
        bash "$REPO_DIR/scripts/aml_submit.sh"
        --checkpoint "$checkpoint"
        --images-path "$IMAGES_PATH"
        --transcriptions-path "$TRANSCRIPTIONS_PATH"
        --total-shards "$TOTAL_SHARDS"
        --batch-size "$BATCH_SIZE"
        --extraction-registry "$EXTRACTION_REGISTRY"
    )

    if [[ -n "$RETRY_BATCH_SIZE" ]]; then
        cmd+=(--retry-batch-size "$RETRY_BATCH_SIZE")
    fi

    if [[ -n "$LIMIT" ]]; then
        cmd+=(--limit "$LIMIT")
    fi
    if [[ -n "$COMPUTE" ]]; then
        cmd+=(--compute "$COMPUTE")
    fi

    cmd+=(extract)

    echo "=== Consensus extraction submit ==="
    echo "Checkpoint: $checkpoint"
    printf 'Command: '
    printf '%q ' "${cmd[@]}"
    echo

    if [[ "$DRY_RUN" == "0" ]]; then
        "${cmd[@]}"
    fi

done

echo "Done submitting consensus extraction jobs."
