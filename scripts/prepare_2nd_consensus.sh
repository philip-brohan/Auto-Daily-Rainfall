#!/bin/bash
#
# Prepare and orchestrate a 2nd-order consensus variant.
#
# ENVIRONMENT: Run this script in the weather-doc-extractor conda environment
#   conda activate weather-doc-extractor
#
# This is a simplified wrapper around run_consensus_pipeline.sh that:
# - Uses the same images/transcriptions from an existing dataset
# - Takes checkpoint directories (typically fine-tuned models)
# - Runs extractions on those checkpoints
# - Builds consensus from the new extractions
#
# Usage:
#   ./scripts/prepare_2nd_consensus.sh <existing-dataset-root> <new-variant-name> \
#     [--threshold N] [--precision P] [--validate] \
#     <checkpoint-1> <checkpoint-2> ... [<checkpoint-5>]
#
# Example:
#   ./scripts/prepare_2nd_consensus.sh outputs/consensus_dataset_1000 consensus_1000_ft \
#     --threshold 4 --precision 3 --validate \
#     outputs/checkpoints/checkpoint_1 \
#     outputs/checkpoints/checkpoint_2 \
#     outputs/checkpoints/checkpoint_3 \
#     outputs/checkpoints/checkpoint_4 \
#     outputs/checkpoints/checkpoint_5
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") <existing-dataset-root> <new-variant-name> \\
  [--threshold N] [--precision P] [--validate] \\
  <checkpoint-1> <checkpoint-2> ... [<checkpoint-5>]

Options:
  --threshold N     Agreement threshold (default: 4)
  --precision P     Decimal precision (default: 3)
  --validate        Generate validation figures

Arguments:
  existing-dataset-root   Dataset with images/transcriptions already sampled
  new-variant-name        Name for this consensus variant
  checkpoint-N            Checkpoint directories to use for extraction

Prerequisites:
  - The extraction orchestration framework (run_extract.sh or Azure jobs) must be set up separately
  - Extraction outputs must be downloaded to outputs/extractions/<model>/<timestamp>/

Example:
  $(basename "$0") outputs/consensus_dataset_1000 consensus_1000_ft \\
    --threshold 4 --validate \\
    outputs/checkpoints/ft_model_1 \\
    outputs/checkpoints/ft_model_2 \\
    outputs/checkpoints/ft_model_3 \\
    outputs/checkpoints/ft_model_4 \\
    outputs/checkpoints/ft_model_5
EOF
    exit 1
}

main() {
    if [[ $# -lt 2 ]]; then
        print_usage
    fi

    local existing_dataset="$1"
    local variant_name="$2"
    shift 2

    local threshold=4
    local precision=3
    local validate=false

    # Parse options
    local checkpoints=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --threshold)
                threshold="$2"
                shift 2
                ;;
            --precision)
                precision="$2"
                shift 2
                ;;
            --validate)
                validate=true
                shift
                ;;
            *)
                checkpoints+=("$1")
                shift
                ;;
        esac
    done

    # Validate inputs
    if [[ ! -d "$existing_dataset" ]]; then
        echo -e "${RED}Error: existing dataset not found: $existing_dataset${NC}" >&2
        exit 1
    fi

    if [[ ! -d "$existing_dataset/images" ]]; then
        echo -e "${RED}Error: images directory not found: $existing_dataset/images${NC}" >&2
        exit 1
    fi

    if [[ ${#checkpoints[@]} -lt 5 ]]; then
        echo -e "${RED}Error: expected at least 5 checkpoint directories, got ${#checkpoints[@]}${NC}" >&2
        print_usage
    fi

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    echo -e "${GREEN}=== 2nd Order Consensus Setup ===${NC}"
    echo "Existing dataset:    $existing_dataset"
    echo "New variant:         $variant_name"
    echo "Threshold:           $threshold"
    echo "Precision:           $precision"
    echo "Validate:            $validate"
    echo "Using checkpoints:   ${#checkpoints[@]}"
    echo

    # NOTE: This script does NOT automatically run extractions.
    # The user must independently:
    # 1. Submit extraction jobs using checkpoints via run_consensus_extractions.sh or Azure CLI
    # 2. Download/await extraction results
    # 3. Verify extraction directories exist before calling run_consensus_pipeline.sh

    echo -e "${YELLOW}IMPORTANT: This script only PREPARES the pipeline.${NC}"
    echo -e "${YELLOW}You must separately run extractions on the checkpoints:${NC}"
    echo
    echo "  For Azure submission, use:"
    echo "    scripts/run_consensus_extractions.sh --variant $variant_name \\"
    for i in "${!checkpoints[@]}"; do
        echo "      --checkpoint${i} \"${checkpoints[$i]}\" \\"
    done
    echo
    echo "  OR run locally with:"
    echo "    scripts/run_extract.sh --dataset outputs/consensus_dataset_1000 \\"
    echo "      --model <model_family> --checkpoint <ckpt_path>"
    echo
    echo "After extraction completes:"
    echo "  1. Download extractions to outputs/extractions/<model_slug>/<timestamp>/"
    echo "  2. Call run_consensus_pipeline.sh with the extraction directories"
    echo
    read -p "Continue? (y/N) " -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi

    echo -e "${GREEN}=== Awaiting Extraction Results ===${NC}"
    echo "Once extractions complete, run:"
    echo
    echo "  $script_dir/run_consensus_pipeline.sh \\
        $existing_dataset $variant_name \\"
    echo "    --threshold $threshold --precision $precision"
    if [[ "$validate" == true ]]; then
        echo "    --validate \\"
    fi
    echo "    -- \\"
    echo "    <extraction_dir_1> <extraction_dir_2> ... <extraction_dir_5>"
    echo
}

main "$@"
