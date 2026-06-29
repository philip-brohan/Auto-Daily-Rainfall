#!/bin/bash
#
# Orchestrate a complete consensus variant pipeline.
#
# ENVIRONMENT: Run this script in the weather-doc-extractor conda environment
#   conda activate weather-doc-extractor
#
# Usage:
#   ./scripts/run_consensus_pipeline.sh \
#     --dataset-root <path> \
#     --variant-name <name> \
#     [--threshold N] [--precision P] [--validate] \
#     --extraction-dir <path> --extraction-dir <path> ... [--extraction-dir <path>]
#
# Example:
#   ./scripts/run_consensus_pipeline.sh \
#     --dataset-root outputs/consensus_dataset_1000 \
#     --variant-name consensus_1000 \
#     --threshold 4 \
#     --precision 3 \
#     --validate \
#     --extraction-dir outputs/extractions/model_A/run_001 \
#     --extraction-dir outputs/extractions/model_B/run_002 \
#     --extraction-dir outputs/extractions/model_C/run_003 \
#     --extraction-dir outputs/extractions/model_D/run_004 \
#     --extraction-dir outputs/extractions/model_E/run_005
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") \\
    --dataset-root <path> \\
    --variant-name <name> \\
    [--threshold N] [--null-threshold N] [--precision P] [--validate] [--validation-sample-denominator N] [--no-config-check] \\
    --extraction-dir <path> [--extraction-dir <path> ...]

Options:
Required options:
    --dataset-root <path>      Root directory for consensus output (creates if missing)
    --variant-name <name>      Name for this variant (e.g., 'consensus_1000')
    --extraction-dir <path>    Extraction directory (repeat for multiple models, typically 5)

Optional options:
    --threshold N              Agreement threshold (default: 4)
    --null-threshold N         Null-value agreement threshold (default: same as --threshold)
    --precision P              Precision for decimals (default: 3)
    --validate                 Generate validation figures
    --images-dir <path>        Directory containing source images for validation
                               (default: <dataset-root>/images). Specify separately if images are elsewhere.
    --validation-sample-denominator N
                               When validating, generate a deterministic 1/N sample of figures
                               (default: 20; use 1 for all figures)
    --ground-truth-dir <path>  Directory containing ground-truth transcriptions.
                               When provided with --validate, figures use 4-category colouring
    --no-config-check          Skip overwrite confirmation if config exists

Example:
  $(basename "$0") \\
    --dataset-root outputs/consensus_dataset_1000 \\
    --variant-name consensus_1000 \\
    --threshold 4 \\
    --precision 3 \\
    --validate \\
    --extraction-dir outputs/extractions/model_A/run_001 \\
    --extraction-dir outputs/extractions/model_B/run_002 \\
    --extraction-dir outputs/extractions/model_C/run_003 \\
    --extraction-dir outputs/extractions/model_D/run_004 \\
    --extraction-dir outputs/extractions/model_E/run_005
EOF
    exit 1
}

main() {
    local dataset_root=""
    local variant_name=""
    local threshold=4
    local null_threshold=""
    local precision=3
    local validate=false
    local validation_sample_denominator=20
    local images_dir=""
    local ground_truth_dir=""
    local no_config_check=false
    local extraction_dirs=()

    # Parse options
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dataset-root)
                dataset_root="$2"
                shift 2
                ;;
            --variant-name)
                variant_name="$2"
                shift 2
                ;;
            --threshold)
                threshold="$2"
                shift 2
                ;;
            --null-threshold)
                null_threshold="$2"
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
            --validation-sample-denominator)
                validation_sample_denominator="$2"
                shift 2
                ;;
            --ground-truth-dir)
                ground_truth_dir="$2"
                shift 2
                ;;
            --images-dir)
                images_dir="$2"
                shift 2
                ;;
            --no-config-check)
                no_config_check=true
                shift
                ;;
            --extraction-dir)
                extraction_dirs+=("$2")
                shift 2
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}" >&2
                print_usage
                ;;
        esac
    done

    # Validate inputs
    if [[ -z "$dataset_root" ]]; then
        echo -e "${RED}Error: --dataset-root is required${NC}" >&2
        print_usage
    fi

    if [[ -e "$dataset_root" && ! -d "$dataset_root" ]]; then
        echo -e "${RED}Error: --dataset-root exists but is not a directory: $dataset_root${NC}" >&2
        exit 1
    fi

    if [[ ! -d "$dataset_root" ]]; then
        echo -e "${YELLOW}Warning: dataset-root not found; creating: $dataset_root${NC}"
        mkdir -p "$dataset_root"
    fi

    if [[ -z "$variant_name" ]]; then
        echo -e "${RED}Error: --variant-name is required${NC}" >&2
        print_usage
    fi

    if [[ ${#extraction_dirs[@]} -eq 0 ]]; then
        echo -e "${RED}Error: no extraction directories provided${NC}" >&2
        print_usage
    fi

    if ! [[ "$threshold" =~ ^[0-9]+$ ]] || [[ "$threshold" -lt 1 ]]; then
        echo -e "${RED}Error: --threshold must be an integer >= 1${NC}" >&2
        print_usage
    fi

    if [[ -z "$null_threshold" ]]; then
        null_threshold="$threshold"
    fi

    if ! [[ "$null_threshold" =~ ^[0-9]+$ ]] || [[ "$null_threshold" -lt 1 ]]; then
        echo -e "${RED}Error: --null-threshold must be an integer >= 1${NC}" >&2
        print_usage
    fi

    if ! [[ "$validation_sample_denominator" =~ ^[0-9]+$ ]] || [[ "$validation_sample_denominator" -lt 1 ]]; then
        echo -e "${RED}Error: --validation-sample-denominator must be an integer >= 1${NC}" >&2
        print_usage
    fi

    for ext_dir in "${extraction_dirs[@]}"; do
        if [[ ! -d "$ext_dir" ]]; then
            echo -e "${RED}Error: extraction directory not found: $ext_dir${NC}" >&2
            exit 1
        fi
    done

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local repo_root="$(dirname "$script_dir")"

    # Ensure Python scripts are executable
    chmod +x "$script_dir"/create_consensus_config.py
    chmod +x "$script_dir"/build_consensus_transcriptions.py
    if [[ "$validate" == true ]]; then
        chmod +x "$script_dir"/validate_consensus.py
    fi

    echo -e "${GREEN}=== Consensus Pipeline ===${NC}"
    echo "Dataset root:      $dataset_root"
    echo "Variant name:      $variant_name"
    echo "Threshold:         $threshold"
    echo "Null threshold:    $null_threshold"
    echo "Precision:         $precision"
    echo "Extraction dirs:   ${#extraction_dirs[@]}"
    echo "Validate:          $validate"
    if [[ "$validate" == true ]]; then
        echo "Validation sample: 1/$validation_sample_denominator"
    fi
    echo

    local variant_dir="$dataset_root/$variant_name"
    local config_file="$variant_dir/consensus_config.json"

    # Create or check config
    if [[ -f "$config_file" ]] && [[ "$no_config_check" != true ]]; then
        echo -e "${YELLOW}Warning: config already exists at $config_file${NC}"
        # Only prompt if running interactively (stdin is a terminal)
        if [[ -t 0 ]]; then
            read -p "Overwrite config? (y/N) " -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Skipping config creation."
            else
                echo -e "${GREEN}Creating config...${NC}"
                python3 "$script_dir"/create_consensus_config.py \
                    --variant-name "$variant_name" \
                    --dataset-root "$dataset_root" \
                    --agreement-threshold "$threshold" \
                    --null-threshold "$null_threshold" \
                    --precision "$precision" \
                    --overwrite \
                    --extraction-dirs "${extraction_dirs[@]}"
            fi
        else
            echo "Running non-interactively; overwriting existing config."
            echo -e "${GREEN}Creating config...${NC}"
            python3 "$script_dir"/create_consensus_config.py \
                --variant-name "$variant_name" \
                --dataset-root "$dataset_root" \
                --agreement-threshold "$threshold" \
                --null-threshold "$null_threshold" \
                --precision "$precision" \
                --overwrite \
                --extraction-dirs "${extraction_dirs[@]}"
        fi
    else
        echo -e "${GREEN}Creating config...${NC}"
        python3 "$script_dir"/create_consensus_config.py \
            --variant-name "$variant_name" \
            --dataset-root "$dataset_root" \
            --agreement-threshold "$threshold" \
            --null-threshold "$null_threshold" \
            --precision "$precision" \
            --extraction-dirs "${extraction_dirs[@]}"
    fi

    echo

    # Build consensus
    echo -e "${GREEN}Building consensus transcriptions...${NC}"
    python3 "$script_dir"/build_consensus_transcriptions.py \
        --config-file "$config_file"

    echo
    echo -e "${GREEN}✓ Consensus built successfully${NC}"
    echo "  Config:           $config_file"
    echo "  Consensus output: $variant_dir/consensus_transcriptions/"
    echo "  Summary:          $variant_dir/consensus_summary.json"
    echo

    # Optionally validate
    if [[ "$validate" == true ]]; then
        echo -e "${GREEN}Generating validation figures...${NC}"
        python3 "$script_dir"/validate_consensus.py \
            --config-file "$config_file" \
            --sample-denominator "$validation_sample_denominator" \
            $(if [[ -n "$images_dir" ]]; then echo "--images-dir $images_dir"; fi) \
            $(if [[ -n "$ground_truth_dir" ]]; then echo "--ground-truth-dir $ground_truth_dir"; fi)
        echo -e "${GREEN}✓ Validation figures saved${NC}"
        echo "  Figures: $variant_dir/validation_figures/"
        echo
    fi

    echo -e "${GREEN}=== Pipeline Complete ===${NC}"
}

main "$@"
