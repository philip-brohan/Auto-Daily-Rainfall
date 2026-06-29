#!/usr/bin/env bash
# list_checkpoints.sh — List registered fine-tuned checkpoints and managed extraction commands

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

usage() {
    cat <<EOF
list_checkpoints.sh — List and manage fine-tuned model checkpoints

Usage:
  bash scripts/list_checkpoints.sh [OPTIONS]

Options:
  --registry FILE         Path to model_registry.json (default: outputs/model_registry.json)
  --format json|table     Output format (default: table)
  --help                  Show this message

Examples:
  # List all registered checkpoints
  bash scripts/list_checkpoints.sh

  # List as JSON
  bash scripts/list_checkpoints.sh --format json

  # Use custom registry file
  bash scripts/list_checkpoints.sh --registry /path/to/registry.json
EOF
    exit "${1:-0}"
}

REGISTRY_FILE="outputs/model_registry.json"
FORMAT="table"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry) REGISTRY_FILE="$2"; shift 2 ;;
        --format)   FORMAT="$2";        shift 2 ;;
        --help|-h)  usage 0 ;;
        *)          echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

if [[ ! -f "$REGISTRY_FILE" ]]; then
    echo "Error: Registry file not found: $REGISTRY_FILE"
    echo "No checkpoints registered yet. Run 'bash scripts/aml_submit.sh finetune' first."
    exit 1
fi

case "$FORMAT" in
    json)
        cat "$REGISTRY_FILE"
        ;;
    table)
        echo "Registered Checkpoints:"
        echo "───────────────────────────────────────────────────────────────────────────────────────────"
        python3 << PYTHON
import json
from pathlib import Path

registry_file = Path("$REGISTRY_FILE")
if not registry_file.exists():
    print("Registry file not found: $REGISTRY_FILE")
    exit(1)

with open(registry_file) as f:
    registry = json.load(f)

if not registry.get("models"):
    print("No checkpoints registered.")
else:
    # Print header
    print(f"{'Checkpoint Name':<34} {'Base Model':<12} {'Dataset':<24} {'Status':<10} {'Created':<12}")
    print("─" * 96)
    
    for model in registry["models"]:
        name = model.get("checkpoint_name", "unknown")
        base = model.get("base_model", "unknown")
        dset = model.get("dataset", "unknown")
        status = model.get("status", "completed")
        created = model.get("created_at", "unknown")[:10]  # Date only
        print(f"{name:<34} {base:<12} {dset:<24} {status:<10} {created:<12}")

    print()
    print("Job IDs:")
    for i, model in enumerate(registry["models"], 1):
        job_id = model.get("job_id") or "(not recorded)"
        print(f"  {i}. {job_id}")
    
    print()
    print("Extract using a checkpoint:")
    for i, model in enumerate(registry["models"], 1):
        path = model.get("checkpoint_path", "unknown")
        print(f"  {i}. bash scripts/aml_submit.sh --checkpoint {path} extract")
PYTHON
        ;;
    *)
        echo "Unknown format: $FORMAT" >&2
        exit 1
        ;;
esac
