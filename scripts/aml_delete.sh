#!/usr/bin/env bash
# aml_delete.sh — Delete contents of a directory in the Azure ML datastore.
#
# ENVIRONMENT: Run in weather-doc-extractor conda environment
#   conda activate weather-doc-extractor
#
# Sources azureml/config.env for workspace coordinates and datastore settings.
#
# Usage:
#   bash scripts/aml_delete.sh <datastore-directory> [--dry-run]
#
# Example:
#   bash scripts/aml_delete.sh fake_daily_rainfall_2
#   bash scripts/aml_delete.sh fake_daily_rainfall_2/images --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$REPO_DIR/azureml/config.env"

[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"

AML_SUBSCRIPTION="${AML_SUBSCRIPTION:-}"
AML_RESOURCE_GROUP="${AML_RESOURCE_GROUP:-}"
AML_WORKSPACE="${AML_WORKSPACE:-}"
AML_DATASTORE_BASE="${AML_DATASTORE_BASE:-azureml://datastores/workspaceblobstore/paths}"

TARGET_DIR=""
DRY_RUN=false

usage() {
    sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage 0 ;;
        -*)
            echo "Unknown option: $1" >&2
            usage 1
            ;;
        *)
            if [[ -n "$TARGET_DIR" ]]; then
                echo "Error: only one datastore directory argument is allowed" >&2
                usage 1
            fi
            TARGET_DIR="$1"
            shift
            ;;
    esac
done

if [[ -z "$TARGET_DIR" ]]; then
    echo "Error: missing datastore directory argument" >&2
    usage 1
fi

# Normalise the target path to avoid accidental broad patterns.
TARGET_DIR="${TARGET_DIR#/}"
TARGET_DIR="${TARGET_DIR%/}"

if [[ -z "$TARGET_DIR" || "$TARGET_DIR" == "." ]]; then
    echo "Error: refusing to delete root/empty datastore path" >&2
    exit 1
fi

[[ -z "$AML_SUBSCRIPTION" ]]   && { echo "Error: AML_SUBSCRIPTION not set" >&2; exit 1; }
[[ -z "$AML_RESOURCE_GROUP" ]] && { echo "Error: AML_RESOURCE_GROUP not set" >&2; exit 1; }
if ! $DRY_RUN; then
    [[ -z "$AML_WORKSPACE" ]] && { echo "Error: AML_WORKSPACE not set" >&2; exit 1; }
fi

DATASTORE_NAME="$(echo "$AML_DATASTORE_BASE" | sed 's|.*/datastores/||;s|/paths.*||')"

if $DRY_RUN; then
    echo "[dry-run] Would resolve datastore '$DATASTORE_NAME' in workspace '$AML_WORKSPACE'"
    echo "[dry-run] Would delete blobs matching pattern: ${TARGET_DIR}/*"
    echo "[dry-run] Command:"
    echo "  az storage blob delete-batch --account-name <resolved-at-runtime> --auth-mode login --source <container> --pattern '${TARGET_DIR}/*'"
    exit 0
fi

echo "Resolving datastore '$DATASTORE_NAME' in workspace '$AML_WORKSPACE'..."
DATASTORE_JSON="$(az ml datastore show \
    --name "$DATASTORE_NAME" \
    --workspace-name "$AML_WORKSPACE" \
    --resource-group "$AML_RESOURCE_GROUP" \
    --subscription "$AML_SUBSCRIPTION" \
    --output json)"

STORAGE_ACCOUNT="$(echo "$DATASTORE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['account_name'])")"
CONTAINER="$(echo "$DATASTORE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['container_name'])")"

echo "Storage account: $STORAGE_ACCOUNT  container: $CONTAINER"
echo "Deleting contents under datastore path: ${TARGET_DIR}/"

az storage blob delete-batch \
    --account-name "$STORAGE_ACCOUNT" \
    --auth-mode login \
    --source "$CONTAINER" \
    --pattern "${TARGET_DIR}/*"

echo "Done. Deleted contents under ${TARGET_DIR}/"
