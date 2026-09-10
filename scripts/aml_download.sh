#!/usr/bin/env bash
# aml_download.sh — Download data from the Azure ML workspace datastore.
#
# Sources azureml/config.env for workspace coordinates and default paths.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/aml_download.sh [what] [options]
#
# ── What to download (pick one or more) ──────────────────────────────────────
#   extractions           Download $AML_OUTPUTS_PATH/extractions → outputs/extractions/
#   eval                  Download $AML_OUTPUTS_PATH/eval        → outputs/eval/
#   checkpoints           Download $AML_OUTPUTS_PATH/checkpoints → outputs/checkpoints/
#   all                   Download all three output directories
#   --src PATH --dst DIR  Download any datastore path to a custom local directory
#
# ── Options ───────────────────────────────────────────────────────────────────
#   --run-name NAME       Download extractions from a specific run (look up in registry)
#   --output-dir DIR      Root local output directory (default: outputs/)
#   --jobs N              Parallel download connections for download-batch (default: 16)
#   --quiet               Print only summary/error lines (faster on huge runs)
#   --dry-run             Print az storage commands without executing them
#   --help
#
# ── Examples ──────────────────────────────────────────────────────────────────
#   bash scripts/aml_download.sh extractions
#   bash scripts/aml_download.sh all
#   bash scripts/aml_download.sh --src my_project/outputs/eval --dst /var/tmp/eval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$REPO_DIR/azureml/config.env"

[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"

AML_SUBSCRIPTION="${AML_SUBSCRIPTION:-}"
AML_RESOURCE_GROUP="${AML_RESOURCE_GROUP:-}"
AML_WORKSPACE="${AML_WORKSPACE:-}"
AML_DATASTORE_BASE="${AML_DATASTORE_BASE:-azureml://datastores/workspaceblobstore/paths}"
AML_OUTPUTS_PATH="${AML_OUTPUTS_PATH:-outputs}"

OUTPUT_DIR="${REPO_DIR}/outputs"
DOWNLOAD_JOBS="${DOWNLOAD_JOBS:-16}"
RUN_NAME=""
EXTRACTION_REGISTRY="${REPO_DIR}/outputs/extraction_registry.json"
QUIET=false
CUSTOM_SRC=""
CUSTOM_DST=""
DRY_RUN=false
TARGETS=()

usage() {
    sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        extractions|eval|checkpoints|all) TARGETS+=("$1"); shift ;;
        --src)             CUSTOM_SRC="$2"; shift 2 ;;
        --dst)             CUSTOM_DST="$2"; shift 2 ;;
        --run-name)        RUN_NAME="$2"; shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
        --jobs)            DOWNLOAD_JOBS="$2"; shift 2 ;;
        --quiet)           QUIET=true; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --help|-h)         usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

# ── Handle --run-name lookup ──────────────────────────────────────────────────
if [[ -n "$RUN_NAME" ]]; then
    if [[ ! -f "$EXTRACTION_REGISTRY" ]]; then
        echo "Error: extraction registry not found at $EXTRACTION_REGISTRY" >&2
        exit 1
    fi
    EXTRACTIONS_PATH=$(python3 -c "
import json, sys
try:
    registry = json.load(open('$EXTRACTION_REGISTRY'))
    for entry in registry.get('extractions', []):
        if entry.get('run_name') == '$RUN_NAME':
            print(entry.get('extractions_path', ''))
            sys.exit(0)
    print('', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'Error reading registry: {e}', file=sys.stderr)
    sys.exit(1)
")
    if [[ -z "$EXTRACTIONS_PATH" ]]; then
        echo "Error: run_name '$RUN_NAME' not found in $EXTRACTION_REGISTRY" >&2
        exit 1
    fi
    TARGETS=()
    CUSTOM_SRC="$EXTRACTIONS_PATH"
    [[ -z "$CUSTOM_DST" ]] && CUSTOM_DST="$OUTPUT_DIR/extractions/$RUN_NAME"
fi

if [[ ${#TARGETS[@]} -eq 0 && -z "$CUSTOM_SRC" ]]; then
    echo "Error: specify what to download (extractions|eval|checkpoints|all) or --src/--dst" >&2
    usage 1
fi

[[ -z "$AML_SUBSCRIPTION" ]]   && { echo "Error: AML_SUBSCRIPTION not set" >&2; exit 1; }
[[ -z "$AML_RESOURCE_GROUP" ]] && { echo "Error: AML_RESOURCE_GROUP not set" >&2; exit 1; }
if ! $DRY_RUN; then
    [[ -z "$AML_SUBSCRIPTION" ]]   && { echo "Error: AML_SUBSCRIPTION not set" >&2; exit 1; }
    [[ -z "$AML_RESOURCE_GROUP" ]] && { echo "Error: AML_RESOURCE_GROUP not set" >&2; exit 1; }
    [[ -z "$AML_WORKSPACE" ]]      && { echo "Error: AML_WORKSPACE not set" >&2; exit 1; }
fi

DATASTORE_NAME="$(echo "$AML_DATASTORE_BASE" | sed 's|.*/datastores/||;s|/paths.*||')"
STORAGE_ACCOUNT=""
CONTAINER=""

# azcopy reuses the current `az login` credentials and honours DOWNLOAD_JOBS as
# its transfer concurrency.
export AZCOPY_AUTO_LOGIN_TYPE="${AZCOPY_AUTO_LOGIN_TYPE:-AZCLI}"
export AZCOPY_CONCURRENCY_VALUE="${AZCOPY_CONCURRENCY_VALUE:-$DOWNLOAD_JOBS}"

# ── Resolve storage account and container (skipped in dry-run) ────────────────
if $DRY_RUN; then
    echo "[dry-run] Would resolve datastore '$DATASTORE_NAME' in workspace '$AML_WORKSPACE'"
    echo
else
    if ! command -v azcopy >/dev/null 2>&1; then
        echo "Error: azcopy not found on PATH." >&2
        echo "       azcopy is required to download folder-marker blobs correctly." >&2
        echo "       Install it from https://aka.ms/downloadazcopy and retry." >&2
        exit 1
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
    echo
fi

# ── Download helper ───────────────────────────────────────────────────────────
# Uses azcopy for bulk parallel transfer.  Azure ML writes a zero-byte
# "hdi_isfolder" marker blob for every folder level (a blob named <dir>
# alongside the real files under <dir>/...).  `az storage blob download-batch`
# downloads such a marker as a file and then aborts with "[Errno 20] Not a
# directory" when it needs that same name as a directory.  azcopy understands
# these markers and materialises them as directories.  --as-subdir=false places
# the source contents directly under $dst; --overwrite=ifSourceNewer makes
# re-runs idempotent and resumable.
do_download() {
    local src_path="$1"   # path prefix in the container, e.g. foo/outputs/extractions
    local dst="$2"        # local destination directory
    local url="https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER}/${src_path}"
    mkdir -p "$dst"
    if $DRY_RUN; then
        echo "[dry-run] azcopy copy \\"
        echo "    'https://<account>.blob.core.windows.net/<container>/${src_path}' \\"
        echo "    '$dst' --recursive --as-subdir=false --overwrite=ifSourceNewer"
    else
        echo "Downloading from: ${src_path}"
        echo "             to:  $dst"
        echo "         workers: ${DOWNLOAD_JOBS}"
        # Skip paths with no blobs so azcopy does not error on an empty source.
        local found
        found="$(az storage blob list \
            --account-name "$STORAGE_ACCOUNT" \
            --auth-mode login \
            --container-name "$CONTAINER" \
            --prefix "${src_path}/" \
            --num-results 1 \
            --query "[0].name" \
            --output tsv 2>/dev/null)"
        if [[ -z "$found" ]]; then
            echo "  (no blobs found under ${src_path}/ — skipping)"
        else
            azcopy copy "$url" "$dst" \
                --recursive \
                --as-subdir=false \
                --overwrite=ifSourceNewer \
                --output-level "$($QUIET && echo quiet || echo essential)"
            echo "Done."
        fi
    fi
    echo
}

# ── Run downloads ─────────────────────────────────────────────────────────────
for target in "${TARGETS[@]}"; do
    case "$target" in
        extractions)
            do_download "$AML_OUTPUTS_PATH/extractions" "$OUTPUT_DIR/extractions"
            ;;
        eval)
            do_download "$AML_OUTPUTS_PATH/eval" "$OUTPUT_DIR/eval"
            ;;
        checkpoints)
            do_download "$AML_OUTPUTS_PATH/checkpoints" "$OUTPUT_DIR/checkpoints"
            ;;
        all)
            do_download "$AML_OUTPUTS_PATH/extractions" "$OUTPUT_DIR/extractions"
            do_download "$AML_OUTPUTS_PATH/eval"        "$OUTPUT_DIR/eval"
            do_download "$AML_OUTPUTS_PATH/checkpoints" "$OUTPUT_DIR/checkpoints"
            ;;
    esac
done

if [[ -n "$CUSTOM_SRC" ]]; then
    [[ -z "$CUSTOM_DST" ]] && { echo "Error: --dst required with --src" >&2; exit 1; }
    do_download "$CUSTOM_SRC" "$CUSTOM_DST"
fi
