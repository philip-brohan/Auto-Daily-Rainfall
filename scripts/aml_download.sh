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

# ── Resolve storage account and container (skipped in dry-run) ────────────────
if $DRY_RUN; then
    echo "[dry-run] Would resolve datastore '$DATASTORE_NAME' in workspace '$AML_WORKSPACE'"
    echo
else
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
# Uses az storage blob download-batch for bulk parallel transfer (much faster
# than per-blob downloads).  Zero-size "directory marker" blobs can cause
# download-batch to create empty files that block subsequent directory
# creation; we remove any such zero-byte files afterwards.
do_download() {
    local src_path="$1"   # path prefix in the container, e.g. foo/outputs/extractions
    local dst="$2"        # local destination directory
    mkdir -p "$dst"
    if $DRY_RUN; then
        echo "[dry-run] az storage blob download-batch \\"
        echo "    --account-name <account> --auth-mode login \\"
        echo "    --source <container> --pattern '${src_path}/*' \\"
        echo "    --destination <tmpdir> --max-connections ${DOWNLOAD_JOBS} --overwrite true"
        echo "    (then move <tmpdir>/${src_path}/ → $dst)"
    else
        echo "Downloading from: ${src_path}"
        echo "             to:  $dst"
        echo "         workers: ${DOWNLOAD_JOBS}"
        local tmp_dir
        tmp_dir=$(mktemp -d)
        az storage blob download-batch \
            --account-name "$STORAGE_ACCOUNT" \
            --auth-mode login \
            --source "$CONTAINER" \
            --destination "$tmp_dir" \
            --pattern "${src_path}/*" \
            --max-connections "${DOWNLOAD_JOBS}" \
            --overwrite true \
            $($QUIET && echo "--only-show-errors" || true)
        # Move contents from prefix subdir to final destination, removing
        # any zero-byte directory marker files left by download-batch.
        find "$tmp_dir/${src_path}" -type f -empty -delete
        mkdir -p "$dst"
        cp -a "$tmp_dir/${src_path}/." "$dst/"
        rm -rf "$tmp_dir"
        echo "Done."
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
