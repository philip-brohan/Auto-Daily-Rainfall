#!/usr/bin/env bash
# aml_backup.sh — Back up all irreplaceable Azure ML workspace contents to local disk.
#
# ENVIRONMENT: Run in the weather-doc-extractor conda environment.
#   conda activate weather-doc-extractor
#   bash scripts/aml_backup.sh --dest /large/capacity/workspace_backup
#
# Use this BEFORE tearing down the Azure ML workspace.  Because a full teardown
# destroys the storage account backing the datastore, every datastore blob is
# lost; this script pulls them (and the local registries/config/environment
# specs) into a self-contained backup bundle that scripts/aml_restore.sh can
# replay into a freshly created bare workspace.
#
# Sources azureml/config.env for workspace coordinates and default paths.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/aml_backup.sh --dest DIR [options]
#
# The backup directory is REQUIRED and must be given explicitly (via --dest or
# the AML_BACKUP_DIR environment variable).  The bundle can be hundreds of GB,
# so point it at a large-capacity location, not the repository.
#
# ── What is backed up ─────────────────────────────────────────────────────────
#   Datastore paths (blobs pulled to <dest>/datastore/<path>):
#     $AML_TRANSCRIPTIONS_PATH        ground-truth transcriptions
#     $AML_OUTPUTS_PATH/checkpoints   fine-tuned checkpoints
#     $AML_OUTPUTS_PATH/extractions   extraction results
#     $AML_OUTPUTS_PATH/eval          evaluation metrics
#     consensus_data                  consensus datasets
#     test_data                       held-out test sets
#     $AML_IMAGES_PATH                raw source images (only with --include-images)
#
#   Local metadata (copied to <dest>/metadata):
#     outputs/model_registry.json, outputs/extraction_registry.json,
#     azureml/config.env, azureml/*.yml
#
#   The 660k raw source images and hf_cache are EXCLUDED by default: the images
#   are re-derivable from the NMLA archive (scripts/download_documents.py +
#   scripts/split_documents.py) and hf_cache is re-downloaded automatically by
#   jobs.  Pass --include-images to also back up $AML_IMAGES_PATH.
#
# ── Options ───────────────────────────────────────────────────────────────────
#   --dest DIR            Backup bundle directory (REQUIRED; or set AML_BACKUP_DIR)
#   --include-images      Also back up the raw source images ($AML_IMAGES_PATH)
#   --path PATH           Add an extra datastore path to back up (repeatable)
#   --jobs N              Parallel download connections (default: 16)
#   --quiet               Print only summary/error lines (faster on huge runs)
#   --dry-run             Print az storage commands without executing them
#   --help
#
# ── Examples ──────────────────────────────────────────────────────────────────
#   bash scripts/aml_backup.sh --dest /data/backups/aml
#   bash scripts/aml_backup.sh --dest /data/backups/aml --include-images
#   AML_BACKUP_DIR=/data/backups/aml bash scripts/aml_backup.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$REPO_DIR/azureml/config.env"

[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"

AML_SUBSCRIPTION="${AML_SUBSCRIPTION:-}"
AML_RESOURCE_GROUP="${AML_RESOURCE_GROUP:-}"
AML_WORKSPACE="${AML_WORKSPACE:-}"
AML_DATASTORE_BASE="${AML_DATASTORE_BASE:-azureml://datastores/workspaceblobstore/paths}"
AML_IMAGES_PATH="${AML_IMAGES_PATH:-Daily_rainfall_sample/images}"
AML_TRANSCRIPTIONS_PATH="${AML_TRANSCRIPTIONS_PATH:-Daily_rainfall_sample/transcriptions}"
AML_OUTPUTS_PATH="${AML_OUTPUTS_PATH:-outputs}"

DEST="${AML_BACKUP_DIR:-}"
INCLUDE_IMAGES=false
DOWNLOAD_JOBS="${DOWNLOAD_JOBS:-16}"
QUIET=false
DRY_RUN=false
EXTRA_PATHS=()

usage() {
    sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)            DEST="$2"; shift 2 ;;
        --include-images)  INCLUDE_IMAGES=true; shift ;;
        --path)            EXTRA_PATHS+=("$2"); shift 2 ;;
        --jobs)            DOWNLOAD_JOBS="$2"; shift 2 ;;
        --quiet)           QUIET=true; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --help|-h)         usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

# ── Validate the (mandatory) backup directory ─────────────────────────────────
if [[ -z "$DEST" ]]; then
    echo "Error: a backup directory is required." >&2
    echo "       Pass --dest DIR or set AML_BACKUP_DIR.  The bundle can be very" >&2
    echo "       large, so choose a large-capacity path outside the repository," >&2
    echo "       e.g. --dest /data/backups/aml" >&2
    exit 1
fi

[[ -z "$AML_SUBSCRIPTION" ]]   && { echo "Error: AML_SUBSCRIPTION not set" >&2; exit 1; }
[[ -z "$AML_RESOURCE_GROUP" ]] && { echo "Error: AML_RESOURCE_GROUP not set" >&2; exit 1; }
if ! $DRY_RUN; then
    [[ -z "$AML_WORKSPACE" ]]  && { echo "Error: AML_WORKSPACE not set" >&2; exit 1; }
fi

DATASTORE_DIR="$DEST/datastore"
METADATA_DIR="$DEST/metadata"
MANIFEST_FILE="$DEST/manifest.json"

# ── Assemble the list of datastore paths to back up ───────────────────────────
BACKUP_PATHS=(
    "$AML_TRANSCRIPTIONS_PATH"
    "$AML_OUTPUTS_PATH/checkpoints"
    "$AML_OUTPUTS_PATH/extractions"
    "$AML_OUTPUTS_PATH/eval"
    "consensus_data"
    "test_data"
)
$INCLUDE_IMAGES && BACKUP_PATHS+=("$AML_IMAGES_PATH")
if [[ ${#EXTRA_PATHS[@]} -gt 0 ]]; then
    BACKUP_PATHS+=("${EXTRA_PATHS[@]}")
fi

DATASTORE_NAME="$(echo "$AML_DATASTORE_BASE" | sed 's|.*/datastores/||;s|/paths.*||')"
STORAGE_ACCOUNT=""
CONTAINER=""

# azcopy reuses the current `az login` credentials and honours DOWNLOAD_JOBS as
# its transfer concurrency.
export AZCOPY_AUTO_LOGIN_TYPE="${AZCOPY_AUTO_LOGIN_TYPE:-AZCLI}"
export AZCOPY_CONCURRENCY_VALUE="${AZCOPY_CONCURRENCY_VALUE:-$DOWNLOAD_JOBS}"

echo "Azure ML workspace backup"
echo "  workspace:  $AML_WORKSPACE"
echo "  datastore:  $DATASTORE_NAME"
echo "  dest:       $DEST"
echo "  images:     $($INCLUDE_IMAGES && echo included || echo excluded)"
echo

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
# "hdi_isfolder" marker blob for every folder level (e.g. a blob named
# <run>/<model> alongside the real files under <run>/<model>/...).  `az storage
# blob download-batch` downloads such a marker as a file and then aborts with
# "[Errno 20] Not a directory" when it needs that same name as a directory.
# azcopy understands these markers and materialises them as directories, so the
# whole tree downloads cleanly.  --overwrite=ifSourceNewer makes re-runs
# idempotent and resumable.
do_backup_path() {
    local src_path="$1"   # path prefix in the container
    local dst="$DATASTORE_DIR/$src_path"
    local parent
    parent="$(dirname "$dst")"
    local url="https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER}/${src_path}"
    if $DRY_RUN; then
        echo "[dry-run] azcopy copy \\"
        echo "    'https://<account>.blob.core.windows.net/<container>/${src_path}' \\"
        echo "    '${DATASTORE_DIR}/$(dirname "$src_path")' --recursive --overwrite=ifSourceNewer"
        echo
        return 0
    fi
    echo "Backing up: ${src_path}"
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
        echo
        return 0
    fi
    echo "        to: $dst"
    mkdir -p "$parent"
    azcopy copy "$url" "$parent" \
        --recursive \
        --overwrite=ifSourceNewer \
        --output-level "$($QUIET && echo quiet || echo essential)"
    echo "Done."
    echo
}

# ── Back up the datastore paths ───────────────────────────────────────────────
for path in "${BACKUP_PATHS[@]}"; do
    do_backup_path "$path"
done

# ── Copy local metadata into the bundle ───────────────────────────────────────
METADATA_FILES=(
    "outputs/model_registry.json"
    "outputs/extraction_registry.json"
    "azureml/config.env"
)
# Environment / conda / job specs (azureml/*.yml).
for yml in "$REPO_DIR"/azureml/*.yml; do
    [[ -e "$yml" ]] || continue
    METADATA_FILES+=("azureml/$(basename "$yml")")
done

copy_metadata() {
    local rel="$1"
    local src="$REPO_DIR/$rel"
    local dst="$METADATA_DIR/$rel"
    if [[ ! -e "$src" ]]; then
        echo "  (skip missing $rel)"
        return 0
    fi
    if $DRY_RUN; then
        echo "[dry-run] cp $src → $dst"
    else
        mkdir -p "$(dirname "$dst")"
        cp -a "$src" "$dst"
        echo "  copied $rel"
    fi
}

echo "Copying local metadata..."
for rel in "${METADATA_FILES[@]}"; do
    copy_metadata "$rel"
done
echo

# ── Write the manifest ────────────────────────────────────────────────────────
write_manifest() {
    BACKUP_PATHS_JSON="$(printf '%s\n' "${BACKUP_PATHS[@]}")" \
    METADATA_JSON="$(printf '%s\n' "${METADATA_FILES[@]}")" \
    DATASTORE_NAME="$DATASTORE_NAME" \
    INCLUDE_IMAGES="$INCLUDE_IMAGES" \
    AML_WORKSPACE="$AML_WORKSPACE" \
    AML_DATASTORE_BASE="$AML_DATASTORE_BASE" \
    python3 - "$MANIFEST_FILE" <<'PY'
import json, os, sys
from datetime import datetime, timezone

manifest_file = sys.argv[1]
paths = [p for p in os.environ["BACKUP_PATHS_JSON"].splitlines() if p]
metadata = [m for m in os.environ["METADATA_JSON"].splitlines() if m]
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "workspace": os.environ["AML_WORKSPACE"],
    "datastore_name": os.environ["DATASTORE_NAME"],
    "datastore_base": os.environ["AML_DATASTORE_BASE"],
    "include_images": os.environ["INCLUDE_IMAGES"] == "true",
    "datastore_paths": [
        {"path": p, "local": f"datastore/{p}"} for p in paths
    ],
    "metadata_files": [
        {"path": m, "local": f"metadata/{m}"} for m in metadata
    ],
}
os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
with open(manifest_file, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote manifest: {manifest_file}")
PY
}

if $DRY_RUN; then
    echo "[dry-run] Would write manifest to $MANIFEST_FILE"
else
    mkdir -p "$DEST"
    write_manifest
fi

echo
echo "Backup complete."
$DRY_RUN && echo "(dry-run — no data was transferred)"
echo "Restore into a fresh bare workspace with:"
echo "  bash scripts/aml_restore.sh --from $DEST"
