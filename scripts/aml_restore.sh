#!/usr/bin/env bash
# aml_restore.sh — Repopulate a freshly created bare Azure ML workspace from a
# local backup bundle produced by scripts/aml_backup.sh.
#
# ENVIRONMENT: Run in the weather-doc-extractor conda environment.
#   conda activate weather-doc-extractor
#   bash scripts/aml_restore.sh --from /large/capacity/workspace_backup
#
# Use this AFTER your bare-workspace spin-up has created the workspace, its
# datastore (large_datastore), and the compute cluster.  This script then:
#   1. Re-registers the curated environments (v100 + a100).
#   2. Re-uploads every datastore path recorded in the backup manifest.
#
# It does NOT create the workspace, datastore, or compute cluster — those come
# from your existing bare-workspace spin-up.
#
# Sources azureml/config.env for workspace coordinates and default paths.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/aml_restore.sh --from DIR [options]
#
# The backup directory is REQUIRED and must be given explicitly (via --from or
# the AML_BACKUP_DIR environment variable).
#
# ── Options ───────────────────────────────────────────────────────────────────
#   --from DIR            Backup bundle directory (REQUIRED; or set AML_BACKUP_DIR)
#   --include-images      Re-upload raw source images even if not flagged in the
#                         manifest (by default images are uploaded only if they
#                         were captured in the backup)
#   --skip-environments   Do not re-register the Azure ML environments
#   --skip-data           Do not re-upload datastore data
#   --restore-metadata    Copy the backed-up registries/config back into the repo
#   --env-variant V       Which environments to register: v100, a100, both
#                         (default: both)
#   --dry-run             Print commands without executing them
#   --help
#
# ── Examples ──────────────────────────────────────────────────────────────────
#   bash scripts/aml_restore.sh --from /data/backups/aml
#   bash scripts/aml_restore.sh --from /data/backups/aml --dry-run
#   bash scripts/aml_restore.sh --from /data/backups/aml --skip-environments

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

FROM="${AML_BACKUP_DIR:-}"
INCLUDE_IMAGES=false
SKIP_ENVIRONMENTS=false
SKIP_DATA=false
RESTORE_METADATA=false
ENV_VARIANT="both"
DRY_RUN=false

usage() {
    sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)              FROM="$2"; shift 2 ;;
        --include-images)    INCLUDE_IMAGES=true; shift ;;
        --skip-environments) SKIP_ENVIRONMENTS=true; shift ;;
        --skip-data)         SKIP_DATA=true; shift ;;
        --restore-metadata)  RESTORE_METADATA=true; shift ;;
        --env-variant)
            ENV_VARIANT="$2"
            [[ "$ENV_VARIANT" =~ ^(v100|a100|both)$ ]] || { echo "Error: --env-variant must be v100, a100, or both" >&2; exit 1; }
            shift 2 ;;
        --dry-run)           DRY_RUN=true; shift ;;
        --help|-h)           usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

# ── Validate the (mandatory) backup directory ─────────────────────────────────
if [[ -z "$FROM" ]]; then
    echo "Error: a backup directory is required." >&2
    echo "       Pass --from DIR or set AML_BACKUP_DIR, pointing at a bundle" >&2
    echo "       created by scripts/aml_backup.sh." >&2
    exit 1
fi

MANIFEST_FILE="$FROM/manifest.json"
DATASTORE_DIR="$FROM/datastore"
METADATA_DIR="$FROM/metadata"

if [[ ! -f "$MANIFEST_FILE" ]]; then
    echo "Error: manifest not found at $MANIFEST_FILE" >&2
    echo "       Is '$FROM' a backup bundle created by scripts/aml_backup.sh?" >&2
    exit 1
fi

[[ -z "$AML_SUBSCRIPTION" ]]   && { echo "Error: AML_SUBSCRIPTION not set" >&2; exit 1; }
[[ -z "$AML_RESOURCE_GROUP" ]] && { echo "Error: AML_RESOURCE_GROUP not set" >&2; exit 1; }
if ! $DRY_RUN; then
    [[ -z "$AML_WORKSPACE" ]]  && { echo "Error: AML_WORKSPACE not set" >&2; exit 1; }
fi

# ── Read the manifest ─────────────────────────────────────────────────────────
MANIFEST_INCLUDE_IMAGES="$(python3 -c "import json,sys; print(json.load(open('$MANIFEST_FILE')).get('include_images', False))")"
mapfile -t MANIFEST_PATHS < <(python3 -c "
import json
m = json.load(open('$MANIFEST_FILE'))
for entry in m.get('datastore_paths', []):
    print(entry['path'])
")

DATASTORE_NAME="$(echo "$AML_DATASTORE_BASE" | sed 's|.*/datastores/||;s|/paths.*||')"

echo "Azure ML workspace restore"
echo "  workspace:  $AML_WORKSPACE"
echo "  datastore:  $DATASTORE_NAME"
echo "  from:       $FROM"
echo "  paths:      ${#MANIFEST_PATHS[@]} datastore path(s) in manifest"
echo

# ── Step 1: re-register environments ──────────────────────────────────────────
if $SKIP_ENVIRONMENTS; then
    echo "Skipping environment registration (--skip-environments)."
    echo
else
    echo "Re-registering Azure ML environments (variant: $ENV_VARIANT)..."
    register_cmd=(bash "$SCRIPT_DIR/azure_register_environments.sh" --variant "$ENV_VARIANT")
    $DRY_RUN && register_cmd+=(--dry-run)
    "${register_cmd[@]}"
    echo
fi

# ── Resolve storage account and container for the data upload ─────────────────
STORAGE_ACCOUNT=""
CONTAINER=""
if ! $SKIP_DATA; then
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
fi

# ── Upload helper ─────────────────────────────────────────────────────────────
# Mirrors scripts/aml_upload.sh: bulk parallel transfer via upload-batch.
do_restore_path() {
    local dst_path="$1"                 # path prefix in the container
    local src="$DATASTORE_DIR/$dst_path"  # local source in the bundle
    if [[ ! -d "$src" ]]; then
        echo "  (skip $dst_path — not present in bundle)"
        echo
        return 0
    fi
    if $DRY_RUN; then
        echo "[dry-run] az storage blob upload-batch \\"
        echo "    --account-name <account> --auth-mode login \\"
        echo "    --source $src \\"
        echo "    --destination <container> --destination-path $dst_path --overwrite true"
        echo
        return 0
    fi
    echo "Restoring: $src"
    echo "       to: ${dst_path}/"
    az storage blob upload-batch \
        --account-name "$STORAGE_ACCOUNT" \
        --auth-mode login \
        --source "$src" \
        --destination "$CONTAINER" \
        --destination-path "$dst_path" \
        --overwrite true
    echo "Done."
    echo
}

# ── Step 2: re-upload datastore data ──────────────────────────────────────────
if $SKIP_DATA; then
    echo "Skipping data upload (--skip-data)."
    echo
else
    echo "Re-uploading datastore data..."
    echo
    for path in "${MANIFEST_PATHS[@]}"; do
        # Images are uploaded only if captured in the backup or forced.
        if [[ "$path" == "$AML_IMAGES_PATH" ]]; then
            if [[ "$MANIFEST_INCLUDE_IMAGES" != "True" ]] && ! $INCLUDE_IMAGES; then
                echo "  (skip images $path — not in backup; pass --include-images to force)"
                echo
                continue
            fi
        fi
        do_restore_path "$path"
    done
fi

# ── Optional: restore local metadata into the repo ────────────────────────────
if $RESTORE_METADATA; then
    echo "Restoring local metadata into the repository..."
    restore_metadata_file() {
        local rel="$1"
        local src="$METADATA_DIR/$rel"
        local dst="$REPO_DIR/$rel"
        if [[ ! -e "$src" ]]; then
            echo "  (skip missing $rel)"
            return 0
        fi
        if $DRY_RUN; then
            echo "[dry-run] cp $src → $dst"
        else
            mkdir -p "$(dirname "$dst")"
            cp -a "$src" "$dst"
            echo "  restored $rel"
        fi
    }
    mapfile -t METADATA_RELS < <(python3 -c "
import json
m = json.load(open('$MANIFEST_FILE'))
for entry in m.get('metadata_files', []):
    print(entry['path'])
")
    for rel in "${METADATA_RELS[@]}"; do
        restore_metadata_file "$rel"
    done
    echo
fi

echo "Restore complete."
$DRY_RUN && echo "(dry-run — no changes were made)"
