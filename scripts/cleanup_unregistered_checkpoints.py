#!/usr/bin/env python3
"""Cleanup checkpoints that are not referenced in the model registry.

Default behavior is dry-run. Use --apply to perform deletions.

This script can clean:
- Local artifacts under outputs/checkpoints
- Azure datastore artifacts under <AML_OUTPUTS_PATH>/checkpoints

Typical flow:
1) Edit outputs/model_registry.json
2) Run this script in dry-run mode and inspect output
3) Re-run with --apply to delete unregistered checkpoints
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class AzureContext:
    subscription: str
    resource_group: str
    workspace: str
    datastore_name: str
    storage_account: str
    container: str
    aml_outputs_path: str


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def _load_config_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env[key] = value
    return env


def _normalize_checkpoint_suffix(raw_path: str) -> str | None:
    """Extract suffix after .../checkpoints/ from a registry path.

    Example:
      Daily_rainfall_sample/outputs/checkpoints/run/model -> run/model
    """
    if not raw_path:
        return None
    p = raw_path.replace("\\", "/").strip().strip("/")
    token = "/checkpoints/"
    if token in f"/{p}":
        return f"/{p}".split(token, 1)[1].strip("/")
    if p.startswith("checkpoints/"):
        return p[len("checkpoints") + 1 :].strip("/")
    if "outputs/checkpoints/" in p:
        return p.split("outputs/checkpoints/", 1)[1].strip("/")
    return None


def _load_keep_checkpoints(model_registry: Path) -> set[str]:
    model_data = _read_json(model_registry)
    keep_checkpoints: set[str] = set()

    for entry in model_data.get("models", []):
        suffix = _normalize_checkpoint_suffix(str(entry.get("checkpoint_path", "")))
        if suffix:
            keep_checkpoints.add(suffix)

    return keep_checkpoints


def _discover_local_suffixes(local_outputs_dir: Path) -> set[str]:
    base = local_outputs_dir / "checkpoints"
    if not base.exists():
        return set()

    suffixes: set[str] = set()
    for first in base.iterdir():
        if not first.is_dir():
            continue
        for second in first.iterdir():
            if second.is_dir():
                suffixes.add(f"{first.name}/{second.name}")
    return suffixes


def _delete_local_paths(
    local_outputs_dir: Path,
    suffixes_to_delete: Iterable[str],
    apply: bool,
) -> tuple[int, int]:
    deleted = 0
    missing = 0

    for suffix in sorted(set(suffixes_to_delete)):
        target = local_outputs_dir / "checkpoints" / suffix
        if not target.exists():
            missing += 1
            continue

        if apply:
            shutil.rmtree(target)

        deleted += 1
        action = "DELETE" if apply else "DRYRUN"
        print(f"[{action}] local checkpoints: {target}")

    return deleted, missing


def _run_az(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Azure CLI command failed:\n" + " ".join(args) + "\n" + proc.stderr.strip()
        )
    return proc.stdout


def _parse_datastore_name(aml_datastore_base: str) -> str:
    marker = "/datastores/"
    if marker not in aml_datastore_base or "/paths" not in aml_datastore_base:
        raise ValueError(
            "AML_DATASTORE_BASE must look like azureml://.../datastores/<name>/paths"
        )
    tail = aml_datastore_base.split(marker, 1)[1]
    return tail.split("/paths", 1)[0]


def _build_azure_context(config_env: dict[str, str]) -> AzureContext:
    subscription = config_env.get("AML_SUBSCRIPTION", "").strip()
    resource_group = config_env.get("AML_RESOURCE_GROUP", "").strip()
    workspace = config_env.get("AML_WORKSPACE", "").strip()
    aml_datastore_base = config_env.get(
        "AML_DATASTORE_BASE", "azureml://datastores/workspaceblobstore/paths"
    ).strip()
    aml_outputs_path = config_env.get("AML_OUTPUTS_PATH", "outputs").strip().strip("/")

    if not subscription or not resource_group or not workspace:
        raise ValueError(
            "Missing AML_SUBSCRIPTION/AML_RESOURCE_GROUP/AML_WORKSPACE in config/env"
        )

    datastore_name = _parse_datastore_name(aml_datastore_base)

    datastore_json = _run_az(
        [
            "az",
            "ml",
            "datastore",
            "show",
            "--name",
            datastore_name,
            "--workspace-name",
            workspace,
            "--resource-group",
            resource_group,
            "--subscription",
            subscription,
            "--output",
            "json",
        ]
    )
    data = json.loads(datastore_json)

    return AzureContext(
        subscription=subscription,
        resource_group=resource_group,
        workspace=workspace,
        datastore_name=datastore_name,
        storage_account=str(data["account_name"]),
        container=str(data["container_name"]),
        aml_outputs_path=aml_outputs_path,
    )


def _list_azure_state(ctx: AzureContext) -> tuple[set[str], set[str]]:
    prefix = f"{ctx.aml_outputs_path}/checkpoints/".strip("/")
    out = _run_az(
        [
            "az",
            "storage",
            "blob",
            "list",
            "--account-name",
            ctx.storage_account,
            "--auth-mode",
            "login",
            "--container-name",
            ctx.container,
            "--prefix",
            prefix,
            "--num-results",
            "*",
            "--output",
            "json",
        ]
    )
    blobs = json.loads(out)

    suffixes: set[str] = set()
    all_blob_names: list[str] = []
    for blob in blobs:
        name = str(blob.get("name", "")).strip("/")
        if not name.startswith(prefix):
            continue
        all_blob_names.append(name)

        rest = name[len(prefix) :].strip("/")
        if not rest:
            continue

        parts = [p for p in rest.split("/") if p]
        # Managed checkpoint artifacts are depth-2 directories:
        # checkpoints/<run>/<model>/...
        if len(parts) >= 3:
            suffixes.add(f"{parts[0]}/{parts[1]}")

    stray_blobs: set[str] = set()
    for name in all_blob_names:
        rest = name[len(prefix) :].strip("/")
        if not rest:
            continue
        parts = [p for p in rest.split("/") if p]
        if len(parts) >= 3:
            continue

        # Keep directory marker blobs that correspond to discovered checkpoint
        # prefixes so we do not report them as stray.
        if len(parts) == 2:
            candidate_suffix = f"{parts[0]}/{parts[1]}"
            if candidate_suffix in suffixes:
                continue
        if len(parts) == 1:
            run_prefix = f"{parts[0]}/"
            if any(s.startswith(run_prefix) for s in suffixes):
                continue

        stray_blobs.add(name)

    return suffixes, stray_blobs


def _delete_azure_blobs(
    ctx: AzureContext, blob_names: Iterable[str], apply: bool
) -> int:
    deleted = 0
    for name in sorted(set(blob_names)):
        action = "DELETE" if apply else "DRYRUN"
        print(f"[{action}] azure stray blob: {name}")
        if not apply:
            deleted += 1
            continue

        proc = subprocess.run(
            [
                "az",
                "storage",
                "blob",
                "delete",
                "--account-name",
                ctx.storage_account,
                "--auth-mode",
                "login",
                "--container-name",
                ctx.container,
                "--name",
                name,
                "--output",
                "none",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            deleted += 1
        else:
            err = (proc.stderr or "").lower()
            if "blobnotfound" in err or "not found" in err:
                continue
            raise RuntimeError(f"Failed to delete blob '{name}': {proc.stderr.strip()}")

    return deleted


def _delete_azure_prefix(ctx: AzureContext, full_prefix: str, apply: bool) -> int:
    full_prefix = full_prefix.strip("/")
    marker = full_prefix
    list_prefix = f"{full_prefix}/"

    out = _run_az(
        [
            "az",
            "storage",
            "blob",
            "list",
            "--account-name",
            ctx.storage_account,
            "--auth-mode",
            "login",
            "--container-name",
            ctx.container,
            "--prefix",
            list_prefix,
            "--num-results",
            "*",
            "--output",
            "json",
        ]
    )
    blobs = json.loads(out)
    names = [str(b.get("name", "")) for b in blobs if b.get("name")]

    # Try deleting directory marker blob too (safe if it doesn't exist).
    names.append(marker)

    deleted = 0
    for name in sorted(set(names)):
        action = "DELETE" if apply else "DRYRUN"
        print(f"[{action}] azure blob: {name}")
        if not apply:
            deleted += 1
            continue

        proc = subprocess.run(
            [
                "az",
                "storage",
                "blob",
                "delete",
                "--account-name",
                ctx.storage_account,
                "--auth-mode",
                "login",
                "--container-name",
                ctx.container,
                "--name",
                name,
                "--output",
                "none",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            deleted += 1
        else:
            err = (proc.stderr or "").lower()
            if "blobnotfound" in err or "not found" in err:
                continue
            raise RuntimeError(f"Failed to delete blob '{name}': {proc.stderr.strip()}")

    return deleted


def _delete_azure_paths(
    ctx: AzureContext,
    suffixes_to_delete: Iterable[str],
    apply: bool,
) -> int:
    total_deleted_blobs = 0
    for suffix in sorted(set(suffixes_to_delete)):
        prefix = f"{ctx.aml_outputs_path}/checkpoints/{suffix}".strip("/")
        action = "DELETE" if apply else "DRYRUN"
        print(f"[{action}] azure checkpoints prefix: {prefix}")
        total_deleted_blobs += _delete_azure_prefix(ctx, prefix, apply)
    return total_deleted_blobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete local/Azure checkpoints not present in model registry."
    )
    parser.add_argument(
        "--model-registry",
        default="outputs/model_registry.json",
        help="Path to model registry JSON (default: outputs/model_registry.json)",
    )
    parser.add_argument(
        "--local-outputs-dir",
        default="outputs",
        help="Local outputs root directory (default: outputs)",
    )
    parser.add_argument(
        "--config",
        default="azureml/config.env",
        help="Azure config env file (default: azureml/config.env)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only cleanup local files",
    )
    parser.add_argument(
        "--azure-only",
        action="store_true",
        help="Only cleanup Azure datastore files",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform deletion (default is dry-run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.local_only and args.azure_only:
        raise SystemExit("Cannot use --local-only and --azure-only together")

    repo_dir = Path(__file__).resolve().parents[1]
    model_registry = (repo_dir / args.model_registry).resolve()
    local_outputs_dir = (repo_dir / args.local_outputs_dir).resolve()
    config_path = (repo_dir / args.config).resolve()

    keep_checkpoints = _load_keep_checkpoints(model_registry)

    print("Registry keep counts:")
    print(f"  checkpoints: {len(keep_checkpoints)}")
    print()

    do_local = not args.azure_only
    do_azure = not args.local_only

    local_checkpoint_delete: set[str] = set()
    if do_local:
        actual_local_checkpoints = _discover_local_suffixes(local_outputs_dir)
        local_checkpoint_delete = actual_local_checkpoints - keep_checkpoints

        print("Local cleanup plan:")
        print(f"  local checkpoints found: {len(actual_local_checkpoints)}")
        print(f"  local checkpoints delete: {len(local_checkpoint_delete)}")
        print()

    azure_checkpoint_delete: set[str] = set()
    azure_stray_blob_delete: set[str] = set()
    azure_ctx: AzureContext | None = None
    if do_azure:
        cfg = _load_config_env(config_path)
        cfg = {**cfg, **os.environ}
        azure_ctx = _build_azure_context(cfg)

        print("Azure datastore context:")
        print(f"  datastore: {azure_ctx.datastore_name}")
        print(f"  account:   {azure_ctx.storage_account}")
        print(f"  container: {azure_ctx.container}")
        print(f"  outputs:   {azure_ctx.aml_outputs_path}")
        print()

        actual_azure_checkpoints, azure_stray_blobs = _list_azure_state(azure_ctx)
        azure_checkpoint_delete = actual_azure_checkpoints - keep_checkpoints
        azure_stray_blob_delete = azure_stray_blobs

        print("Azure cleanup plan:")
        print(f"  azure checkpoints found: {len(actual_azure_checkpoints)}")
        print(f"  azure checkpoints delete: {len(azure_checkpoint_delete)}")
        print(f"  azure stray blobs delete: {len(azure_stray_blob_delete)}")
        print()

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Execution mode: {mode}")
    print()

    if do_local:
        _delete_local_paths(local_outputs_dir, local_checkpoint_delete, args.apply)

    if do_azure and azure_ctx is not None:
        _delete_azure_paths(azure_ctx, azure_checkpoint_delete, args.apply)
        _delete_azure_blobs(azure_ctx, azure_stray_blob_delete, args.apply)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
