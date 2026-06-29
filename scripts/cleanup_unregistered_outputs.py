#!/usr/bin/env python3
"""Cleanup checkpoints/extractions that are not referenced in registries.

Default behavior is dry-run. Use --apply to perform deletions.

This script can clean:
- Local artifacts under outputs/checkpoints and outputs/extractions
- Azure datastore artifacts under <AML_OUTPUTS_PATH>/checkpoints and /extractions

Typical flow:
1) Edit outputs/model_registry.json and outputs/extraction_registry.json
2) Run this script in dry-run mode and inspect output
3) Re-run with --apply to delete unregistered artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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


def _normalize_kind_suffix(raw_path: str, kind: str) -> str | None:
    """Extract suffix after .../<kind>/ from a registry path.

    Examples:
      Daily_rainfall_sample/outputs/checkpoints/run/model -> run/model
      outputs/extractions/model/run -> model/run
    """
    if not raw_path:
        return None
    p = raw_path.replace("\\", "/").strip().strip("/")
    token = f"/{kind}/"
    if token in f"/{p}":
        return f"/{p}".split(token, 1)[1].strip("/")
    if p.startswith(f"{kind}/"):
        return p[len(kind) + 1 :].strip("/")
    if f"outputs/{kind}/" in p:
        return p.split(f"outputs/{kind}/", 1)[1].strip("/")
    return None


def _load_keep_sets(
    model_registry: Path, extraction_registry: Path
) -> tuple[set[str], set[str]]:
    model_data = _read_json(model_registry)
    extraction_data = _read_json(extraction_registry)

    keep_checkpoints: set[str] = set()
    for entry in model_data.get("models", []):
        suffix = _normalize_kind_suffix(
            str(entry.get("checkpoint_path", "")), "checkpoints"
        )
        if suffix:
            keep_checkpoints.add(suffix)

    keep_extractions: set[str] = set()
    for entry in extraction_data.get("extractions", []):
        suffix = _normalize_kind_suffix(
            str(entry.get("extractions_path", "")), "extractions"
        )
        if suffix:
            keep_extractions.add(suffix)

    return keep_checkpoints, keep_extractions


def _discover_local_suffixes(local_outputs_dir: Path, kind: str) -> set[str]:
    base = local_outputs_dir / kind
    if not base.exists():
        return set()
    suffixes: set[str] = set()
    for first in base.iterdir():
        if not first.is_dir():
            continue
        second_level_dirs = [p for p in first.iterdir() if p.is_dir()]
        if not second_level_dirs:
            # One-level layout, e.g. outputs/extractions/<run_name>
            suffixes.add(first.name)
            continue

        # Two-level layout, e.g. outputs/extractions/<model>/<run>
        for second in first.iterdir():
            if second.is_dir():
                suffixes.add(f"{first.name}/{second.name}")
    return suffixes


def _delete_local_paths(
    local_outputs_dir: Path,
    kind: str,
    suffixes_to_delete: Iterable[str],
    apply: bool,
) -> tuple[int, int]:
    deleted = 0
    missing = 0
    for suffix in sorted(set(suffixes_to_delete)):
        target = local_outputs_dir / kind / suffix
        if not target.exists():
            missing += 1
            continue
        if apply:
            shutil.rmtree(target)
        deleted += 1
        action = "DELETE" if apply else "DRYRUN"
        print(f"[{action}] local {kind}: {target}")
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
    storage_account = str(data["account_name"])
    container = str(data["container_name"])

    return AzureContext(
        subscription=subscription,
        resource_group=resource_group,
        workspace=workspace,
        datastore_name=datastore_name,
        storage_account=storage_account,
        container=container,
        aml_outputs_path=aml_outputs_path,
    )


def _list_azure_suffixes(ctx: AzureContext, kind: str) -> set[str]:
    prefix = f"{ctx.aml_outputs_path}/{kind}/".strip("/")
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
    for blob in blobs:
        name = str(blob.get("name", "")).strip("/")
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :].strip("/")
        if not rest:
            continue
        parts = [p for p in rest.split("/") if p]
        # We only treat depth-2 directories as managed artifacts:
        #   checkpoints/<a>/<b>/... and extractions/<a>/<b>/...
        # Requiring at least 3 segments avoids misclassifying file paths like
        # checkpoints/<model>/README.md as artifact prefixes.
        if len(parts) >= 3:
            suffixes.add(f"{parts[0]}/{parts[1]}")
    return suffixes


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
            # Ignore missing marker errors; re-raise everything else.
            err = (proc.stderr or "").lower()
            if "blobnotfound" in err or "not found" in err:
                continue
            raise RuntimeError(f"Failed to delete blob '{name}': {proc.stderr.strip()}")

    return deleted


def _delete_azure_paths(
    ctx: AzureContext,
    kind: str,
    suffixes_to_delete: Iterable[str],
    apply: bool,
) -> int:
    total_deleted_blobs = 0
    for suffix in sorted(set(suffixes_to_delete)):
        prefix = f"{ctx.aml_outputs_path}/{kind}/{suffix}".strip("/")
        action = "DELETE" if apply else "DRYRUN"
        print(f"[{action}] azure {kind} prefix: {prefix}")
        total_deleted_blobs += _delete_azure_prefix(ctx, prefix, apply)
    return total_deleted_blobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete local/Azure checkpoints and extractions not present in registries."
    )
    parser.add_argument(
        "--model-registry",
        default="outputs/model_registry.json",
        help="Path to model registry JSON (default: outputs/model_registry.json)",
    )
    parser.add_argument(
        "--extraction-registry",
        default="outputs/extraction_registry.json",
        help="Path to extraction registry JSON (default: outputs/extraction_registry.json)",
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
    extraction_registry = (repo_dir / args.extraction_registry).resolve()
    local_outputs_dir = (repo_dir / args.local_outputs_dir).resolve()
    config_path = (repo_dir / args.config).resolve()

    keep_checkpoints, keep_extractions = _load_keep_sets(
        model_registry, extraction_registry
    )

    print("Registry keep counts:")
    print(f"  checkpoints: {len(keep_checkpoints)}")
    print(f"  extractions: {len(keep_extractions)}")
    print()

    do_local = not args.azure_only
    do_azure = not args.local_only

    local_checkpoint_delete: set[str] = set()
    local_extraction_delete: set[str] = set()

    if do_local:
        actual_local_checkpoints = _discover_local_suffixes(
            local_outputs_dir, "checkpoints"
        )
        actual_local_extractions = _discover_local_suffixes(
            local_outputs_dir, "extractions"
        )

        local_checkpoint_delete = actual_local_checkpoints - keep_checkpoints
        local_extraction_delete = actual_local_extractions - keep_extractions

        print("Local cleanup plan:")
        print(f"  local checkpoints found: {len(actual_local_checkpoints)}")
        print(f"  local checkpoints delete: {len(local_checkpoint_delete)}")
        print(f"  local extractions found: {len(actual_local_extractions)}")
        print(f"  local extractions delete: {len(local_extraction_delete)}")
        print()

    azure_checkpoint_delete: set[str] = set()
    azure_extraction_delete: set[str] = set()
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

        actual_azure_checkpoints = _list_azure_suffixes(azure_ctx, "checkpoints")
        actual_azure_extractions = _list_azure_suffixes(azure_ctx, "extractions")

        azure_checkpoint_delete = actual_azure_checkpoints - keep_checkpoints
        azure_extraction_delete = actual_azure_extractions - keep_extractions

        print("Azure cleanup plan:")
        print(f"  azure checkpoints found: {len(actual_azure_checkpoints)}")
        print(f"  azure checkpoints delete: {len(azure_checkpoint_delete)}")
        print(f"  azure extractions found: {len(actual_azure_extractions)}")
        print(f"  azure extractions delete: {len(azure_extraction_delete)}")
        print()

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Execution mode: {mode}")
    print()

    if do_local:
        _delete_local_paths(
            local_outputs_dir, "checkpoints", local_checkpoint_delete, args.apply
        )
        _delete_local_paths(
            local_outputs_dir, "extractions", local_extraction_delete, args.apply
        )

    if do_azure and azure_ctx is not None:
        _delete_azure_paths(
            azure_ctx, "checkpoints", azure_checkpoint_delete, args.apply
        )
        _delete_azure_paths(
            azure_ctx, "extractions", azure_extraction_delete, args.apply
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
