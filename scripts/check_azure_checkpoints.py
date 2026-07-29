#!/usr/bin/env python3
"""Check if all checkpoints in model_registry.json exist on Azure."""

import json
import subprocess
import sys
from pathlib import Path

# Load model registry
registry_path = Path("outputs/model_registry.json")
with open(registry_path) as f:
    registry = json.load(f)

# Load Azure config
config_env = Path("azureml/config.env")
config = {}
if config_env.exists():
    with open(config_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                config[key] = val.strip("\"'")

account_name = config.get("AML_STORAGE_ACCOUNT", "sallmdatarescue02")
container = config.get("AML_CONTAINER", "default")

print(
    f"Checking {len(registry['models'])} checkpoints on Azure ({account_name}/{container})...\n"
)

results = {"found": [], "missing": []}

for model in registry["models"]:
    path = model["checkpoint_path"].rstrip("/")

    # Check if path exists by listing with prefix and checking if any blobs exist
    try:
        output = subprocess.check_output(
            [
                "az",
                "storage",
                "blob",
                "list",
                "--account-name",
                account_name,
                "--container-name",
                container,
                "--prefix",
                path + "/",
                "--query",
                "length([].name)",
                "--output",
                "tsv",
                "--auth-mode",
                "login",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        count = int(output) if output else 0
        if count > 0:
            results["found"].append((model["checkpoint_name"], path, count))
            print(f"✓ {model['checkpoint_name']:<50} ({count} blobs)")
        else:
            results["missing"].append((model["checkpoint_name"], path))
            print(f"✗ {model['checkpoint_name']:<50} (MISSING)")
    except Exception as e:
        results["missing"].append((model["checkpoint_name"], path))
        print(f"✗ {model['checkpoint_name']:<50} (ERROR: {e})")

print(f"\n{'='*80}")
print(f"Summary: {len(results['found'])} found, {len(results['missing'])} missing")
if results["missing"]:
    print(f"\nMissing checkpoints:")
    for name, path in results["missing"]:
        print(f"  - {path}")
    sys.exit(1)
