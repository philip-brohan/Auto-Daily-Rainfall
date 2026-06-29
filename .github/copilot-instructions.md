---
description: 'Instructions for building AI pipelines to extract weather data from historical documents, including fine-tuning and checkpoint management.'
applyTo: '**'
---

# Project Scope

This repository extracts daily rainfall tables from historical document images.

- Scale: about 660,000 JPEG images.
- Unit of data: one station-year table per image.
- Target output: station metadata plus daily and monthly precipitation values.

The workflow is multi-model and staged:

1. Initial fine-tuning on synthetic data with known truth.
2. Multi-model extraction on real samples.
3. Consensus dataset creation from model agreement.
4. Second-round fine-tuning on high-confidence consensus samples.
5. Full-dataset extraction with the updated models.

# Compute and Persistence

- Local machine: low-resource (6 GB RAM, no GPU). Use it for code changes, orchestration, and validation only.
- Azure ML: primary execution target for extraction, evaluation, and fine-tuning.
- Any artifact that must persist between jobs must live on Azure datastore paths.
- Do not rely on local ephemeral caches from Azure runs.

# Environments

- Local environment: weather-doc-extractor from environment.yml.
- Azure environments: files in azureml/ (for example conda.yml and conda-a100.yml).
- Environment registration script: scripts/azure_register_environments.sh.

When dependencies or model support change:

1. Update the relevant Azure ML environment YAML file(s).
2. Re-register environments if needed.
3. Keep local environment.yml aligned where applicable.

## MANDATORY: Script Execution Environment

**ALL scripts (Python, shell, notebooks) must be executed in the weather-doc-extractor conda environment.**

- Activate: `conda activate weather-doc-extractor`
- Do not run scripts with system Python or other environments
- Do not suggest or create workarounds that bypass the environment
- If creating new scripts, always include environment setup instructions in docstrings/comments
- When documenting CLI commands, always prefix with environment activation

This is non-negotiable to maintain reproducibility and dependency isolation.

# Required Workflow Scripts

Prefer repository scripts over ad hoc commands:

- scripts/aml_submit.sh: submit Azure ML jobs.
- scripts/aml_upload.sh: upload datasets and assets.
- scripts/run_extract.sh: Azure extraction entrypoint.
- scripts/create_model_registry_entry.py: register model checkpoints.
- scripts/list_checkpoints.sh: inspect checkpoint registry.

For consensus-stage work:

- scripts/build_consensus_transcriptions.py: build consensus JSONs.
- scripts/prepare_consensus_dataset.py: create dataset layout for consensus runs.
- scripts/plot_consensus_validation.py: single-image consensus visual check.
- scripts/validate_consensus.py: batch consensus validation figures.

# Data and Checkpoint Rules

Treat datasets and checkpoints as Azure-hosted first-class artifacts.

For datasets:

- Keep images and transcriptions paired.
- Preserve existing stem naming.
- Maintain compatibility with ingest and evaluation pipelines.

For checkpoints:

- Store on Azure datastore.
- Use registry scripts and documented checkpoint workflow.
- Keep checkpoints selectable through submission scripts.

# Model Change Rules

When changing model family handling:

1. Preserve backward compatibility for existing presets.
2. Update presets, inference routing, fine-tuning logic, and CLI help together.
3. Update or add tests for model detection, message formatting, and training example construction.
4. Update documentation in the same change set.

Model families may need different processor logic, prompt/message formatting, and environment variants.

# Editing Expectations

- Make small, focused changes that address the root cause.
- Avoid unrelated refactors.
- Do not change model behavior without test updates.
- Do not add dependencies unless necessary and documented.
- Do not move expected Azure-hosted artifacts back to local-only paths.

If extraction or fine-tuning code changes, validate both paths when practical.

# Documentation Requirements

If you change CLI behavior, model presets, checkpoint handling, or Azure submission behavior:

1. Update README/docs.
2. Keep command examples accurate.
3. Keep docs aligned with current Azure workflow.

# New Script and Feature Implementation

When creating new scripts or implementing new features:

1. **Always include environment activation** in script docstrings, comments, or shell script headers
2. **Document environment requirements** explicitly (e.g., "ENVIRONMENT: Run in weather-doc-extractor conda environment")
3. **Add examples** showing correct environment activation in docstrings and help text
4. **Write tests** that run in the proper environment
5. **Update this instructions file** if the new feature needs special environment handling
6. **Do not create scripts that run outside the weather-doc-extractor environment**

Example for new Python scripts:
```python
#!/usr/bin/env python3
"""Script description.

ENVIRONMENT: Run this script in the weather-doc-extractor conda environment:
  conda activate weather-doc-extractor
  python scripts/my_script.py [options]
"""
```

Example for new shell scripts:
```bash
#!/bin/bash
# ENVIRONMENT: Run in weather-doc-extractor conda environment
#   conda activate weather-doc-extractor

# Rest of script...
```

# General Guidance

Follow existing project style and structure.
Prefer repository-native Azure ML solutions over one-off workarounds.

