# Multi-Consensus Dataset Support Guide

**IMPORTANT**: All work in this project must be done in the `weather-doc-extractor` conda environment. See [environment setup](#environment-setup) below.

This guide explains the multi-consensus workflow that supports building multiple consensus variants from the same sampled dataset, using different model sets and parameters.

## Environment Setup

```bash
# Activate the project environment (one-time, then stays active for your session)
conda activate weather-doc-extractor

# Verify activation
python -m src.main info
```

All commands in this guide assume you're in the `weather-doc-extractor` environment.

## Overview

The consensus process involves:

1. **Sampling**: Randomly select N images from the full archive
2. **Preparation**: Create a flat dataset structure with images and placeholder transcriptions
3. **Extraction**: Run multiple models to extract tables from the images
4. **Consensus Building**: Vote across model extractions to produce high-confidence ground truth
5. **Validation**: Visually inspect consensus results
6. **Fine-tuning** (optional): Use consensus as training signal for improved models
7. **2nd/3rd Consensus** (optional): Repeat with fine-tuned models

## Directory Structure

Multiple consensus variants coexist in a single dataset:

```
outputs/consensus_dataset_1000/
├── images/                          # Shared across all variants
├── transcriptions/                  # Shared placeholder transcriptions
│
├── consensus_1000/                  # First consensus variant (5 base models)
│   ├── consensus_config.json
│   ├── consensus_transcriptions/    # Consensus votes
│   ├── consensus_summary.json       # Stats and metadata
│   └── validation_figures/          # PNG comparisons
│
├── consensus_1000_ft/               # Second variant (5 fine-tuned models)
│   ├── consensus_config.json
│   ├── consensus_transcriptions/
│   ├── consensus_summary.json
│   └── validation_figures/
│
└── consensus_1000_granite/          # Third variant (Granite only)
    ├── consensus_config.json
    ├── consensus_transcriptions/
    ├── consensus_summary.json
    └── validation_figures/
```

## Quick Start: Building Your First Consensus Variant

### 1. Sample and Prepare Dataset

```bash
# Sample 1000 images (one-time setup)
python scripts/sample_unseen_images.py \
  --count 1000 \
  --seed 42 \
  --output-dir outputs/consensus_dataset_1000

# Prepare the dataset structure (creates images/ and transcriptions/)
python scripts/prepare_consensus_dataset.py \
  --manifest-csv outputs/consensus_dataset_1000/sample_manifest.csv \
  --dataset-root outputs/consensus_dataset_1000
```

### 2. Extract with Multiple Models

Run extraction jobs with 5 models (typically using Azure ML):

```bash
# Submit extraction jobs
bash scripts/run_consensus_extractions.sh \
  --dataset outputs/consensus_dataset_1000 \
  --output-root outputs/extractions

# OR submit via Azure:
bash scripts/aml_submit.sh --dataset consensus extract
```

After extractions complete, you should have:
```
outputs/extractions/
├── model_A/20260601-120000/
├── model_B/20260601-120500/
├── model_C/20260601-121000/
├── model_D/20260601-121500/
└── model_E/20260602-100000/
```

### 3. Build Consensus Using the Pipeline

The easiest way to build consensus is using the orchestration script:

```bash
bash scripts/run_consensus_pipeline.sh \
  --dataset-root outputs/consensus_dataset_1000 \
  --variant-name consensus_1000 \
  --threshold 4 \
  --precision 3 \
  --validate \
  --extraction-dir outputs/extractions/model_A/20260601-120000 \
  --extraction-dir outputs/extractions/model_B/20260601-120500 \
  --extraction-dir outputs/extractions/model_C/20260601-121000 \
  --extraction-dir outputs/extractions/model_D/20260601-121500 \
  --extraction-dir outputs/extractions/model_E/20260602-100000
  outputs/extractions/model_A/20260601-120000 \
  outputs/extractions/model_B/20260601-120500 \
  outputs/extractions/model_C/20260601-121000 \
  outputs/extractions/model_D/20260601-121500 \
  outputs/extractions/model_E/20260602-100000
```

This script automatically:
- Creates the config file
- Builds consensus transcriptions
- Generates validation figures (optional)

## Advanced Usage

### Manual Workflow (Low-Level)

If you prefer fine-grained control, use the individual scripts:

#### 1. Create Config File

```bash
python scripts/create_consensus_config.py \
  --variant-name consensus_1000 \
  --dataset-root outputs/consensus_dataset_1000 \
  --agreement-threshold 4 \
  --precision 3 \
  --description "First consensus with 5 base models" \
  outputs/extractions/model_A/20260601-120000 \
  outputs/extractions/model_B/20260601-120500 \
  outputs/extractions/model_C/20260601-121000 \
  outputs/extractions/model_D/20260601-121500 \
  outputs/extractions/model_E/20260602-100000
```

This creates: `outputs/consensus_dataset_1000/consensus_1000/consensus_config.json`

#### 2. Build Consensus from Config

```bash
python scripts/build_consensus_transcriptions.py \
  --config-file outputs/consensus_dataset_1000/consensus_1000/consensus_config.json
```

Output:
- `{variant_dir}/consensus_transcriptions/` — consensus votes for each stem
- `{variant_dir}/consensus_summary.json` — statistics and metadata

#### 3. Validate Consensus Results

```bash
python scripts/validate_consensus.py \
  --config-file outputs/consensus_dataset_1000/consensus_1000/consensus_config.json
```

Output: PNG figures in `{variant_dir}/validation_figures/`

### Config File Format

Each consensus variant has a `consensus_config.json`:

```json
{
  "variant_name": "consensus_1000",
  "description": "First consensus round with 5 base models",
  "extraction_dirs": [
    "outputs/extractions/model_A/20260601-120000",
    "outputs/extractions/model_B/20260601-120500",
    ...
  ],
  "agreement_threshold": 4,
  "precision": 3,
  "timestamp": "2026-06-01T12:00:00",
  "notes": "Trained on full archive ~660k images"
}
```

## Multi-Round Consensus (2nd Order)

The true power of variants appears when iterating:

### Workflow

1. **1st Consensus**: Build from 5 base models → extract high-confidence ground truth
2. **Fine-tune**: Train improved models on `correct=true` cells from 1st consensus
3. **2nd Extraction**: Run fine-tuned models on the same 1000 images
4. **2nd Consensus**: Vote across 5 fine-tuned models → better quality results
5. **3rd Round** (optional): Repeat

### Example: 2nd Consensus with Fine-Tuned Models

```bash
# After fine-tuning 5 models and extracting with them:
bash scripts/run_consensus_pipeline.sh \
  --dataset-root outputs/consensus_dataset_1000 \
  --variant-name consensus_1000_ft \
  --threshold 4 \
  --precision 3 \
  --validate \
  --extraction-dir outputs/extractions/ft_granite/20260610-150000 \
  --extraction-dir outputs/extractions/ft_smolvlm/20260610-151000 \
  --extraction-dir outputs/extractions/ft_gemma/20260610-152000 \
  --extraction-dir outputs/extractions/ft_mistral/20260610-153000 \
  --extraction-dir outputs/extractions/ft_granite4/20260610-154000
  outputs/extractions/ft_granite/20260610-150000 \
  outputs/extractions/ft_smolvlm/20260610-151000 \
  outputs/extractions/ft_gemma/20260610-152000 \
  outputs/extractions/ft_mistral/20260610-153000 \
  outputs/extractions/ft_granite4/20260610-154000
```

Or use the 2nd-consensus helper:

```bash
bash scripts/prepare_2nd_consensus.sh \
  outputs/consensus_dataset_1000 \
  consensus_1000_ft \
  --threshold 4 \
  --validate \
  outputs/checkpoints/ft_model_1 \
  outputs/checkpoints/ft_model_2 \
  outputs/checkpoints/ft_model_3 \
  outputs/checkpoints/ft_model_4 \
  outputs/checkpoints/ft_model_5
```

## Inspecting Variants

List all consensus variants in a dataset:

```bash
python scripts/list_consensus_variants.py \
  --dataset-root outputs/consensus_dataset_1000
```

Output:
```
Consensus variants in outputs/consensus_dataset_1000:

1. consensus_1000
   Path: outputs/consensus_dataset_1000/consensus_1000
   Description: First consensus with 5 base models
   Extraction dirs: 5 model(s)
   Agreement threshold: 4
   Precision: 3
   Consensus transcriptions: 925
   Summary: 925 stems, 212674/355200 cells correct (59.8%)

2. consensus_1000_ft
   Path: outputs/consensus_dataset_1000/consensus_1000_ft
   Description: Second consensus with fine-tuned models
   Extraction dirs: 5 model(s)
   Agreement threshold: 4
   Precision: 3
   Consensus transcriptions: 925
   Summary: 925 stems, 237540/355200 cells correct (66.9%)
```

## Parameters Explained

### agreement_threshold

Minimum number of models that must vote for the same value for it to be marked `correct=true`.

- **2–3**: Lenient; ~75–85% coverage, ~40–50% correct cells
- **4** (recommended): Balanced; ~60% coverage, ~60% correct cells  
- **5**: Strict; ~20–30% coverage, ~95%+ correct cells

Use lower thresholds when you have fewer models or want more training data.

### precision

Decimal precision used to normalize model values before voting.

- **1**: Round to 1 decimal place (e.g., 2.5, 3.1)
- **3** (default): 3 decimal places (e.g., 2.543, 3.111)
- **5**: 5 decimal places (exact model outputs)

Precision=3 balances floating-point variance with meaningful differences.

## Backwards Compatibility

The new multi-variant system is fully backwards compatible:

### Old Single-Consensus Workflow

```bash
# This still works exactly as before
python scripts/build_consensus_transcriptions.py \
  --input-dir outputs/extractions/model_A/20260601-120000 \
  --input-dir outputs/extractions/model_B/20260601-120500 \
  --input-dir outputs/extractions/model_C/20260601-121000 \
  --input-dir outputs/extractions/model_D/20260601-121500 \
  --input-dir outputs/extractions/model_E/20260602-100000 \
  --output-dir outputs/consensus_transcriptions \
  --agreement-threshold 4 \
  --precision 3
```

When using `prepare_consensus_dataset.py` without `--variant-name`, it automatically creates both:
- New path: `consensus_dataset_1000/consensus_1000/transcriptions/`
- Old path: `consensus_dataset_1000/transcriptions/` (for compatibility)

This ensures existing scripts and workflows continue to work.

## Troubleshooting

### Config file not found

```
Error: config file not found: outputs/consensus_dataset_1000/consensus_1000/consensus_config.json
```

**Solution**: Run `create_consensus_config.py` first, or verify the config path and variant name.

### Extraction directory not found

```
Error: extraction directory not found: outputs/extractions/model_A/wrong_timestamp
```

**Solution**: Check the actual extraction timestamp in `outputs/extractions/` and use the correct path.

### No extraction JSON files found

```
Error: No extraction JSON files found in input directories
```

**Solution**: Verify that extraction jobs completed successfully and files are downloaded to the expected locations.

### Consensus coverage too low

If `agreement_coverage` in `consensus_summary.json` is below 20%:
- Lower the `--agreement-threshold` (from 4 → 3 or 2)
- Check that extraction outputs are consistent (not corrupted or partial)
- Verify that all 5 models produced outputs for the stems

## Reference: Output Schema

### Consensus Transcription JSON

Each consensus JSON has the same structure as extraction outputs:

```json
{
  "Day 1": [
    {"value": 2.543, "correct": true},    // Jan: all 5 models agree
    {"value": 1.2, "correct": false},     // Feb: only 3 models agree
    {"value": null, "correct": false},    // Mar: no agreement
    ...
  ],
  "Day 2": [...],
  ...
  "Totals": [...]
}
```

- **value**: `float | null` — the consensus value (or `null` if no agreement)
- **correct**: `bool` — `true` if ≥ `agreement_threshold` models voted for this value

### Summary JSON

```json
{
  "variant_name": "consensus_1000",
  "total_stems": 925,
  "total_cells": 355200,
  "correct_cells": 212674,
  "incorrect_cells": 142526,
  "agreement_threshold": 4,
  "precision": 3,
  "agreement_coverage": 0.5984,
  "input_dirs": ["...", "...", ...],
  "timestamp": "2026-06-01T12:00:00"
}
```

## Tips & Best Practices

1. **Use consistent model sets**: For fair comparison, keep the same 5 models across runs.
2. **Document your variants**: Add meaningful descriptions and notes in `create_consensus_config.py`.
3. **Compare summaries**: Use `list_consensus_variants.py` to see which variants performed best.
4. **Validate before fine-tuning**: Always inspect at least a sample of consensus figures to spot systematic errors.
5. **Version your checkpoints**: Track which fine-tuned checkpoints produced each 2nd/3rd consensus round.
6. **Archive results**: Keep consensus outputs on persistent storage (Azure datastore) for reproducibility.

## See Also

- [Checkpoint Management Guide](CHECKPOINT_MANAGEMENT.md) — Fine-tuning and checkpoint registration
- [Extraction Guide](docs/guide/extraction.md) — Running extractions locally or on Azure
- [Evaluation Guide](docs/guide/evaluation.md) — Comparing model quality
