# Lab Notebook Template

Use this as the human-readable companion to any run of extraction, validation, or fine-tuning.
Keep the notebook file in the top-level `notebooks/` directory, not in `logs/`.

## Purpose

- Record the exact commands used to produce a model or output.
- Keep the sequence readable and copy/paste friendly.
- Capture the context that matters later: checkpoint path, dataset path, job id, and any deviations.

## Suggested workflow

1. Write a short session header with date, run label, and git commit.
2. Copy one command block into a terminal and run it.
3. Paste the result into the notebook or append a short note.
4. Record the job id, output path, and any observations.
5. Repeat for the next step.

## Template structure

- Session setup
- Inputs and fixed paths
- Command blocks
- Observations
- Results and next steps

## Four examples

### 1. Extraction run

Use this when you are generating model outputs from a checkpoint.

```bash
bash scripts/aml_submit.sh \
	--checkpoint Daily_rainfall_sample/outputs/checkpoints/<run>/<checkpoint> \
	--images-path documents/DailyRainfall_UK/consensus_1000/images \
	--transcriptions-path documents/DailyRainfall_UK/consensus_1000/transcriptions_2 \
	--total-shards 1 \
	--batch-size 50 \
	--extraction-registry outputs/extraction_registry.json \
	extract
```

Record:
- checkpoint used
- job id(s)
- output registry entry
- any shard or batch-size changes

### 2. Standard fine-tune

Use this for a fresh LoRA fine-tune starting from the base model.

```bash
bash scripts/aml_submit.sh \
	--model granite4 \
	--images-path Daily_rainfall_sample/images \
	--transcriptions-path Daily_rainfall_sample/transcriptions \
	finetune
```

Record:
- selected model preset or model id
- dataset paths
- checkpoint output location
- job id

### 3. Consensus fine-tune

Use this when training on strict consensus transcriptions.

```bash
bash scripts/aml_submit.sh \
	--model Daily_rainfall_sample/outputs/checkpoints/<run>/<checkpoint> \
	--images-path documents/DailyRainfall_UK/consensus_1000/images \
	--consensus-transcriptions-path documents/DailyRainfall_UK/consensus_1000/transcriptions \
	finetune-consensus
```

Record:
- input checkpoint path
- consensus transcription path
- model family
- output checkpoint path
- job id

### 4. Evaluation or validation run

Use this after extraction to measure output quality or generate figures.

```bash
bash scripts/aml_submit.sh \
	--dataset test_real \
	--checkpoint Daily_rainfall_sample/outputs/checkpoints/<run>/<checkpoint> \
	--limit 10 \
	--total-shards 1 \
	extract
```

Record:
- evaluation dataset name
- checkpoint under test
- limits or shard counts
- result path or report path

## What to record for each command

- Timestamp in UTC
- Command exactly as run
- Input checkpoint or dataset paths
- Output path or job id
- Whether it completed successfully
- Any warnings or unexpected behavior

## Recommended notebook naming

Use a name that includes the run date and task, for example:

- `notebooks/2026-06-12-consensus-round2.ipynb`
- `notebooks/2026-06-12-granite-consensus-validation.ipynb`

## Minimal entry format

```text
2026-06-12T12:34:56Z | consensus-round2 | submit extraction for granite4 | job=... | ok
```

## Notes

- Keep the notebook linear and short.
- If you rerun a step, record both the failed attempt and the fix.
- Use the runbook to preserve the narrative; use the notebook to preserve the exact commands and outcomes.