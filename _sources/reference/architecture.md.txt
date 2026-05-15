# Architecture

## Module overview

```
src/weather_doc_extractor/
├── cli.py          Entry point — routes CLI commands
├── config.py       Dataclasses: AppConfig, ModelConfig, TrainingConfig, …
├── schemas.py      DailyRainfallGrid data model
├── ingest.py       Scan images/transcriptions → DatasetRecord list
├── inference.py    Load model, run extraction, parse JSON response
├── evaluate.py     Compare extracted grids to ground truth
├── finetune.py     LoRA fine-tuning with TRL SFTTrainer
├── pipeline.py     Orchestration: wraps the above into single-call functions
└── visualize.py    Diagnostic figure generation (matplotlib)
```

## Data flow

```
images/ + transcriptions/
         │
         ▼
      ingest.py
    DatasetRecord[]
         │
    ┌────┴────────────────────────────────────┐
    │                                         │
    ▼                                         ▼
inference.py                            finetune.py
  DailyRainfallGrid                     LoRA adapter
         │                                    │
         ▼                                    │
   evaluate.py ◄──────────────────────────────┘
   EvalReport
         │
         ▼
   visualize.py
   diagnostic PNG
```

## Key design decisions

### Model-agnostic inference

`inference.py` detects the model family (`smolvlm` or `granite`) from the model
name and applies the appropriate chat template and image tokenisation.  New
model families can be added by extending `detect_model_family()` and the
collate function in `finetune.py`.

### LoRA adapter detection

When `--model` points to a local directory containing `adapter_config.json`,
the code automatically:

1. Reads the base model name from `adapter_config["base_model_name_or_path"]`.
2. Loads the base model.
3. Wraps it with `PeftModel.from_pretrained(base, adapter_dir)`.

This means the same `--model` flag works for both base models and fine-tuned
adapters with no extra flags required.

### Training example construction

`finetune.py` builds training examples in the format expected by each model
family:

- **SmolVLM** — uses an `{"type": "image"}` placeholder in the content list;
  the processor resolves the placeholder to pixel values.
- **Granite** — embeds the PIL image object directly in the content list.

### Visualisation positioning

All panel sizing and positioning in `visualize.py` is done via `fig.add_axes()`
coordinates.  The `ax.table()` `bbox` parameter is always `[0, 0, 1, 1]` (fill
the axes).  This avoids double-applying offsets.

## Testing

Tests live in `tests/` and use `pytest`.  Every module has a corresponding test
file.  Tests avoid loading real models by mocking `transformers` calls.

```bash
/path/to/weather-doc-extractor/bin/pytest tests/ -q
```
