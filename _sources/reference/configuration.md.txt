# Configuration Reference

All configuration is managed through Python dataclasses defined in
`src/weather_doc_extractor/config.py`.  There is no external config file;
values are changed by passing CLI flags or by editing the dataclasses directly
for programmatic use.

## `AppConfig`

The top-level configuration object, composed of the sections below.

## `IngestConfig`

Controls where the ingest stage reads data from.

| Field | Default | Description |
|-------|---------|-------------|
| `images_dir` | `Daily_rainfall_sample/images` | Directory of document images |
| `transcriptions_dir` | `Daily_rainfall_sample/transcriptions` | Directory of JSON transcriptions |
| `output_dir` | `data/dataset` | Where to write processed dataset records |

## `ModelConfig`

Controls which model is used for inference.

| Field | Default | Description |
|-------|---------|-------------|
| `model_name` | `HuggingFaceTB/SmolVLM-500M-Instruct` | HuggingFace model ID or local adapter path |
| `max_new_tokens` | `2048` | Maximum tokens the model may generate |
| `temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `device` | `auto` | Device placement (`auto`, `cpu`, `cuda`) |

### Model presets

The `--model` CLI flag accepts short preset names as well as full HuggingFace
model IDs or local adapter paths.

| Preset | Model ID | Notes |
|--------|----------|-------|
| `smolvlm` | `HuggingFaceTB/SmolVLM-500M-Instruct` | Lightweight baseline (~500 M params) |
| `smolvlm2` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | SmolVLM2, 2.2 B |
| `granite` | `ibm-granite/granite-vision-3.2-2b` | IBM Granite Vision 3.2, 2 B |
| `granite4` | `ibm-granite/granite-vision-4.1-4b` | IBM Granite Vision 4.1, 4 B |
| `gemma3` | `google/gemma-3-4b-it` | Google Gemma 3, 4 B; uses pan-and-scan tiling for high-res scans |
| `gemma4` | `google/gemma-4-E4B-it` | Google Gemma 4, 4 B edge; variable-resolution token budget |
| `ministral` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Mistral Small 3.1, 24 B; Pixtral vision encoder, 128 k context |

## `TrainingConfig`

Controls the fine-tuning process.

| Field | Default | Description |
|-------|---------|-------------|
| `output_dir` | `outputs/checkpoints` | Root directory for saved adapters |
| `learning_rate` | `2e-4` | AdamW learning rate |
| `epochs` | `3` | Number of training epochs |
| `batch_size` | `1` | Per-device training batch size |
| `gradient_accumulation_steps` | `8` | Effective batch = batch_size × this |
| `eval_split` | `0.1` | Fraction of data used for validation |
| `lora_r` | `8` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha scaling factor |
| `lora_dropout` | `0.05` | Dropout applied to LoRA layers |
| `lora_target_modules` | `None` | Linear layers to adapt; `None` = all |

## `ProjectPaths`

General-purpose path configuration (used internally by some pipeline functions).

| Field | Default |
|-------|---------|
| `data_dir` | `data` |
| `raw_images_dir` | `data/raw_images` |
| `annotations_dir` | `data/annotations` |
| `outputs_dir` | `outputs` |
| `models_dir` | `models` |
