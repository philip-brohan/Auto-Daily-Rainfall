# CLI Reference

The `weather-extract` command is the primary interface to the project.

```
weather-extract <command> [options]
```

## Commands

### `info`

Print the current project configuration as JSON.

```bash
weather-extract info
```

---

### `stages`

Print a human-readable description of each pipeline stage.

```bash
weather-extract stages
```

---

### `ingest`

Scan the sample directory and report how many image/transcription pairs exist.

```bash
weather-extract ingest
```

---

### `extract`

Run a VLM over a single image and print the extracted JSON.

```bash
weather-extract extract [--model <name|id|path>] <image_path>
```

| Flag | Description |
|------|-------------|
| `--model` | Model preset (`smolvlm`, `smolvlm2`, `granite`, `granite4`), HuggingFace ID, or local adapter path |

---

### `evaluate`

Evaluate a model over all paired images and print accuracy statistics.

```bash
weather-extract evaluate [--model <name>] [--limit N] [--tolerance F]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `smolvlm` | Model or adapter path |
| `--limit` | all | Maximum images to evaluate |
| `--tolerance` | `0.005` | Tolerance for cell match (inches) |

---

### `finetune`

Fine-tune a base model on the paired dataset using LoRA.

```bash
weather-extract finetune [--model <name>] [--epochs N] [--output-dir PATH] \
                         [--eval-split F] [--lora-r N]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `smolvlm` | Base model to fine-tune |
| `--epochs` | `3` | Training epochs |
| `--output-dir` | `outputs/checkpoints` | Where to save the adapter |
| `--eval-split` | `0.1` | Validation fraction |
| `--lora-r` | `8` | LoRA rank |

---

### `visualize`

Build a diagnostic figure (PNG) showing the image alongside the extracted table.

```bash
weather-extract visualize [--model <name>] [--output PATH] \
                          [--ground-truth PATH] <image_path>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(none)* | Model or adapter for extraction |
| `--output` | `outputs/figures/<stem>_figure.png` | Output path |
| `--ground-truth` | *(sibling JSON)* | Ground-truth file |

---

## Model presets

| Preset | HuggingFace ID |
|--------|---------------|
| `smolvlm` | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| `smolvlm2` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` |
| `granite` | `ibm-granite/granite-vision-3.2-2b` |
| `granite4` | `ibm-granite/granite-vision-4.1-4b` |

Any full HuggingFace model ID or a local adapter directory can be passed to
`--model` in place of a preset name.
