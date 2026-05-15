# Fine-tuning

Fine-tuning adapts a pre-trained VLM to the specific visual style and layout of
your document images using [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank
Adaptation).  A LoRA adapter is a small set of additional weights that are
trained on top of the frozen base model, so the process is fast and the output
is a compact adapter directory rather than a full model copy.

## Prerequisites

Install the training extras:

```bash
pip install -e ".[train]"
```

## Running fine-tuning

```bash
weather-extract finetune --model smolvlm --epochs 3
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `smolvlm` | Base model to fine-tune |
| `--epochs` | `3` | Number of training epochs |
| `--output-dir` | `outputs/checkpoints` | Where to save the adapter |
| `--eval-split` | `0.1` | Fraction of data held out for validation |
| `--lora-r` | `8` | LoRA rank (higher = more capacity, slower) |

## What happens during fine-tuning

1. All paired image/transcription records are loaded.
2. Each record is converted to a chat-format training example with the image
   embedded and the JSON transcription as the target response.
3. A LoRA adapter is attached to the base model (targeting all linear layers by
   default).
4. The model is trained with `SFTTrainer` from the TRL library.
5. The adapter is saved to `<output_dir>/<sanitised-model-name>/`.

## Output

The adapter directory contains:

```
outputs/checkpoints/HuggingFaceTB--SmolVLM-500M-Instruct/
├── adapter_config.json      ← records the base model name
├── adapter_model.safetensors
└── tokenizer files ...
```

## Using the fine-tuned adapter

Pass the adapter directory as `--model` to any command:

```bash
weather-extract extract \
  --model outputs/checkpoints/HuggingFaceTB--SmolVLM-500M-Instruct \
  Daily_rainfall_sample/images/DRain_1871-1880_Cornwall-59.jpg
```

## Tips

- Start with `--lora-r 8` (the default).  Increase to `16` or `32` if the
  model underfits.
- Use `--eval-split 0.15` to monitor validation loss and catch overfitting.
- `smolvlm` (500 M) trains in minutes on a modern GPU; `granite` (2 B) and
  `gemma3` / `gemma4` (4 B) take longer but tend to generalise better.
  `ministral` (24 B) requires a large GPU (A100 80 GB or equivalent).
- Gated models (`gemma3`, `gemma4`, `ministral`) require a HuggingFace token —
  set `HF_TOKEN` in your environment before launching.
