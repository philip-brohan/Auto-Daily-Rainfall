# Extraction

The extraction stage runs a vision-language model over a document image and
returns a structured JSON rainfall grid.

## Running extraction

```bash
weather-extract extract --model smolvlm <image_path>
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `smolvlm` | Model preset or HuggingFace ID |

### Example

```bash
weather-extract extract \
  --model smolvlm \
  Daily_rainfall_sample/images/DRain_1871-1880_Cornwall-59.jpg
```

Output:

```json
{
  "days": {
    "Day 1": [null, 0.5, 1.2, ...],
    ...
  },
  "totals": [12.4, 8.1, ...]
}
```

## How extraction works

1. The image is loaded and passed to the model along with a prompt asking it
   to return the rainfall data as JSON.
2. The raw text response is parsed with a lenient JSON extractor that strips
   markdown fences and partial responses.
3. If parsing succeeds, a `DailyRainfallGrid` object is returned.  On failure,
   the raw model output is printed to stderr.

## Using a fine-tuned adapter

Pass the adapter directory in place of a model name:

```bash
weather-extract extract \
  --model outputs/checkpoints/HuggingFaceTB--SmolVLM-500M-Instruct \
  Daily_rainfall_sample/images/DRain_1871-1880_Cornwall-59.jpg
```

The code detects that the path is a local LoRA adapter (by checking for
`adapter_config.json`) and loads the base model automatically before applying
the adapter weights.

## Model presets

| Preset | HuggingFace ID | Notes |
|--------|----------------|-------|
| `smolvlm` | `HuggingFaceTB/SmolVLM-500M-Instruct` | Lightweight baseline |
| `granite` | `ibm-granite/granite-vision-3.2-2b` | IBM Granite Vision 3.2, 2 B |
| `gemma3` | `google/gemma-3-4b-it` | Google Gemma 3, 4 B |
| `gemma4` | `google/gemma-4-E4B-it` | Google Gemma 4, 4 B edge |
| `ministral` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Mistral Small 3.1, 24 B; 128 k context |

Any full HuggingFace model ID can also be passed directly.
Gated models (Gemma, Mistral) require a HuggingFace token set via the
`HF_TOKEN` environment variable or `~/.cache/huggingface/token`.
