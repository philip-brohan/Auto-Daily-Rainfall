# Evaluation

The evaluation stage runs extraction over all paired images and compares the
model's output against the human transcriptions.

## Running evaluation

```bash
weather-extract evaluate --model smolvlm
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `smolvlm` | Model or adapter path |
| `--limit` | *(all)* | Maximum number of images to evaluate |
| `--tolerance` | `0.005` | Absolute tolerance (inches) for a cell to count as correct |

## Output

The command prints a JSON summary followed by per-image results:

```json
{
  "total_images": 18,
  "successful_extractions": 16,
  "failed_extractions": 2,
  "cell_accuracy": 0.842,
  "mean_absolute_error": 0.31
}
```

`cell_accuracy` is the fraction of non-null data cells whose extracted value
matches the ground truth within the given tolerance.

## Evaluating a fine-tuned adapter

```bash
weather-extract evaluate \
  --model outputs/checkpoints/HuggingFaceTB--SmolVLM-500M-Instruct
```

## Limiting to a subset

Use `--limit` to do a quick sanity-check on the first few images:

```bash
weather-extract evaluate --model smolvlm --limit 3
```

## Per-image breakdown

After the summary, each image's result is printed separately, including which
specific cells were mismatched.  Redirect to a file for later analysis:

```bash
weather-extract evaluate --model smolvlm > outputs/eval_baseline.json
```
