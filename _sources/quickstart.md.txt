# Quick Start

This page walks through the most common workflow end-to-end.

## 1 — Inspect the sample data

```bash
weather-extract ingest
```

Output shows how many image/transcription pairs were found in
`Daily_rainfall_sample/`.

## 2 — Extract a single image (baseline model)

```bash
weather-extract extract --model smolvlm Daily_rainfall_sample/images/DRain_1871-1880_Cornwall-59.jpg
```

The model downloads on first use (~1 GB) and outputs a JSON rainfall grid.

## 3 — Visualise the extraction against ground truth

```bash
weather-extract visualize \
  --model smolvlm \
  --ground-truth Daily_rainfall_sample/transcriptions/DRain_1871-1880_Cornwall-59.json \
  Daily_rainfall_sample/images/DRain_1871-1880_Cornwall-59.jpg
```

A PNG is saved to `outputs/figures/`.  Numbers are coloured:

- **Blue** — matches ground truth
- **Red** — mismatch or missing

## 4 — Evaluate the baseline over all paired images

```bash
weather-extract evaluate --model smolvlm
```

Prints per-image and aggregate accuracy statistics.

## 5 — Fine-tune on the sample data

```bash
weather-extract finetune --model smolvlm --epochs 3
```

The LoRA adapter is saved to `outputs/checkpoints/HuggingFaceTB--SmolVLM-500M-Instruct/`.

## 6 — Evaluate the fine-tuned model

Pass the adapter directory as `--model`:

```bash
weather-extract evaluate \
  --model outputs/checkpoints/HuggingFaceTB--SmolVLM-500M-Instruct
```

## 7 — Visualise fine-tuned extraction

```bash
weather-extract visualize \
  --model outputs/checkpoints/HuggingFaceTB--SmolVLM-500M-Instruct \
  Daily_rainfall_sample/images/DRain_1871-1880_Cornwall-59.jpg
```
