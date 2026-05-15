# Visualisation

The `visualize` command produces a side-by-side diagnostic figure: the original
document image on the left, and the extracted data table on the right.  When
ground-truth data is available, table cells are coloured blue (correct) or red
(wrong/missing).

## Basic usage

**Show ground-truth data only (no model required):**

```bash
weather-extract visualize \
  Daily_rainfall_sample/images/DRain_1871-1880_Cornwall-59.jpg
```

If a matching transcription exists in `Daily_rainfall_sample/transcriptions/`,
it is loaded automatically and displayed in blue.

**Show model extraction vs ground truth:**

```bash
weather-extract visualize \
  --model smolvlm \
  Daily_rainfall_sample/images/DRain_1871-1880_Cornwall-59.jpg
```

Blue cells match ground truth; red cells differ.

**Specify an output path:**

```bash
weather-extract visualize \
  --model smolvlm \
  --output outputs/figures/my_figure.png \
  Daily_rainfall_sample/images/DRain_1871-1880_Cornwall-59.jpg
```

**Point to a ground-truth file explicitly:**

```bash
weather-extract visualize \
  --model smolvlm \
  --ground-truth path/to/truth.json \
  path/to/image.jpg
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(none)* | Model or adapter for extraction |
| `--output` | `outputs/figures/<stem>_figure.png` | Output PNG path |
| `--ground-truth` | *(auto-detected sibling JSON)* | Ground-truth JSON file |

## Cell colour scheme

| Colour | Meaning |
|--------|---------|
| Blue | Value matches ground truth (within tolerance) |
| Red | Value differs from ground truth or is missing |
| Blue (no ground truth) | All data cells default to blue |

## Figure layout

- **Left panel** — the original scanned document image, sized to its natural portrait aspect ratio.
- **Right panel** — a 32 × 12 table (Day 1–31 rows, Jan–Dec columns) with the extracted values.
- The table panel title shows the model name, or *"Training data"* when no model is used.
