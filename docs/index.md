# Auto Daily Rainfall

**Rescuing daily rainfall observations from historical documents, using small
vision-language models instead of thousands of human volunteers.**

---

The [Rainfall Rescue project](https://climatelabbook.substack.com/p/rainfall-rescue-5-years-on)
showed that historical weather records could be recovered from paper archives by
an army of volunteers. Its successor,
[Robot Rainfall Rescue](https://brohan.org/Robot_Rainfall_Rescue/), showed that
an ensemble of small [vision-language models (VLMs)](https://huggingface.co/blog/vlms)
could do the same job for the *monthly* rainfall sheets — matching volunteer
accuracy without recruiting, training, and managing anyone.

This project applies that approach to the far larger and harder collection of
**daily** rainfall registers: about **660,000** scanned station-year images, each
a dense grid of daily rainfall totals. Success means converting every image into
a structured table of numbers that can be ingested into a database.

```{figure} _static/figures/sample_document.jpg
:alt: A scanned daily rainfall register
:width: 70%

A single scanned daily rainfall register. Each image holds one station's daily
rainfall totals (mm) — rows for days 1–31, columns for the months, plus monthly
totals. The task is to turn this into a table of numbers.
```

## The approach

We don't try to impose our own structure on the problem. Instead we take small,
open-weight VLMs, fine-tune them on the rainfall task, and let them read the
images directly into JSON. The work proceeds in staged rounds, and — importantly —
**each round is driven by a notebook** that you can open, read, and run.

1. **Preparation** — generate synthetic training data with known values, and
   import a small hand-transcribed test set for validation.
2. **Zeroth order** — measure the raw, un-fine-tuned models to establish a
   baseline.
3. **First order** — fine-tune every model on the synthetic data and re-measure.
4. **Second order** — build a *consensus* training set from where the first-order
   models agree on real images, fine-tune again on that, and re-measure.
5. **Operations** — run the finished ensemble over the full 660,000-image
   dataset.

The [workflow overview](workflow/overview.md) explains how the notebooks fit
together; the pages under it walk through each stage.

## Does it work?

Yes. Measured on 64 real, hand-transcribed images (Ciara Ryan's Irish daily
rainfall sheets), the models improve dramatically across the rounds:

| Model | Zeroth order (raw) | First order (fake) | Second order (consensus) |
|-------|:---:|:---:|:---:|
| Granite | 46% | 92% | **95%** |
| Ministral | 62% | 85% | **94%** |
| Gemma-3 | 21% | 70% | **90%** |
| Gemma edge | 39% | 66% | **89%** |
| SmolVLM | 32% | 68% | **87%** |

*Per-cell accuracy against hand-transcribed ground truth.*

The raw models are hopeless — the best gets under two thirds of the values right.
After two rounds of fine-tuning, every model is in the high eighties or better,
and an ensemble that requires agreement between models does better still. As with
the monthly sheets, this is comparable to human volunteers, and it scales.

## Get started

- [Installation](installation.md) — set up the environment.
- [Workflow overview](workflow/overview.md) — the staged, notebook-driven pipeline.
- [How to reproduce and extend](reproduce.md) — code, compute, and credits.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting started

installation
workflow/overview
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: The workflow

workflow/preparation
workflow/zeroth-order
workflow/first-order
workflow/second-order
workflow/operations
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Reference

reference/cli
reference/configuration
reference/architecture
reference/azure
reproduce
```
