# Zeroth order — the baseline

Before any fine-tuning, we measure the models exactly as they come from
Hugging Face. This "zeroth order" baseline is what every later improvement is
judged against.

Two notebooks:

- [validate_0th_order_fake.ipynb](https://github.com/Philip-Brohan/Auto-Daily-Rainfall/blob/main/notebooks/zeroth_order/validate_0th_order_fake.ipynb) tests the performance on synthetic data
- [validate_0th_order_real.ipynb](https://github.com/Philip-Brohan/Auto-Daily-Rainfall/blob/main/notebooks/zeroth_order/validate_0th_order_real.ipynb) tests the performance on real examples (Ciara Ryan's Irish dsample).

Each notebook specifies the models to use, runs an extraction job for the test set
on Azure ML, downloads the transcriptions, and compares them against the ground
truth to produce validation summaries and comparison figures.

## The raw models are not good enough

On the real test set, the un-fine-tuned models range from hopeless to mediocre:

| Model | Accuracy on real test set |
|-------|:---:|
| Ministral | 62% |
| Granite | 46% |
| Gemma edge | 39% |
| SmolVLM | 32% |
| Gemma-3 | 21% |

Even the best model gets fewer than two thirds of the values right. Here is what
that looks like on a single test image. Because we are running an ensemble of models, we are looking for cases where there is a consensus between the models (at least three agree), and where that consensus is right. Dark blue values indicate we have a consensus, and it is right, red values have a consensus, but it's wrong, and pale values have no consensus. Dark blue is the colour of success - no consensus is failure, but we know we've failed - red is bad that's where we think we have succeeded, but we are wrong.:

```{figure} ../_static/figures/zeroth_order_example.png
:alt: Raw model transcription with many errors
:width: 90%

A raw, un-fine-tuned model on one real test image. It has little consensus — nowhere near good enough to use.
```

That is the starting point. The next stage shows how much a single round of
fine-tuning helps.

## What you have after this stage

- Baseline accuracy for every model, on both fake and real data.
- A yardstick to measure the effect of fine-tuning.

Next: [First order](first-order.md).
