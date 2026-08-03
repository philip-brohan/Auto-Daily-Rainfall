# Preparation

Before we can train or measure anything, we need the source images to work on,
plus two datasets: something to *train* on, and something to *test* on. This
stage builds all three.

Three notebooks:

- [download_document_images.ipynb](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/preparation/download_document_images.ipynb)
- [make_fake_daily_rainfall_training_data.ipynb](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/preparation/make_fake_daily_rainfall_training_data.ipynb)
- [add_test_data_from_Ciara.ipynb](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/preparation/add_test_data_from_Ciara.ipynb)

## The source document images

The raw material is the scanned daily-rainfall registers held by the Met Office
[National Meteorological Library and Archive](https://digital.nmla.metoffice.gov.uk/index.php?name=SO_51194883-b9dd-4e27-93db-958f8fbea38b)
(published under the Open Government Licence). The archive stores one multi-page
PDF per county per decade; every page is one station-year rainfall table.

The [download_document_images](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/preparation/download_document_images.ipynb)
notebook downloads those PDFs (`scripts/download_documents.py`) and splits them
into single-page JPEGs (`scripts/split_documents.py`), named like
`DRain_1871-1880_Cornwall-59.jpg`. Both steps are idempotent and support cluster
sharding, because the full collection runs to hundreds of gigabytes.

## Synthetic training data

Transcribing images into data is hard — that is the whole problem we are trying to
solve. But the reverse is easy: given some numbers, we can write a Python program
that lays them out into an image with the same structure as a real Daily Weather
Record. So we generate about **1000 fake images** whose values we already know, and
use them as training data.

The [make_fake_daily_rainfall_training_data](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/preparation/make_fake_daily_rainfall_training_data.ipynb)
notebook produces paired `images/` and `transcriptions/` — each image with a JSON
file holding its exact values. The capabilities the models learn on this synthetic
data carry over to the real records.

```{figure} ../_static/figures/sample_document.jpg
:alt: A real daily rainfall register
:width: 65%

A real daily rainfall register. The synthetic images imitate this layout — the same
day rows, month columns, and monthly totals — so that skills learned on fake data
transfer to the real thing.
```

## Real test data

To know whether the models are actually working, we test them against *real*
images with known-good values. **Ciara Ryan** transcribed and quality-controlled
64 Irish daily rainfall sheets during her PhD; these share the format of the UK
records we are targeting. Sixty-four images is not enough to train on, but it is
ideal for validation.

The [add_test_data_from_Ciara](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/preparation/add_test_data_from_Ciara.ipynb)
notebook reformats her results into the project's `images/` + `transcriptions/`
layout, giving a trustworthy real-data test set that every later stage measures
against.

## What you have after this stage

- The real archive document images, split into per-page JPEGs.
- ~1000 synthetic training images with perfect labels.
- 64 real test images with hand-checked ground truth.

Next: measure the raw models against both, in [Zeroth order](zeroth-order.md).
