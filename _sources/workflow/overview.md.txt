# Workflow overview

From a user's point of view, the whole project is controlled through a series of
[Jupyter notebooks](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/tree/main/notebooks).
Each notebook documents and drives one stage of the pipeline: it submits the heavy
work to [Azure ML](../reference/azure.md), waits for the jobs, downloads the
results, and produces validation summaries. To reproduce or extend the project, you
open the notebooks and run them in order.

```{important}
Run every notebook in the `weather-doc-extractor` Conda environment
(`conda activate weather-doc-extractor`). See [Installation](../installation.md).
```

## The stages

The notebooks are grouped into stages, run top to bottom:

```text
preparation/          Download the source images, build synthetic training
      │               data, and import the real test set
      ▼
zeroth_order/         Measure the raw, un-fine-tuned models  (baseline)
      │
      ▼
first_order/          Fine-tune on synthetic data, then re-measure
      │
      ▼
second_order/         Build a consensus set from real images, fine-tune again,
      │               then re-measure
      ▼
operations/           Run the finished ensemble over the full dataset
```

Two ideas run through the whole workflow:

- **Fake data for training, real data for testing.** We can cheaply generate
  synthetic images whose values we know, and use them to teach the models the
  task. We keep a small, precious set of hand-transcribed *real* images
  (from [Ciara Ryan](../reproduce.md#credits-and-acknowledgements)) purely for
  validation, so every accuracy number is measured against genuine records.
- **Consensus instead of ground truth.** We have no transcriptions for the real
  images, so we can't fine-tune on them directly. Instead we run several models,
  keep the cells where they *agree*, and treat that agreement as training truth
  for a second round of fine-tuning.

## The notebooks, in order

| # | Stage | Notebook | What it does |
|---|-------|----------|--------------|
| 1 | Preparation | [download_document_images](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/preparation/download_document_images.ipynb) | Download the archive PDFs and split them into per-page images |
| 2 | Preparation | [make_fake_daily_rainfall_training_data](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/preparation/make_fake_daily_rainfall_training_data.ipynb) | Generate ~1000 synthetic images with known values |
| 3 | Preparation | [add_test_data_from_Ciara](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/preparation/add_test_data_from_Ciara.ipynb) | Import the 64-image hand-transcribed test set |
| 4 | Zeroth order | [validate_0th_order_fake](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/zeroth_order/validate_0th_order_fake.ipynb) | Baseline of raw models on fake data |
| 5 | Zeroth order | [validate_0th_order_real](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/zeroth_order/validate_0th_order_real.ipynb) | Baseline of raw models on real data |
| 6 | First order | [finetune_original_models](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/first_order/finetune_original_models.ipynb) | Fine-tune all models on the synthetic data |
| 7 | First order | [validate_1st_order_fake](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/first_order/validate_1st_order_fake.ipynb) | First-order models on fake data |
| 8 | First order | [validate_1st_order_real](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/first_order/validate_1st_order_real.ipynb) | First-order models on real data |
| 9 | Second order | [make_1st_training_consensus](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/second_order/make_1st_training_consensus.ipynb) | Build the consensus training set from real images |
| 10 | Second order | [finetune_1st_order_models](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/second_order/finetune_1st_order_models.ipynb) | Fine-tune again on the consensus set |
| 11 | Second order | [validate_2nd_order_real](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/second_order/validate_2nd_order_real.ipynb) | Second-order models on real data |
| 12 | Operations | [run_operational_extractions_sample](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/operations/run_operational_extractions_sample.ipynb) | Ensemble extraction on a large sample |
| 13 | Operations | [run_operational_extractions_all](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/operations/run_operational_extractions_all.ipynb) | Ensemble extraction on all ~660,000 images |

Work through the stage pages next, starting with [Preparation](preparation.md).

## Reading the validation figures

The validation notebooks produce a comparison figure for each test image: the
scanned document on the left, and the model's transcription laid over the table
grid. Numbers in **blue** match the ground truth; numbers in **red** are wrong.
These figures appear throughout the stage pages that follow, and make the effect
of each round of fine-tuning immediately visible.
