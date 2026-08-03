# Operations — run on everything

The transcription models are now good enough. Time to put them to use: run the
ensemble over the full collection of scanned images and produce the rescued data.

Two notebooks:

- [run_operational_extractions_sample.ipynb](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/operations/run_operational_extractions_sample.ipynb)
- [run_operational_extractions_all.ipynb](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/operations/run_operational_extractions_all.ipynb)

## Start with a large sample

Before committing to the full dataset, the
[run_operational_extractions_sample](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/operations/run_operational_extractions_sample.ipynb)
notebook runs the second-order ensemble over a large sample of images. This
exercises the operational path at scale — batching, Azure ML job submission, and
result collection — and confirms the pipeline behaves before the full run.

## Then run everything

The [run_operational_extractions_all](https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO/blob/main/notebooks/operations/run_operational_extractions_all.ipynb)
notebook applies the ensemble to the whole collection — about **660,000** images.
Each image is independent, so the extraction is embarrassingly parallel and runs as
GPU job arrays on [Azure ML](../reference/azure.md). The output is one structured
JSON transcription per image, ready for ingestion into a database.

Requiring agreement between models in the ensemble gives both a best-estimate value
and a built-in confidence signal: cells where the models agree are highly reliable,
and cells where they disagree can be flagged for review.

## What you have after this stage

- Structured rainfall transcriptions for the full image collection.
- A repeatable, scalable process that can be re-run on new archives on demand.

For the compute setup behind these runs, see [Azure ML](../reference/azure.md).
