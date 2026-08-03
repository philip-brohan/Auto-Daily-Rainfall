# Auto Daily Rainfall MO

Pipeline for extracting daily rainfall tables from historical document images,
building consensus labels, and fine-tuning multimodal models in staged rounds.

## Scope

- Input scale: approximately 660,000 JPEG station-year images.
- Output target: station metadata plus daily and monthly precipitation values.
- Primary compute: Azure ML jobs for extraction, evaluation, and fine-tuning.

## Environment (required)

All scripts and notebooks must be run in the `weather-doc-extractor` Conda
environment.

```bash
conda env create -f environment.yml
conda activate weather-doc-extractor
python -m unittest discover -s tests
```

## Workflow

The end-to-end workflow is now documented and executed through notebooks plus
the `scripts/` automation wrappers.

High-level sequence:

1. Build synthetic training data and run first-order fine-tuning.
2. Run multi-model extraction on real samples.
3. Build and validate first-order consensus labels.
4. Run second-order fine-tuning on consensus data.
5. Run second-order extractions and validation.

Core notebook sequence (run in order):

1. `notebooks/preparation/download_document_images.ipynb`
2. `notebooks/preparation/make_fake_daily_rainfall_training_data.ipynb`
3. `notebooks/preparation/add_test_data_from_Ciara.ipynb`
4. `notebooks/zeroth_order/validate_0th_order_fake.ipynb`
5. `notebooks/zeroth_order/validate_0th_order_real.ipynb`
6. `notebooks/first_order/finetune_original_models.ipynb`
7. `notebooks/first_order/validate_1st_order_fake.ipynb`
8. `notebooks/first_order/validate_1st_order_real.ipynb`
9. `notebooks/second_order/make_1st_training_consensus.ipynb`
10. `notebooks/second_order/finetune_1st_order_models.ipynb`
11. `notebooks/second_order/validate_2nd_order_real.ipynb`
12. `notebooks/operations/run_operational_extractions_sample.ipynb`
13. `notebooks/operations/run_operational_extractions_all.ipynb`

Note: both fine-tuning notebooks are required and run sequentially because they
cover different stages.

## Ciara's data assets

Ciara Ryan provided 64 manually-transcribed images - these are used for validation:

- `notebooks/preparation/add_test_data_from_Ciara.ipynb`
- `scripts/convert_ciara_test_data.py`
- `test_data/from_Ciara/`

## Project structure

- `scripts/`: operational entrypoints for Azure ML submission, upload, and
  orchestration.
- `src/weather_doc_extractor/`: extraction, evaluation, and fine-tuning logic.
- `azureml/`: Azure ML environments and job specs.
- `notebooks/`: staged workflow documentation and validation notebooks.
- `outputs/model_registry.json`: model/checkpoint registry.
- `outputs/extraction_registry.json`: extraction run registry.

## Key automation scripts

- `scripts/aml_submit.sh`: submit extraction/evaluation/fine-tuning jobs.
- `scripts/aml_upload.sh`: upload datasets and artifacts.
- `scripts/run_extract.sh`: extraction entrypoint used by Azure jobs.
- `scripts/build_consensus_transcriptions.py`: build consensus transcription
  datasets.
- `scripts/build_ensemble_transcriptions.py`: build per-model value arrays
  without consensus voting.

## Documentation

- Online docs (GitHub Pages): <https://Philip-Brohan-MO.github.io/Auto-Daily-Rainfall-MO/>
- `docs/`: Sphinx site sources (Markdown via MyST); published automatically on
  push to `main`. Build locally with:

  ```bash
  pip install sphinx myst-parser
  sphinx-build -b html docs docs/_build/html
  ```

- `CONSENSUS_GUIDE.md`: consensus data and validation workflow.
- `CHECKPOINT_MANAGEMENT.md`: checkpoint and registry management.
