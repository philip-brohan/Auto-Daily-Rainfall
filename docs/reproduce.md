# How to reproduce and extend

This project is designed to be reproduced and extended. Everything — code, notebooks,
environment specification, and this documentation — lives in a single Git repository:
[github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO](https://github.com/Philip-Brohan/Auto-Daily-Rainfall).

If you are familiar with GitHub, fork or clone the repository. If you'd rather not,
you can download the whole thing as a
[zip file](https://github.com/Philip-Brohan/Auto-Daily-Rainfall/archive/refs/heads/main.zip).

## Software environment

Everything runs in the `weather-doc-extractor` Conda environment specified in
`environment.yml`. See [Installation](installation.md) for the one-time setup, then
activate it before doing anything else:

```bash
conda activate weather-doc-extractor
```

## Compute

The local machine is used only for orchestration and validation. The
compute-intensive work — model extraction over hundreds of thousands of images,
and GPU fine-tuning — runs on [Microsoft Azure ML](reference/azure.md). The training
and inference code itself is not Azure-specific and will run in any suitable Python
environment with access to a GPU, but the submission scripts and job specs under
`azureml/` and `scripts/` are written for an Azure ML workspace.

## Running the workflow

The workflow is driven by the notebooks under `notebooks/`, run in order. Start
with the [workflow overview](workflow/overview.md), which lays out the stages and
links to each notebook.

## The documentation

These web pages are built with [Sphinx](https://www.sphinx-doc.org/) from the
Markdown sources in the `docs/` directory, and published to
[GitHub Pages](https://pages.github.com/) automatically on every push to `main`.
To build them locally:

```bash
pip install sphinx myst-parser
sphinx-build -b html docs docs/_build/html
```

This document is distributed under the terms of the
[Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/2/).
Source code is distributed under the terms of the
[BSD licence](https://opensource.org/licenses/BSD-2-Clause).
