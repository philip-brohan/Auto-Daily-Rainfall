# Installation

Everything in this project — every script and every notebook — is intended to run
inside the `weather-doc-extractor` Conda environment. Setting that up is the only
prerequisite for reading and running the workflow notebooks.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git
- A local machine is enough for orchestration and validation. The heavy work —
  extraction and fine-tuning — runs on GPUs via [Azure ML](reference/azure.md);
  see that page for the compute setup.

## 1 — Clone the repository

```bash
git clone https://github.com/Philip-Brohan-MO/Auto-Daily-Rainfall-MO.git
cd Auto-Daily-Rainfall-MO
```

## 2 — Create the Conda environment

The `environment.yml` file pins all dependencies including PyTorch, Transformers,
TRL, and PEFT:

```bash
conda env create -f environment.yml
conda activate weather-doc-extractor
```

```{important}
Activate `weather-doc-extractor` before running any script or notebook. This is
non-negotiable for reproducibility — the notebooks assume this environment.
```

## 3 — Install the package in editable mode

The Conda environment already runs `pip install -e .` as part of its post-link
step.  If you need to reinstall manually:

```bash
pip install -e .
```

This makes the `weather-extract` command available on your PATH.

## 4 — Verify the installation

```bash
weather-extract info
```

You should see a JSON summary of the project configuration.

## Optional: training dependencies

The base install is inference-only.  To enable fine-tuning, install the
`train` extras:

```bash
pip install -e ".[train]"
```

This adds `accelerate`, `datasets`, `peft`, `torch`, `transformers`, and `trl`.

## Updating

```bash
git pull
conda env update -f environment.yml --prune
```
