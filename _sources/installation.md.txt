# Installation

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git
- A machine with at least 4 GB RAM (8 GB+ recommended for the 2 B model)
- A CUDA-capable GPU is optional but speeds up inference and fine-tuning significantly

## 1 — Clone the repository

```bash
git clone https://github.com/philip-brohan/Auto-Daily-Rainfall.git
cd Auto-Daily-Rainfall
```

## 2 — Create the Conda environment

The `environment.yml` file pins all dependencies including PyTorch, Transformers,
TRL, and PEFT:

```bash
conda env create -f environment.yml
conda activate weather-doc-extractor
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
