# China 1-km cropland net-profit reconstruction

This repository provides a compact reference implementation of the principal
modelling, reconciliation, spatial-allocation, and validation procedures used
to construct the 2001–2024 cropland net-profit dataset for maize, winter wheat,
and rice in China.

The published raster dataset is available from figshare:
https://doi.org/10.6084/m9.figshare.34333050

## Scope

The repository includes code for:

- fitting component-specific CatBoost models to relative log-cost targets;
- frozen predictor lists and CatBoost parameters for all 12 final models;
- leave-one-province-out (LOPO) and leave-one-year-out (LOYO) diagnostics;
- reconciliation of county estimates with province-level official constraints;
- proportional allocation of a residual cost term;
- within-county rescaling of pixel predictions to reconciled county targets;
- allocation of provincial per-mu output values using relative pixel yield;
- calculation of total output value and net profit; and
- accounting-closure and basic validity checks.

The repository does **not** redistribute third-party input data subject to
licensing or institutional access restrictions. It also excludes raw survey
records, statistical-yearbook extracts, large intermediate rasters, trained
model binaries, local file paths, and internal production logs. Users must
prepare input tables and rasters from the sources described in the associated
paper. The included synthetic example demonstrates the required schemas and
executes without restricted data.

The model-family comparison and exploratory parameter-search scripts are not
included. The frozen specifications selected for the 12 final models are
provided in `config/model_specifications.json`; `cross_validate` evaluates a
specified model under LOPO or LOYO and does not repeat model-family selection.

## Installation

Python 3.11 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

## Quick start

Run the self-contained example:

```bash
python examples/run_synthetic_example.py
```

Run the tests:

```bash
python -m pytest -q
```

## Input conventions

Training records represent province–crop–year observations. Required columns
are `province`, `crop`, `year`, the selected predictor columns, and a positive
cost column. Reconciliation records additionally require a positive allocation
weight and the corresponding official provincial constraint.

Pixel-level revenue allocation requires pixel yield, the corresponding
provincial crop-area-weighted mean yield, provincial per-mu main-product and
by-product values, other income, and total production cost. All monetary values
are nominal CNY per mu unless explicitly stated otherwise.

Winter wheat is represented by the internal crop identifier `wheat` in tabular
examples and by the public raster prefix `winter_wheat`.

## Reproducibility boundary

This code exposes the analytical transformations used in the study. Exact
regeneration of the published national rasters additionally requires the
third-party data listed in the paper, the common 1-km China Albers grid, and the
crop-support layers used for each year.

## Licence

The code is released under the MIT License. The published data are released
separately under CC BY 4.0.
