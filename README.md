# Code for "A 1-km annual dataset of crop-specific net profit for wheat, maize and rice across China, 2000–2019"

## Repository structure

```
├── 01_cost_downscaling/              # Stage 3: Cost spatial downscaling
│   ├── detrend_xgboost_loyo.py       #   Detrending + XGBoost + LOYO cross-validation
│   ├── model_comparison.py           #   Ridge / Random Forest / XGBoost comparison
│   ├── bias_correction.py            #   Multiplicative bias correction
│   └── plot_model_comparison_scatter.py
│
└── 02_revenue_and_net_profit/        # Stage 4–5: Revenue + net profit
    ├── yield_price_revenue.py        #   EVI-based yield downscaling + main product revenue
    ├── byproduct_value.py            #   By-product output value
    └── net_profit_calibrate.py       #   Net profit = revenue − cost; additive calibration
```

## Environment

Python 3.11+, with dependencies listed in `environment.yml`:

```
conda env create -f environment.yml
```

Core libraries: `xgboost`, `shap`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `scipy`, `gdal`

## Input data

Scripts assume the input data (`.csv` and `.tif` files) are organised according to the paths
specified within each script. The preprocessed input data — including 15 spatial explanatory
variables aggregated to provincial means, detrended cost residuals, and 1-km crop
distribution masks — is available from the corresponding author upon request.

The 1-km net profit dataset itself is publicly available at Zenodo:
https://doi.org/10.5281/zenodo.20116880

## Usage

Each script can be run independently. Example:

```bash
cd 01_cost_downscaling
python detrend_xgboost_loyo.py
```

Scripts expect the input directory paths to be updated for your local environment.
