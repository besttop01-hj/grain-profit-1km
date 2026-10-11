"""Run the public workflow on synthetic, non-study data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crop_profit.cost_model import ModelConfig, cross_validate  # noqa: E402
from crop_profit.economics import allocate_output_values, calculate_net_profit  # noqa: E402
from crop_profit.reconciliation import closure_error, reconcile_to_province  # noqa: E402


def make_training_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for province in [f"P{i:02d}" for i in range(1, 9)]:
        for year in range(2018, 2024):
            soil = rng.normal(0.0, 1.0)
            climate = rng.normal(0.0, 1.0)
            cost = np.exp(5.5 + 0.25 * soil - 0.20 * climate)
            rows.append(
                {
                    "province": province,
                    "crop": "maize",
                    "year": year,
                    "soil": soil,
                    "climate": climate,
                    "component_cost": cost,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    training = make_training_data()
    _, metrics = cross_validate(
        training,
        features=["soil", "climate"],
        cost_col="component_cost",
        protocol="LOPO",
        config=ModelConfig(iterations=100, depth=3),
    )

    county = pd.DataFrame(
        {
            "province": ["P01", "P01", "P02", "P02"],
            "crop": ["maize"] * 4,
            "year": [2023] * 4,
            "prediction": [380.0, 420.0, 460.0, 500.0],
            "crop_area_ha": [20.0, 30.0, 40.0, 20.0],
            "official_cost": [410.0, 410.0, 470.0, 470.0],
        }
    )
    reconciled = reconcile_to_province(
        county, "prediction", "crop_area_ha", "official_cost"
    )
    max_closure_error = float(
        closure_error(
            reconciled, "calibrated_cost", "crop_area_ha", "official_cost"
        ).abs().max()
    )

    pixels = pd.DataFrame(
        {
            "pixel_yield": [400.0, 600.0],
            "province_weighted_mean_yield": [500.0, 500.0],
            "province_main_product_value_per_mu": [900.0, 900.0],
            "province_byproduct_value_per_mu": [60.0, 60.0],
            "other_income": [10.0, 10.0],
            "total_production_cost": [750.0, 900.0],
        }
    )
    pixels = allocate_output_values(pixels)
    pixels = calculate_net_profit(pixels)

    summary = {
        "lopo_metrics": metrics,
        "maximum_reconciliation_error": max_closure_error,
        "pixel_net_profit": pixels["net_profit"].round(6).tolist(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
