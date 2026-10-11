from __future__ import annotations

import numpy as np
import pandas as pd

from crop_profit.economics import allocate_output_values, calculate_net_profit
from crop_profit.reconciliation import (
    allocate_residual,
    closure_error,
    reconcile_to_province,
)
from crop_profit.spatial_allocation import pixel_closure_error, rescale_pixels_to_county


def test_province_reconciliation_closes() -> None:
    frame = pd.DataFrame(
        {
            "province": ["A", "A", "B", "B"],
            "crop": ["rice"] * 4,
            "year": [2020] * 4,
            "prediction": [100.0, 200.0, 200.0, 400.0],
            "weight": [1.0, 3.0, 2.0, 2.0],
            "official": [300.0, 300.0, 250.0, 250.0],
        }
    )
    result = reconcile_to_province(frame, "prediction", "weight", "official")
    error = closure_error(result, "calibrated_cost", "weight", "official")
    assert np.allclose(error, 0.0, atol=1e-12)


def test_relative_yield_allocation_and_profit_identity() -> None:
    frame = pd.DataFrame(
        {
            "pixel_yield": [400.0, 600.0],
            "province_weighted_mean_yield": [500.0, 500.0],
            "province_main_product_value_per_mu": [1000.0, 1000.0],
            "province_byproduct_value_per_mu": [100.0, 100.0],
            "other_income": [10.0, 10.0],
            "total_production_cost": [700.0, 1000.0],
        }
    )
    result = calculate_net_profit(allocate_output_values(frame))
    assert np.allclose(result["relative_yield"], [0.8, 1.2])
    expected = result["total_output_value"] + 10.0 - frame["total_production_cost"]
    assert np.allclose(result["net_profit"], expected)


def test_missing_official_constraint_retains_prediction() -> None:
    frame = pd.DataFrame(
        {
            "province": ["A", "A"],
            "crop": ["maize", "maize"],
            "year": [2020, 2020],
            "prediction": [100.0, 200.0],
            "weight": [1.0, 1.0],
            "official": [np.nan, np.nan],
        }
    )
    result = reconcile_to_province(frame, "prediction", "weight", "official")
    assert np.allclose(result["calibrated_cost"], frame["prediction"])
    assert set(result["constraint_status"]) == {"unconstrained"}


def test_residual_is_proportional_and_closes() -> None:
    frame = pd.DataFrame(
        {
            "province": ["A", "A"],
            "crop": ["rice", "rice"],
            "year": [2020, 2020],
            "residual": [30.0, 30.0],
            "base": [100.0, 200.0],
            "weight": [1.0, 1.0],
        }
    )
    result = allocate_residual(frame, "residual", "base", "weight")
    assert np.allclose(result["allocated_residual_cost"], [20.0, 40.0])
    assert np.allclose(
        closure_error(result, "allocated_residual_cost", "weight", "residual"),
        0.0,
    )


def test_pixel_rescaling_closes_to_county_target() -> None:
    frame = pd.DataFrame(
        {
            "county_code": ["001", "001", "001"],
            "crop": ["wheat"] * 3,
            "year": [2020] * 3,
            "raw": [100.0, 200.0, 999.0],
            "county_target": [240.0] * 3,
            "support": [1.0, 3.0, 0.0],
        }
    )
    result = rescale_pixels_to_county(
        frame, "raw", "county_target", "support"
    )
    error = pixel_closure_error(
        result, "pixel_component_cost", "county_target", "support"
    )
    assert np.allclose(error, 0.0, atol=1e-12, equal_nan=False)
    assert np.isnan(result.loc[2, "pixel_component_cost"])
