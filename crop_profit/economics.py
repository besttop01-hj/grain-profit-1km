"""Output-value allocation and net-profit accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def allocate_output_values(
    frame: pd.DataFrame,
    yield_col: str = "pixel_yield",
    provincial_mean_yield_col: str = "province_weighted_mean_yield",
    main_value_col: str = "province_main_product_value_per_mu",
    byproduct_value_col: str = "province_byproduct_value_per_mu",
) -> pd.DataFrame:
    """Allocate provincial per-mu values using relative pixel yield.

    value_i = value_province * yield_i / weighted_mean_yield_province
    """
    required = {
        yield_col,
        provincial_mean_yield_col,
        main_value_col,
        byproduct_value_col,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out = frame.copy()
    pixel_yield = pd.to_numeric(out[yield_col], errors="coerce")
    mean_yield = pd.to_numeric(out[provincial_mean_yield_col], errors="coerce")
    if pixel_yield.isna().any() or not np.isfinite(pixel_yield).all() or (pixel_yield < 0).any():
        raise ValueError("pixel yield must be finite and non-negative")
    if mean_yield.isna().any() or (mean_yield <= 0).any():
        raise ValueError("provincial mean yield must be finite and positive")
    ratio = pixel_yield / mean_yield
    out["relative_yield"] = ratio
    out["main_product_value"] = (
        pd.to_numeric(out[main_value_col], errors="coerce") * ratio
    )
    out["byproduct_value"] = (
        pd.to_numeric(out[byproduct_value_col], errors="coerce") * ratio
    )
    out["total_output_value"] = (
        out["main_product_value"] + out["byproduct_value"]
    )
    return out


def calculate_net_profit(
    frame: pd.DataFrame,
    total_output_value_col: str = "total_output_value",
    other_income_col: str = "other_income",
    total_cost_col: str = "total_production_cost",
) -> pd.DataFrame:
    """Calculate net profit in nominal CNY per mu."""
    required = {total_output_value_col, other_income_col, total_cost_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out = frame.copy()
    output = pd.to_numeric(out[total_output_value_col], errors="coerce")
    income = pd.to_numeric(out[other_income_col], errors="coerce")
    cost = pd.to_numeric(out[total_cost_col], errors="coerce")
    for name, values in ((total_output_value_col, output), (other_income_col, income),
                         (total_cost_col, cost)):
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite values")
    out["net_profit"] = output + income - cost
    out["is_loss_making"] = out["net_profit"] < 0
    return out
