"""Province-level reconciliation and residual-cost allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_COLUMNS = ["province", "crop", "year"]


def _weighted_group_mean(
    frame: pd.DataFrame, value_col: str, weight_col: str
) -> pd.Series:
    value = pd.to_numeric(frame[value_col], errors="coerce")
    weight = pd.to_numeric(frame[weight_col], errors="coerce")
    valid = value.notna() & np.isfinite(value) & weight.gt(0)
    numerator = (value.where(valid, 0.0) * weight.where(valid, 0.0)).groupby(
        [frame[c] for c in GROUP_COLUMNS]
    ).transform("sum")
    denominator = weight.where(valid, 0.0).groupby(
        [frame[c] for c in GROUP_COLUMNS]
    ).transform("sum")
    return numerator / denominator.replace(0.0, np.nan)


def _require_finite_nonnegative(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{column} must contain finite non-negative values")
    return values


def _require_group_constant(frame: pd.DataFrame, column: str) -> None:
    counts = frame.groupby(GROUP_COLUMNS, dropna=False)[column].nunique(dropna=False)
    if (counts > 1).any():
        raise ValueError(f"{column} must be constant within each province-crop-year group")


def reconcile_to_province(
    frame: pd.DataFrame,
    prediction_col: str,
    weight_col: str,
    official_col: str,
) -> pd.DataFrame:
    """Scale county predictions to official province–crop–year means."""
    required = {*GROUP_COLUMNS, prediction_col, weight_col, official_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out = frame.copy()
    prediction = _require_finite_nonnegative(out, prediction_col)
    weight = _require_finite_nonnegative(out, weight_col)
    if not weight.gt(0).any():
        raise ValueError("at least one positive allocation weight is required")
    _require_group_constant(out, official_col)
    out["preclosure_weighted_mean"] = _weighted_group_mean(
        out, prediction_col, weight_col
    )
    official = pd.to_numeric(out[official_col], errors="coerce")
    denominator = out["preclosure_weighted_mean"]
    valid = official.gt(0) & denominator.gt(0) & np.isfinite(denominator)
    out["calibration_factor"] = np.where(valid, official / denominator, 1.0)
    out["constraint_status"] = np.where(valid, "official_calibrated", "unconstrained")
    out["calibrated_cost"] = prediction * out["calibration_factor"]
    return out


def allocate_residual(
    frame: pd.DataFrame,
    residual_col: str,
    base_cost_col: str,
    weight_col: str,
    output_col: str = "allocated_residual_cost",
) -> pd.DataFrame:
    """Allocate a provincial residual in proportion to available base cost.

    allocated_i = base_cost_i * residual_province / weighted_mean(base_cost)
    """
    required = {*GROUP_COLUMNS, residual_col, base_cost_col, weight_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out = frame.copy()
    residual = _require_finite_nonnegative(out, residual_col)
    base_cost = _require_finite_nonnegative(out, base_cost_col)
    weight = _require_finite_nonnegative(out, weight_col)
    _require_group_constant(out, residual_col)
    base_mean = _weighted_group_mean(out, base_cost_col, weight_col)
    valid = weight.gt(0) & base_mean.gt(0)
    out["residual_base_weighted_mean"] = base_mean
    out["residual_multiplier"] = np.where(valid, residual / base_mean, np.nan)
    out[output_col] = np.where(valid, base_cost * out["residual_multiplier"], np.nan)
    return out


def closure_error(
    frame: pd.DataFrame,
    value_col: str,
    weight_col: str,
    official_col: str,
) -> pd.Series:
    """Return reconciled weighted mean minus official value for each row."""
    mean = _weighted_group_mean(frame, value_col, weight_col)
    return mean - pd.to_numeric(frame[official_col], errors="coerce")
