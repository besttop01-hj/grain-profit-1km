"""Within-county allocation of modelled component costs to grid cells."""

from __future__ import annotations

import numpy as np
import pandas as pd

PIXEL_GROUP = ["county_code", "crop", "year"]


def rescale_pixels_to_county(
    frame: pd.DataFrame,
    raw_prediction_col: str,
    county_target_col: str,
    crop_support_weight_col: str,
    output_col: str = "pixel_component_cost",
) -> pd.DataFrame:
    """Rescale raw pixel predictions to a reconciled county per-mu target.

    Only pixels with positive crop-support weights participate. The rescaling
    preserves the within-county relative prediction pattern and forces the
    crop-support-weighted pixel mean to equal the county target.
    """
    required = {
        *PIXEL_GROUP,
        raw_prediction_col,
        county_target_col,
        crop_support_weight_col,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out = frame.copy()
    raw = pd.to_numeric(out[raw_prediction_col], errors="coerce")
    target = pd.to_numeric(out[county_target_col], errors="coerce")
    weight = pd.to_numeric(out[crop_support_weight_col], errors="coerce")
    if raw.isna().any() or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("raw pixel predictions must be finite and non-negative")
    if target.isna().any() or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("county targets must be finite and non-negative")
    if weight.isna().any() or not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("crop-support weights must be finite and non-negative")
    target_counts = out.groupby(PIXEL_GROUP)[county_target_col].nunique(dropna=False)
    if (target_counts > 1).any():
        raise ValueError("county target must be constant within each county-crop-year")

    valid = weight.gt(0)
    numerator = (raw.where(valid, 0.0) * weight.where(valid, 0.0)).groupby(
        [out[c] for c in PIXEL_GROUP]
    ).transform("sum")
    denominator = weight.where(valid, 0.0).groupby(
        [out[c] for c in PIXEL_GROUP]
    ).transform("sum")
    raw_mean = numerator / denominator.replace(0.0, np.nan)
    factor = target / raw_mean
    if (valid & (~np.isfinite(factor) | factor.lt(0))).any():
        raise ValueError("pixel rescaling is undefined for a supported county")
    out["raw_pixel_weighted_mean"] = raw_mean
    out["pixel_rescaling_factor"] = factor.where(valid)
    out[output_col] = (raw * factor).where(valid)
    return out


def pixel_closure_error(
    frame: pd.DataFrame,
    value_col: str,
    county_target_col: str,
    crop_support_weight_col: str,
) -> pd.Series:
    """Return weighted pixel mean minus county target for every row."""
    value = pd.to_numeric(frame[value_col], errors="coerce")
    weight = pd.to_numeric(frame[crop_support_weight_col], errors="coerce")
    valid = value.notna() & np.isfinite(value) & weight.gt(0)
    numerator = (value.where(valid, 0.0) * weight.where(valid, 0.0)).groupby(
        [frame[c] for c in PIXEL_GROUP]
    ).transform("sum")
    denominator = weight.where(valid, 0.0).groupby(
        [frame[c] for c in PIXEL_GROUP]
    ).transform("sum")
    return numerator / denominator.replace(0.0, np.nan) - frame[county_target_col]
