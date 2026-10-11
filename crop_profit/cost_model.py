"""Component-cost modelling and grouped cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

Protocol = Literal["LOPO", "LOYO"]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration shared by component-specific CatBoost models."""

    iterations: int = 500
    depth: int = 6
    learning_rate: float = 0.03
    loss_function: str = "RMSE"
    random_seed: int = 20260915
    verbose: bool = False


def _build_model(config: ModelConfig) -> Pipeline:
    params = {
        "iterations": config.iterations,
        "depth": config.depth,
        "learning_rate": config.learning_rate,
        "loss_function": config.loss_function,
        "random_seed": config.random_seed,
        "verbose": config.verbose,
        "allow_writing_files": False,
    }
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CatBoostRegressor(**params)),
        ]
    )


def _validate_training_frame(
    frame: pd.DataFrame, features: list[str], cost_col: str
) -> None:
    required = {"province", "crop", "year", cost_col, *features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    cost = pd.to_numeric(frame[cost_col], errors="coerce")
    if cost.isna().any() or not np.isfinite(cost).all() or (cost <= 0).any():
        raise ValueError(f"{cost_col} must contain finite positive values")


def _relative_log_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    log_cost_col: str,
    protocol: Protocol,
) -> tuple[pd.Series, np.ndarray]:
    """Construct fold-specific targets without using held-out observations."""
    annual_baseline = train.groupby("year")[log_cost_col].mean()
    y_train = train[log_cost_col] - train["year"].map(annual_baseline)

    if protocol == "LOPO":
        baseline = test["year"].map(annual_baseline)
        if baseline.isna().any():
            raise ValueError("a held-out LOPO year is absent from the training fold")
        return y_train, baseline.to_numpy(dtype=np.float64)

    # A held-out year has no same-year training baseline. LOYO is therefore a
    # complementary temporal diagnostic using the training-fold grand mean.
    return y_train, np.full(len(test), train[log_cost_col].mean(), dtype=np.float64)


def cross_validate(
    frame: pd.DataFrame,
    features: list[str],
    cost_col: str,
    protocol: Protocol = "LOPO",
    config: ModelConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return out-of-fold predictions and summary metrics.

    Models must be fitted separately for each crop–component combination.
    """
    if protocol not in {"LOPO", "LOYO"}:
        raise ValueError("protocol must be 'LOPO' or 'LOYO'")
    _validate_training_frame(frame, features, cost_col)
    if frame["crop"].nunique() != 1:
        raise ValueError("cross_validate expects one crop at a time")
    config = config or ModelConfig()
    data = frame.copy()
    data["_log_cost"] = np.log(data[cost_col].astype(np.float64))
    group_col = "province" if protocol == "LOPO" else "year"
    heldouts = sorted(data[group_col].drop_duplicates().tolist())
    if len(heldouts) < 2:
        raise ValueError(f"{protocol} requires at least two {group_col} groups")

    parts: list[pd.DataFrame] = []
    for heldout in heldouts:
        test_mask = data[group_col].eq(heldout)
        train = data.loc[~test_mask].copy()
        test = data.loc[test_mask].copy()
        y_train, baseline = _relative_log_target(train, test, "_log_cost", protocol)
        model = _build_model(config)
        model.fit(train[features], y_train)
        model_output = np.asarray(model.predict(test[features]), dtype=np.float64)
        predicted = np.exp(model_output + baseline)
        parts.append(
            pd.DataFrame(
                {
                    "province": test["province"].to_numpy(),
                    "crop": test["crop"].to_numpy(),
                    "year": test["year"].to_numpy(),
                    "protocol": protocol,
                    "heldout_unit": heldout,
                    "observed_cost": test[cost_col].to_numpy(dtype=np.float64),
                    "predicted_cost": predicted,
                    "training_fold_baseline": baseline,
                    "model_output": model_output,
                }
            )
        )

    oof = pd.concat(parts, ignore_index=True)
    observed = oof["observed_cost"].to_numpy()
    predicted = oof["predicted_cost"].to_numpy()
    rho = spearmanr(observed, predicted).statistic
    metrics = {
        "n": int(len(oof)),
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman_rho": float(rho),
    }
    return oof, metrics


def fit_full_model(
    frame: pd.DataFrame,
    features: list[str],
    cost_col: str,
    config: ModelConfig | None = None,
) -> tuple[Pipeline, pd.Series]:
    """Fit a full-sample relative-log model and return annual baselines."""
    _validate_training_frame(frame, features, cost_col)
    config = config or ModelConfig()
    data = frame.copy()
    log_cost = np.log(data[cost_col].astype(np.float64))
    baseline = log_cost.groupby(data["year"]).mean()
    target = log_cost - data["year"].map(baseline)
    model = _build_model(config)
    model.fit(data[features], target)
    return model, baseline
