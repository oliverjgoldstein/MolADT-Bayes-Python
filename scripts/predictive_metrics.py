from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

GAUSSIAN_Z_90 = 1.6448536269514722
CALIBRATION_LEVELS = (0.5, 0.8, 0.9, 0.95)


def gaussian_mean_log_predictive_density(
    actual: np.ndarray,
    predicted_mean: np.ndarray,
    predictive_sd: np.ndarray,
) -> float:
    sd = np.clip(np.asarray(predictive_sd, dtype=float), 1e-6, None)
    mean = np.asarray(predicted_mean, dtype=float)
    y = np.asarray(actual, dtype=float)
    log_density = -0.5 * np.log(2.0 * math.pi * np.square(sd)) - 0.5 * np.square((y - mean) / sd)
    return float(np.mean(log_density))


def gaussian_interval_coverage(
    actual: np.ndarray,
    predicted_mean: np.ndarray,
    predictive_sd: np.ndarray,
    *,
    nominal: float = 0.9,
) -> float:
    if nominal != 0.9:
        z_value = _z_value_for_nominal(nominal)
    else:
        z_value = GAUSSIAN_Z_90
    mean = np.asarray(predicted_mean, dtype=float)
    sd = np.clip(np.asarray(predictive_sd, dtype=float), 1e-6, None)
    y = np.asarray(actual, dtype=float)
    lower = mean - z_value * sd
    upper = mean + z_value * sd
    return float(np.mean((y >= lower) & (y <= upper)))


def regression_summary(
    actual: np.ndarray,
    predicted_mean: np.ndarray,
    predictive_sd: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    mean = np.asarray(predicted_mean, dtype=float)
    sd = np.clip(np.asarray(predictive_sd, dtype=float), 1e-6, None)
    residuals = mean - y
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    mae = float(np.mean(np.abs(residuals)))
    total_sum_squares = float(np.sum(np.square(y - np.mean(y))))
    residual_sum_squares = float(np.sum(np.square(residuals)))
    r2 = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0.0 else 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mean_log_predictive_density": gaussian_mean_log_predictive_density(y, mean, sd),
        "coverage_90": gaussian_interval_coverage(y, mean, sd, nominal=0.9),
        "predictive_sd_mean": float(np.mean(sd)),
    }


def build_metric_row(
    *,
    dataset_name: str,
    representation: str,
    model_name: str,
    method: str,
    split_name: str,
    mol_ids: tuple[str, ...],
    actual: np.ndarray,
    predicted_mean: np.ndarray,
    predictive_sd: np.ndarray,
    runtime_seconds: float,
    feature_count: int,
    n_train: int,
    split_scheme: str,
    source_row_count: int,
    used_row_count: int,
    seed: int | str,
    draw_count: int | None,
    student_df: float | None = None,
    parameter_count: int | None = None,
    extra_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    metrics = regression_summary(actual, predicted_mean, predictive_sd)
    row: dict[str, Any] = {
        "dataset": dataset_name,
        "representation": representation,
        "model": model_name,
        "method": method,
        "split": split_name,
        "split_scheme": split_scheme,
        "source_row_count": source_row_count,
        "used_row_count": used_row_count,
        "n_train": n_train,
        "n_eval": int(len(mol_ids)),
        "feature_count": feature_count,
        "draw_count": draw_count,
        "runtime_seconds": runtime_seconds,
        "seed": seed,
        "parameter_count": parameter_count,
        "student_df": student_df,
    }
    row.update(metrics)
    if extra_metrics:
        row.update(extra_metrics)
    return row


def build_prediction_rows(
    *,
    dataset_name: str,
    representation: str,
    model_name: str,
    method: str,
    split_name: str,
    mol_ids: tuple[str, ...],
    actual: np.ndarray,
    predicted_mean: np.ndarray,
    predictive_sd: np.ndarray,
    seed: int | str,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    arrays = extra_columns or {}
    rows: list[dict[str, Any]] = []
    for index, (mol_id, y, mean, sd) in enumerate(zip(mol_ids, actual, predicted_mean, predictive_sd, strict=True)):
        row: dict[str, Any] = {
            "dataset": dataset_name,
            "representation": representation,
            "model": model_name,
            "method": method,
            "split": split_name,
            "mol_id": mol_id,
            "actual": float(y),
            "predicted_mean": float(mean),
            "predictive_sd": float(sd),
            "seed": seed,
        }
        for name, values in arrays.items():
            row[name] = float(values[index])
        rows.append(row)
    return rows


def build_calibration_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    if predictions.empty:
        return []
    required = {"dataset", "representation", "model", "method", "split", "actual", "predicted_mean", "predictive_sd"}
    if not required.issubset(predictions.columns):
        return []
    rows: list[dict[str, Any]] = []
    for key, frame in predictions.groupby(["dataset", "representation", "model", "method", "split"], sort=True):
        actual = frame["actual"].to_numpy(dtype=float)
        predicted_mean = frame["predicted_mean"].to_numpy(dtype=float)
        predictive_sd = frame["predictive_sd"].to_numpy(dtype=float)
        seed_value = frame["seed"].iloc[0] if "seed" in frame.columns else ""
        for nominal in CALIBRATION_LEVELS:
            rows.append(
                {
                    "dataset": key[0],
                    "representation": key[1],
                    "model": key[2],
                    "method": key[3],
                    "split": key[4],
                    "seed": seed_value,
                    "nominal_coverage": nominal,
                    "empirical_coverage": gaussian_interval_coverage(
                        actual,
                        predicted_mean,
                        predictive_sd,
                        nominal=nominal,
                    ),
                }
            )
    return rows


def aggregate_seed_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "seed" not in metrics.columns:
        return pd.DataFrame()
    group_columns = [
        "dataset",
        "representation",
        "model",
        "method",
        "split",
        "split_scheme",
        "source_row_count",
        "used_row_count",
        "n_train",
        "n_eval",
        "feature_count",
        "draw_count",
        "student_df",
        "parameter_count",
    ]
    numeric_targets = [
        "runtime_seconds",
        "rmse",
        "mae",
        "r2",
        "mean_log_predictive_density",
        "coverage_90",
        "predictive_sd_mean",
    ]
    subset = metrics.loc[metrics["seed"].astype(str) != "aggregate"].copy()
    if subset.empty:
        return pd.DataFrame()
    aggregated = subset.groupby(group_columns, dropna=False, as_index=False)[numeric_targets].agg(["mean", "std"])
    aggregated.columns = [
        "_".join(part for part in column if part).rstrip("_")
        for column in aggregated.columns.to_flat_index()
    ]
    aggregated = aggregated.rename(
        columns={
            "dataset_": "dataset",
            "representation_": "representation",
            "model_": "model",
            "method_": "method",
            "split_": "split",
            "split_scheme_": "split_scheme",
            "source_row_count_": "source_row_count",
            "used_row_count_": "used_row_count",
            "n_train_": "n_train",
            "n_eval_": "n_eval",
            "feature_count_": "feature_count",
            "draw_count_": "draw_count",
            "student_df_": "student_df",
            "parameter_count_": "parameter_count",
        }
    )
    aggregated["seed"] = "aggregate"
    return aggregated


def _z_value_for_nominal(nominal: float) -> float:
    if nominal <= 0.0 or nominal >= 1.0:
        raise ValueError("Nominal coverage must be between 0 and 1")
    # Two-sided interval under a standard Gaussian.
    tail = 0.5 + nominal / 2.0
    return float(norm.ppf(tail))
