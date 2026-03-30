from __future__ import annotations

import pandas as pd
import pytest

from scripts.report_graphs import write_predicted_vs_actual_overview, write_split_rmse_overview
from scripts.run_all import _build_generalization_frame


def test_build_generalization_frame_selects_lowest_test_rmse_per_representation() -> None:
    metrics = pd.DataFrame(
        [
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "fast", "split": "train", "n_eval": 8, "rmse": 0.8, "mae": 0.6, "r2": 0.9, "mean_log_predictive_density": -1.0, "runtime_seconds": 1.0},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "fast", "split": "valid", "n_eval": 2, "rmse": 1.0, "mae": 0.8, "r2": 0.7, "mean_log_predictive_density": -1.1, "runtime_seconds": 1.0},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "fast", "split": "test", "n_eval": 2, "rmse": 1.2, "mae": 0.9, "r2": 0.6, "mean_log_predictive_density": -1.2, "runtime_seconds": 1.0},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "slow", "split": "train", "n_eval": 8, "rmse": 0.7, "mae": 0.5, "r2": 0.92, "mean_log_predictive_density": -0.9, "runtime_seconds": 2.0},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "slow", "split": "valid", "n_eval": 2, "rmse": 0.9, "mae": 0.7, "r2": 0.75, "mean_log_predictive_density": -1.0, "runtime_seconds": 2.0},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "slow", "split": "test", "n_eval": 2, "rmse": 1.0, "mae": 0.8, "r2": 0.65, "mean_log_predictive_density": -1.1, "runtime_seconds": 2.0},
            {"dataset": "demo", "representation": "sdf", "model": "m2", "method": "fast", "split": "train", "n_eval": 8, "rmse": 0.5, "mae": 0.4, "r2": 0.95, "mean_log_predictive_density": -0.8, "runtime_seconds": 1.5},
            {"dataset": "demo", "representation": "sdf", "model": "m2", "method": "fast", "split": "valid", "n_eval": 2, "rmse": 0.7, "mae": 0.5, "r2": 0.8, "mean_log_predictive_density": -0.9, "runtime_seconds": 1.5},
            {"dataset": "demo", "representation": "sdf", "model": "m2", "method": "fast", "split": "test", "n_eval": 2, "rmse": 0.9, "mae": 0.6, "r2": 0.7, "mean_log_predictive_density": -1.0, "runtime_seconds": 1.5},
        ]
    )

    generalization = _build_generalization_frame(metrics)

    assert list(generalization["representation"]) == ["sdf", "smiles"]
    smiles_row = generalization.loc[generalization["representation"] == "smiles"].iloc[0]
    assert smiles_row["method"] == "slow"
    assert smiles_row["test_rmse"] == 1.0
    assert smiles_row["test_minus_train_rmse"] == pytest.approx(0.3)


def test_report_graphs_write_svg_files(tmp_path) -> None:
    metrics = pd.DataFrame(
        [
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "slow", "split": "train", "rmse": 0.7, "mae": 0.5, "r2": 0.92},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "slow", "split": "valid", "rmse": 0.9, "mae": 0.7, "r2": 0.75},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "slow", "split": "test", "rmse": 1.0, "mae": 0.8, "r2": 0.65},
        ]
    )
    predictions = pd.DataFrame(
        [
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "slow", "split": "train", "actual": 0.0, "predicted_mean": 0.1},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "slow", "split": "valid", "actual": 1.0, "predicted_mean": 0.8},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "slow", "split": "test", "actual": 2.0, "predicted_mean": 1.7},
        ]
    )

    rmse_path = tmp_path / "split_rmse_overview.svg"
    parity_path = tmp_path / "predicted_vs_actual_overview.svg"
    write_split_rmse_overview(metrics, rmse_path)
    write_predicted_vs_actual_overview(predictions, parity_path, max_points_per_split=10)

    rmse_svg = rmse_path.read_text(encoding="utf-8")
    parity_svg = parity_path.read_text(encoding="utf-8")
    assert "<svg" in rmse_svg
    assert "demo / smiles" in rmse_svg
    assert "<svg" in parity_svg
    assert "demo / smiles" in parity_svg
