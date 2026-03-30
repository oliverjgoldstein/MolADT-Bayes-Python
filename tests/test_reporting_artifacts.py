from __future__ import annotations

import pandas as pd
import pytest

from scripts.report_graphs import (
    write_review_rmse_overview,
    write_timing_stage_overview,
)
from scripts.run_all import _build_generalization_frame, _build_simple_review_frame, _remove_legacy_report_artifacts, _write_results_csv


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


def test_build_simple_review_frame_keeps_qm9_as_partial_context() -> None:
    generalization = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "smiles",
                "model": "m1",
                "method": "fast",
                "train_rmse": 1.3,
                "test_rmse": 1.5,
                "test_minus_train_rmse": 0.2,
            },
            {
                "dataset": "qm9",
                "representation": "sdf",
                "model": "m2",
                "method": "slow",
                "train_rmse": 0.8,
                "test_rmse": 0.9,
                "test_minus_train_rmse": 0.1,
            },
        ]
    )

    review = _build_simple_review_frame(generalization)

    freesolv = review.loc[review["dataset"] == "freesolv"].iloc[0]
    qm9 = review.loc[review["dataset"] == "qm9"].iloc[0]
    assert freesolv["literature_rmse"] == pytest.approx(1.15)
    assert "MoleculeNet" in freesolv["literature_display"]
    assert pd.isna(qm9["literature_rmse"])
    assert "Gilmer supplementary Table 2" in qm9["literature_display"]
    assert qm9["directly_comparable"] == "partial"


def test_review_pack_graphs_write_svg_files(tmp_path) -> None:
    review = pd.DataFrame(
        [
            {
                "task": "freesolv / smiles",
                "dataset": "freesolv",
                "representation": "smiles",
                "model": "m1",
                "method": "fast",
                "train_rmse": 1.2,
                "test_rmse": 1.4,
                "test_minus_train_rmse": 0.2,
                "literature_display": "MoleculeNet MPNN RMSE 1.15",
                "literature_rmse": 1.15,
                "note": "Split differs",
            }
        ]
    )
    timing = pd.DataFrame(
        [
            {
                "stage": "raw_file_read",
                "molecule_count": 100,
                "success_count": 100,
                "failure_count": 0,
                "total_runtime_seconds": 0.5,
                "molecules_per_second": 200.0,
            },
            {
                "stage": "moladt_parse_render",
                "molecule_count": 100,
                "success_count": 60,
                "failure_count": 40,
                "total_runtime_seconds": 1.0,
                "molecules_per_second": 100.0,
            },
        ]
    )

    rmse_path = tmp_path / "rmse_vs_literature_context.svg"
    timing_path = tmp_path / "timing_overview.svg"
    write_review_rmse_overview(review, rmse_path)
    write_timing_stage_overview(timing, timing_path)

    rmse_svg = rmse_path.read_text(encoding="utf-8")
    timing_svg = timing_path.read_text(encoding="utf-8")
    assert "<svg" in rmse_svg
    assert "freesolv / smiles" in rmse_svg
    assert "MoleculeNet MPNN RMSE 1.15" in rmse_svg
    assert "<svg" in timing_svg
    assert "raw_file_read" in timing_svg
    assert "moladt_parse_render" in timing_svg


def test_results_csv_combines_predictive_and_timing_rows(tmp_path, monkeypatch) -> None:
    import scripts.run_all as run_all

    monkeypatch.setattr(run_all, "RESULTS_DIR", tmp_path)
    review = pd.DataFrame(
        [
            {
                "task": "freesolv / smiles",
                "dataset": "freesolv",
                "representation": "smiles",
                "model": "m1",
                "method": "fast",
                "train_rmse": 1.2,
                "test_rmse": 1.4,
                "test_minus_train_rmse": 0.2,
                "train_mae": 0.9,
                "test_mae": 1.0,
                "train_r2": 0.8,
                "test_r2": 0.7,
                "fit_runtime_seconds": 2.5,
                "literature_display": "MoleculeNet MPNN RMSE 1.15",
                "literature_rmse": 1.15,
                "literature_metric": "RMSE",
                "directly_comparable": "partial",
                "note": "Split differs",
                "source": "https://example.com",
            }
        ]
    )
    timing = pd.DataFrame(
        [
            {
                "stage": "raw_file_read",
                "molecule_count": 100,
                "success_count": 100,
                "failure_count": 0,
                "total_runtime_seconds": 0.5,
                "molecules_per_second": 200.0,
                "median_latency_us": 10.0,
                "p95_latency_us": 20.0,
                "peak_rss_mb": 123.0,
            }
        ]
    )

    _write_results_csv(command="benchmark", generated_at="20260330_170000", review_frame=review, timing=timing)

    frame = pd.read_csv(tmp_path / "results.csv")
    assert set(frame["row_type"]) == {"predictive_summary", "timing_stage"}
    predictive = frame.loc[frame["row_type"] == "predictive_summary"].iloc[0]
    timing_row = frame.loc[frame["row_type"] == "timing_stage"].iloc[0]
    assert predictive["task"] == "freesolv / smiles"
    assert predictive["train_rmse"] == pytest.approx(1.2)
    assert timing_row["stage"] == "raw_file_read"
    assert timing_row["molecules_per_second"] == pytest.approx(200.0)


def test_remove_legacy_report_artifacts_cleans_old_top_level_files(tmp_path, monkeypatch) -> None:
    import scripts.run_all as run_all

    monkeypatch.setattr(run_all, "RESULTS_DIR", tmp_path)
    for name in (
        "summary.md",
        "model_report.md",
        "generalization_report.md",
        "literature_context.md",
        "zinc_timing.md",
        "predictive_metrics.csv",
        "predictions.csv",
        "model_coefficients.csv",
        "generalization_metrics.csv",
        "zinc_timing.csv",
        "split_rmse_overview.svg",
        "predicted_vs_actual_overview.svg",
    ):
        (tmp_path / name).write_text("legacy\n", encoding="utf-8")
    (tmp_path / "stan_output").mkdir()
    (tmp_path / "review_20260330_170000").mkdir()
    (tmp_path / "results.csv").write_text("keep\n", encoding="utf-8")

    _remove_legacy_report_artifacts()

    assert (tmp_path / "results.csv").exists()
    assert not (tmp_path / "summary.md").exists()
    assert not (tmp_path / "stan_output").exists()
    assert not (tmp_path / "review_20260330_170000").exists()
