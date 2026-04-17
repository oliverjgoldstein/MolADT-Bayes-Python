from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.literature_baselines import literature_baselines_frame
from scripts.report_graphs import write_moleculenet_comparison_overviews, write_timing_stage_overview
from scripts.run_all import (
    _attach_moleculenet_uncertainty,
    _build_freesolv_bayesian_artifact,
    _build_generalization_frame,
    _build_moleculenet_comparison_frame,
    _build_simple_review_frame,
    _remove_legacy_report_artifacts,
    _selected_prediction_rows,
    _write_results_csv,
)


def test_literature_baselines_frame_keeps_only_moleculenet_rows() -> None:
    baselines = literature_baselines_frame().sort_values(["dataset"]).reset_index(drop=True)

    assert list(baselines["dataset"]) == ["freesolv", "qm9"]
    assert list(baselines["model_name"]) == ["MPNN", "DTNN"]
    assert list(baselines["metric_name"]) == ["RMSE", "MAE"]
    assert baselines.loc[0, "metric_value"] == pytest.approx(1.15)
    assert baselines.loc[1, "metric_value"] == pytest.approx(2.35)


def test_build_generalization_frame_uses_validation_selection_and_dataset_primary_metric() -> None:
    metrics = pd.DataFrame(
        [
            {"dataset": "freesolv", "representation": "moladt", "model": "bayes_linear_student_t", "method": "sample", "split": "train", "n_eval": 8, "rmse": 1.00, "mae": 0.80, "r2": 0.80, "mean_log_predictive_density": -1.0, "runtime_seconds": 5.0, "split_scheme": "subset:fractional_0.8/0.1/0.1", "source_row_count": 20, "used_row_count": 20},
            {"dataset": "freesolv", "representation": "moladt", "model": "bayes_linear_student_t", "method": "sample", "split": "valid", "n_eval": 1, "rmse": 1.10, "mae": 0.85, "r2": 0.78, "mean_log_predictive_density": -1.1, "runtime_seconds": 5.0, "split_scheme": "subset:fractional_0.8/0.1/0.1", "source_row_count": 20, "used_row_count": 20},
            {"dataset": "freesolv", "representation": "moladt", "model": "bayes_linear_student_t", "method": "sample", "split": "test", "n_eval": 1, "rmse": 1.20, "mae": 0.90, "r2": 0.75, "mean_log_predictive_density": -1.2, "runtime_seconds": 5.0, "split_scheme": "subset:fractional_0.8/0.1/0.1", "source_row_count": 20, "used_row_count": 20},
            {"dataset": "freesolv", "representation": "moladt", "model": "bayes_hierarchical_shrinkage", "method": "optimize", "split": "train", "n_eval": 8, "rmse": 0.95, "mae": 0.78, "r2": 0.81, "mean_log_predictive_density": -0.95, "runtime_seconds": 2.0, "split_scheme": "subset:fractional_0.8/0.1/0.1", "source_row_count": 20, "used_row_count": 20},
            {"dataset": "freesolv", "representation": "moladt", "model": "bayes_hierarchical_shrinkage", "method": "optimize", "split": "valid", "n_eval": 1, "rmse": 1.00, "mae": 0.82, "r2": 0.79, "mean_log_predictive_density": -1.0, "runtime_seconds": 2.0, "split_scheme": "subset:fractional_0.8/0.1/0.1", "source_row_count": 20, "used_row_count": 20},
            {"dataset": "freesolv", "representation": "moladt", "model": "bayes_hierarchical_shrinkage", "method": "optimize", "split": "test", "n_eval": 1, "rmse": 1.10, "mae": 0.88, "r2": 0.77, "mean_log_predictive_density": -1.1, "runtime_seconds": 2.0, "split_scheme": "subset:fractional_0.8/0.1/0.1", "source_row_count": 20, "used_row_count": 20},
            {"dataset": "qm9", "representation": "moladt", "model": "bayes_linear_student_t", "method": "sample", "split": "train", "n_eval": 8, "rmse": 0.060, "mae": 0.049, "r2": 0.90, "mean_log_predictive_density": -0.6, "runtime_seconds": 6.0, "split_scheme": "paper:110462/10000/10000", "source_row_count": 133885, "used_row_count": 130462},
            {"dataset": "qm9", "representation": "moladt", "model": "bayes_linear_student_t", "method": "sample", "split": "valid", "n_eval": 1, "rmse": 0.061, "mae": 0.048, "r2": 0.89, "mean_log_predictive_density": -0.61, "runtime_seconds": 6.0, "split_scheme": "paper:110462/10000/10000", "source_row_count": 133885, "used_row_count": 130462},
            {"dataset": "qm9", "representation": "moladt", "model": "bayes_linear_student_t", "method": "sample", "split": "test", "n_eval": 1, "rmse": 0.062, "mae": 0.049, "r2": 0.88, "mean_log_predictive_density": -0.62, "runtime_seconds": 6.0, "split_scheme": "paper:110462/10000/10000", "source_row_count": 133885, "used_row_count": 130462},
            {"dataset": "qm9", "representation": "moladt_featurized", "model": "bayes_linear_student_t", "method": "optimize", "split": "train", "n_eval": 8, "rmse": 0.050, "mae": 0.044, "r2": 0.91, "mean_log_predictive_density": -0.5, "runtime_seconds": 3.0, "split_scheme": "paper:110462/10000/10000", "source_row_count": 133885, "used_row_count": 130462},
            {"dataset": "qm9", "representation": "moladt_featurized", "model": "bayes_linear_student_t", "method": "optimize", "split": "valid", "n_eval": 1, "rmse": 0.051, "mae": 0.040, "r2": 0.90, "mean_log_predictive_density": -0.51, "runtime_seconds": 3.0, "split_scheme": "paper:110462/10000/10000", "source_row_count": 133885, "used_row_count": 130462},
            {"dataset": "qm9", "representation": "moladt_featurized", "model": "bayes_linear_student_t", "method": "optimize", "split": "test", "n_eval": 1, "rmse": 0.052, "mae": 0.046, "r2": 0.89, "mean_log_predictive_density": -0.52, "runtime_seconds": 3.0, "split_scheme": "paper:110462/10000/10000", "source_row_count": 133885, "used_row_count": 130462},
        ]
    )

    generalization = _build_generalization_frame(metrics).sort_values(["dataset"]).reset_index(drop=True)

    freesolv = generalization.loc[generalization["dataset"] == "freesolv"].iloc[0]
    qm9 = generalization.loc[
        (generalization["dataset"] == "qm9") & (generalization["representation"] == "moladt_featurized")
    ].iloc[0]
    assert freesolv["method"] == "optimize"
    assert freesolv["test_rmse"] == pytest.approx(1.10)
    assert qm9["method"] == "optimize"
    assert qm9["test_mae"] == pytest.approx(0.046)
    assert qm9["test_rmse"] == pytest.approx(0.052)


def test_build_simple_review_frame_attaches_moleculenet_context() -> None:
    generalization = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "moladt",
                "model": "bayes_hierarchical_shrinkage",
                "method": "optimize",
                "split_scheme": "subset:fractional_0.8/0.1/0.1",
                "source_row_count": 642,
                "used_row_count": 642,
                "train_n_eval": 512,
                "valid_n_eval": 64,
                "test_n_eval": 66,
                "train_rmse": 1.00,
                "valid_rmse": 1.05,
                "test_rmse": 1.10,
                "test_minus_train_rmse": 0.10,
                "train_mae": 0.80,
                "valid_mae": 0.84,
                "test_mae": 0.88,
                "train_r2": 0.80,
                "valid_r2": 0.79,
                "test_r2": 0.78,
                "fit_runtime_seconds": 2.0,
            },
            {
                "dataset": "qm9",
                "representation": "moladt_featurized",
                "model": "bayes_linear_student_t",
                "method": "optimize",
                "split_scheme": "paper:110462/10000/10000",
                "source_row_count": 133885,
                "used_row_count": 130462,
                "train_n_eval": 110462,
                "valid_n_eval": 10000,
                "test_n_eval": 10000,
                "train_rmse": 0.060,
                "valid_rmse": 0.061,
                "test_rmse": 0.062,
                "test_minus_train_rmse": 0.002,
                "train_mae": 0.040,
                "valid_mae": 0.041,
                "test_mae": 0.042,
                "train_r2": 0.90,
                "valid_r2": 0.89,
                "test_r2": 0.88,
                "fit_runtime_seconds": 6.0,
            },
        ]
    )

    review = _build_simple_review_frame(generalization, baselines_frame=literature_baselines_frame())
    freesolv = review.loc[review["dataset"] == "freesolv"].iloc[0]
    qm9 = review.loc[review["dataset"] == "qm9"].iloc[0]

    assert freesolv["local_metric_name"] == "RMSE"
    assert freesolv["local_metric_value"] == pytest.approx(1.10)
    assert freesolv["train_metric_value"] == pytest.approx(1.00)
    assert freesolv["valid_metric_value"] == pytest.approx(1.05)
    assert freesolv["test_metric_value"] == pytest.approx(1.10)
    assert freesolv["selection_split"] == "valid"
    assert freesolv["paper_model_name"] == "MPNN"
    assert freesolv["paper_metric_name"] == "RMSE"
    assert freesolv["paper_metric_value"] == pytest.approx(1.15)
    assert "MoleculeNet Table 3" in freesolv["note"]

    assert qm9["local_metric_name"] == "MAE"
    assert qm9["representation"] == "moladt_featurized"
    assert qm9["method"] == "optimize"
    assert qm9["local_metric_value"] == pytest.approx(0.042)
    assert qm9["train_metric_value"] == pytest.approx(0.040)
    assert qm9["valid_metric_value"] == pytest.approx(0.041)
    assert qm9["test_metric_value"] == pytest.approx(0.042)
    assert qm9["selection_split"] == "valid"
    assert qm9["paper_model_name"] == "DTNN"
    assert qm9["paper_metric_name"] == "MAE"
    assert qm9["paper_metric_value"] == pytest.approx(2.35)
    assert "MoleculeNet" in qm9["note"]


def test_build_simple_review_frame_prefers_best_validation_representation_per_dataset() -> None:
    generalization = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "moladt",
                "model": "bayes_hierarchical_shrinkage",
                "method": "laplace",
                "split_scheme": "fractional:0.800/0.100/0.100",
                "source_row_count": 637,
                "used_row_count": 637,
                "train_n_eval": 509,
                "valid_n_eval": 63,
                "test_n_eval": 65,
                "train_rmse": 0.98,
                "valid_rmse": 1.31,
                "test_rmse": 1.22,
                "test_minus_train_rmse": 0.24,
                "train_mae": 0.74,
                "valid_mae": 0.99,
                "test_mae": 0.95,
                "train_r2": 0.82,
                "valid_r2": 0.78,
                "test_r2": 0.76,
                "fit_runtime_seconds": 12.0,
            },
            {
                "dataset": "freesolv",
                "representation": "moladt_featurized",
                "model": "bayes_gp_rbf_screened",
                "method": "laplace",
                "split_scheme": "fractional:0.800/0.100/0.100",
                "source_row_count": 642,
                "used_row_count": 642,
                "train_n_eval": 513,
                "valid_n_eval": 64,
                "test_n_eval": 65,
                "train_rmse": 0.51,
                "valid_rmse": 1.12,
                "test_rmse": 0.74,
                "test_minus_train_rmse": 0.23,
                "train_mae": 0.33,
                "valid_mae": 0.77,
                "test_mae": 0.53,
                "train_r2": 0.98,
                "valid_r2": 0.94,
                "test_r2": 0.94,
                "fit_runtime_seconds": 20.0,
            },
        ]
    )

    review = _build_simple_review_frame(generalization, baselines_frame=literature_baselines_frame())

    assert len(review) == 1
    freesolv = review.iloc[0]
    assert freesolv["representation"] == "moladt_featurized"
    assert freesolv["model"] == "bayes_gp_rbf_screened"
    assert freesolv["method"] == "laplace"
    assert freesolv["valid_metric_value"] == pytest.approx(1.12)
    assert freesolv["test_metric_value"] == pytest.approx(0.74)


def test_moleculenet_comparison_graphs_write_expected_svg_files(tmp_path) -> None:
    review = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "dataset_label": "FreeSolv",
                "representation": "moladt_featurized",
                "local_metric_name": "RMSE",
                "train_metric_value": 1.00,
                "valid_metric_value": 1.05,
                "test_metric_value": 1.10,
                "local_metric_value": 1.10,
                "model": "bayes_hierarchical_shrinkage",
                "method": "optimize",
                "paper_metric_value": 1.15,
                "selection_split": "valid",
                "paper_model_name": "MPNN",
                "paper_source_title": "MoleculeNet: a benchmark for molecular machine learning",
                "note": "Local split differs from the paper split.",
            },
            {
                "dataset": "qm9",
                "dataset_label": "QM9",
                "representation": "moladt_featurized",
                "local_metric_name": "MAE",
                "train_metric_value": 0.040,
                "valid_metric_value": 0.041,
                "test_metric_value": 0.042,
                "local_metric_value": 0.042,
                "model": "bayes_linear_student_t",
                "method": "optimize",
                "paper_metric_value": 2.35,
                "selection_split": "valid",
                "paper_model_name": "DTNN",
                "paper_source_title": "MoleculeNet: a benchmark for molecular machine learning",
                "note": "Local split differs from the paper split.",
            },
        ]
    )

    comparison = _build_moleculenet_comparison_frame(review)
    write_moleculenet_comparison_overviews(comparison, tmp_path)

    freesolv_svg = (tmp_path / "freesolv_rmse_vs_moleculenet.svg").read_text(encoding="utf-8")
    qm9_svg = (tmp_path / "qm9_mae_vs_moleculenet.svg").read_text(encoding="utf-8")
    freesolv_caption = (tmp_path / "freesolv_rmse_vs_moleculenet.caption.txt").read_text(encoding="utf-8")
    qm9_caption = (tmp_path / "qm9_mae_vs_moleculenet.caption.txt").read_text(encoding="utf-8")
    assert "FreeSolv: RMSE" in freesolv_svg
    assert "Training" in freesolv_svg
    assert ">Validation</text>" in freesolv_svg
    assert "Test" in freesolv_svg
    assert "Paper" in freesolv_svg
    assert "moladt_featurized" not in freesolv_svg
    assert "MPNN" not in freesolv_svg
    assert "moladt_featurized" in freesolv_caption
    assert "MPNN" in freesolv_caption
    assert "QM9: MAE" in qm9_svg
    assert "Training" in qm9_svg
    assert ">Validation</text>" not in qm9_svg
    assert "Test" in qm9_svg
    assert "Paper" in qm9_svg
    assert "moladt_featurized" not in qm9_svg
    assert "DTNN" not in qm9_svg
    assert "moladt_featurized" in qm9_caption
    assert "DTNN" in qm9_caption


def test_attach_moleculenet_uncertainty_adds_freesolv_intervals() -> None:
    comparison = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "dataset_label": "FreeSolv",
                "representation": "moladt_featurized",
                "metric_name": "RMSE",
                "train_value": 0.510,
                "valid_value": 1.127,
                "test_value": 0.738,
                "local_value": 0.738,
                "paper_value": 1.150,
                "model": "bayes_gp_rbf_screened",
                "method": "laplace",
                "selection_split": "valid",
                "paper_model_name": "MPNN",
                "note": "Local split differs from the paper split.",
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {"dataset": "freesolv", "representation": "moladt_featurized", "model": "bayes_gp_rbf_screened", "method": "laplace", "split": "train", "actual": 0.0, "predicted_mean": 0.0, "predictive_sd": 0.2},
            {"dataset": "freesolv", "representation": "moladt_featurized", "model": "bayes_gp_rbf_screened", "method": "laplace", "split": "train", "actual": 1.0, "predicted_mean": 1.0, "predictive_sd": 0.2},
            {"dataset": "freesolv", "representation": "moladt_featurized", "model": "bayes_gp_rbf_screened", "method": "laplace", "split": "valid", "actual": 0.0, "predicted_mean": 0.2, "predictive_sd": 0.3},
            {"dataset": "freesolv", "representation": "moladt_featurized", "model": "bayes_gp_rbf_screened", "method": "laplace", "split": "valid", "actual": 1.0, "predicted_mean": 0.9, "predictive_sd": 0.3},
            {"dataset": "freesolv", "representation": "moladt_featurized", "model": "bayes_gp_rbf_screened", "method": "laplace", "split": "test", "actual": 0.0, "predicted_mean": 0.1, "predictive_sd": 0.25},
            {"dataset": "freesolv", "representation": "moladt_featurized", "model": "bayes_gp_rbf_screened", "method": "laplace", "split": "test", "actual": 1.0, "predicted_mean": 1.1, "predictive_sd": 0.25},
        ]
    )

    enriched = _attach_moleculenet_uncertainty(comparison, predictions_frame=predictions)

    row = enriched.iloc[0]
    assert row["train_interval_high"] > row["train_interval_low"]
    assert row["valid_interval_high"] > row["valid_interval_low"]
    assert row["test_interval_high"] > row["test_interval_low"]


def test_freesolv_graph_omits_uncertainty_bar_markup(tmp_path) -> None:
    comparison = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "dataset_label": "FreeSolv",
                "representation": "moladt_featurized",
                "metric_name": "RMSE",
                "train_value": 0.510,
                "valid_value": 1.127,
                "test_value": 0.738,
                "local_value": 0.738,
                "paper_value": 1.150,
                "train_interval_low": 0.540,
                "train_interval_high": 0.810,
                "valid_interval_low": 1.180,
                "valid_interval_high": 1.520,
                "test_interval_low": 0.790,
                "test_interval_high": 1.060,
                "model": "bayes_gp_rbf_screened",
                "method": "laplace",
                "selection_split": "valid",
                "paper_model_name": "MPNN",
                "paper_source_title": "MoleculeNet: a benchmark for molecular machine learning",
                "note": "Local split differs from the paper split.",
            }
        ]
    )

    write_moleculenet_comparison_overviews(comparison, tmp_path)
    freesolv_svg = (tmp_path / "freesolv_rmse_vs_moleculenet.svg").read_text(encoding="utf-8")
    freesolv_caption = (tmp_path / "caption.txt").read_text(encoding="utf-8")

    assert 'data-uncertainty="Training"' not in freesolv_svg
    assert 'data-uncertainty="Validation"' not in freesolv_svg
    assert 'data-uncertainty="Test"' not in freesolv_svg
    assert 'data-uncertainty-cap-lower="Training"' not in freesolv_svg
    assert "posterior predictive RMSE" not in freesolv_svg
    assert "Stan fit" not in freesolv_svg
    assert "posterior predictive RMSE" not in freesolv_caption
    assert "Stan fit" not in freesolv_caption


def test_qm9_graph_keeps_training_and_test_values_in_their_own_bars(tmp_path) -> None:
    review = pd.DataFrame(
        [
            {
                "dataset": "qm9",
                "dataset_label": "QM9",
                "representation": "moladt_featurized",
                "local_metric_name": "MAE",
                "train_metric_value": 0.111,
                "valid_metric_value": 0.222,
                "test_metric_value": 0.333,
                "local_metric_value": 0.333,
                "model": "catboost_uncertainty",
                "method": "predictive",
                "paper_metric_value": 2.35,
                "selection_split": "valid",
                "paper_model_name": "DTNN",
                "paper_source_title": "MoleculeNet: a benchmark for molecular machine learning",
                "note": "Local split differs from the paper split.",
            },
        ]
    )

    comparison = _build_moleculenet_comparison_frame(review)

    assert comparison.loc[0, "train_value"] == pytest.approx(0.111)
    assert comparison.loc[0, "test_value"] == pytest.approx(0.333)
    assert comparison.loc[0, "local_value"] == pytest.approx(0.333)

    write_moleculenet_comparison_overviews(comparison, tmp_path)
    qm9_svg = (tmp_path / "qm9_mae_vs_moleculenet.svg").read_text(encoding="utf-8")

    assert ">0.111</text>" in qm9_svg
    assert ">0.333</text>" in qm9_svg
    assert ">0.222</text>" not in qm9_svg
    assert qm9_svg.index(">0.111</text>") < qm9_svg.index(">0.333</text>")


def test_build_freesolv_bayesian_artifact_formats_uncertainty_and_model_text() -> None:
    review = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "moladt_featurized",
                "model": "bayes_gp_rbf_screened",
                "method": "laplace",
                "split_scheme": "fractional:0.800/0.100/0.100",
                "train_n_eval": 513,
                "valid_n_eval": 64,
                "test_n_eval": 65,
            }
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "moladt_featurized",
                "model": "bayes_gp_rbf_screened",
                "method": "laplace",
                "split": "train",
                "n_eval": 513,
                "rmse": 0.510,
                "mae": 0.330,
                "r2": 0.980,
                "predictive_sd_mean": 0.145,
                "coverage_90": 0.912,
                "mean_log_predictive_density": -0.440,
                "draw_count": 2000,
                "runtime_seconds": 77.5,
            },
            {
                "dataset": "freesolv",
                "representation": "moladt_featurized",
                "model": "bayes_gp_rbf_screened",
                "method": "laplace",
                "split": "valid",
                "n_eval": 64,
                "rmse": 1.127,
                "mae": 0.812,
                "r2": 0.950,
                "predictive_sd_mean": 0.201,
                "coverage_90": 0.844,
                "mean_log_predictive_density": -1.102,
                "draw_count": 2000,
                "runtime_seconds": 77.5,
            },
            {
                "dataset": "freesolv",
                "representation": "moladt_featurized",
                "model": "bayes_gp_rbf_screened",
                "method": "laplace",
                "split": "test",
                "n_eval": 65,
                "rmse": 0.738,
                "mae": 0.530,
                "r2": 0.941,
                "predictive_sd_mean": 0.212,
                "coverage_90": 0.877,
                "mean_log_predictive_density": -0.691,
                "draw_count": 2000,
                "runtime_seconds": 77.5,
            },
        ]
    )
    coefficients = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "moladt_featurized",
                "model": "bayes_gp_rbf_screened",
                "method": "laplace",
                "parameter_name": "alpha",
                "posterior_mean": 0.111111,
                "posterior_sd": 0.010000,
                "posterior_p05": 0.095000,
                "posterior_p95": 0.128000,
            },
            {
                "dataset": "freesolv",
                "representation": "moladt_featurized",
                "model": "bayes_gp_rbf_screened",
                "method": "laplace",
                "parameter_name": "signal_scale",
                "posterior_mean": 1.234567,
                "posterior_sd": 0.050000,
                "posterior_p05": 1.160000,
                "posterior_p95": 1.320000,
            },
            {
                "dataset": "freesolv",
                "representation": "moladt_featurized",
                "model": "bayes_gp_rbf_screened",
                "method": "laplace",
                "parameter_name": "lengthscale",
                "posterior_mean": 2.345678,
                "posterior_sd": 0.080000,
                "posterior_p05": 2.220000,
                "posterior_p95": 2.480000,
            },
            {
                "dataset": "freesolv",
                "representation": "moladt_featurized",
                "model": "bayes_gp_rbf_screened",
                "method": "laplace",
                "parameter_name": "sigma",
                "posterior_mean": 0.222222,
                "posterior_sd": 0.020000,
                "posterior_p05": 0.190000,
                "posterior_p95": 0.255000,
            },
        ]
    )

    artifact = _build_freesolv_bayesian_artifact(review, metrics, coefficients)

    assert artifact is not None
    uncertainty = artifact["uncertainty_frame"]
    assert list(uncertainty["split"].astype(str)) == ["train", "test"]
    assert "Train/test predictive uncertainty" in artifact["model_text"]
    assert "Posterior hyperparameters" in artifact["model_text"]
    assert "signal_scale = 1.234567" in artifact["model_text"]
    assert "lengthscale = 2.345678" in artifact["model_text"]
    assert "sigma = 0.222222" in artifact["model_text"]
    assert "coverage_90=0.912" in "\n".join(artifact["summary_lines"])


def test_timing_stage_overview_writes_svg(tmp_path) -> None:
    timing = pd.DataFrame(
        [
            {
                "stage": "smiles_csv_to_string",
                "description": "Read source SMILES rows.",
                "molecule_count": 100,
                "success_count": 100,
                "failure_count": 0,
                "total_runtime_seconds": 0.5,
                "molecules_per_second": 200.0,
                "median_latency_us": 10.0,
                "p95_latency_us": 20.0,
                "peak_rss_mb": 12.0,
            },
            {
                "stage": "smiles_to_json",
                "description": "Parse SMILES strings and serialize JSON payloads.",
                "molecule_count": 100,
                "success_count": 100,
                "failure_count": 0,
                "total_runtime_seconds": 0.7,
                "molecules_per_second": 142.9,
                "median_latency_us": 9.0,
                "p95_latency_us": 16.0,
                "peak_rss_mb": 13.0,
            },
            {
                "stage": "sdf_to_moladt",
                "description": "Parse SDF files into MolADT objects.",
                "molecule_count": 100,
                "success_count": 100,
                "failure_count": 0,
                "total_runtime_seconds": 0.25,
                "molecules_per_second": 400.0,
                "median_latency_us": 5.0,
                "p95_latency_us": 9.0,
                "peak_rss_mb": 10.0,
            },
            {
                "stage": "sdf_to_smiles",
                "description": "Read SDF files and render SMILES strings.",
                "molecule_count": 100,
                "success_count": 100,
                "failure_count": 0,
                "total_runtime_seconds": 0.6,
                "molecules_per_second": 166.7,
                "median_latency_us": 8.0,
                "p95_latency_us": 13.0,
                "peak_rss_mb": 11.0,
            },
            {
                "stage": "moladt_to_json",
                "description": "Serialize MolADT objects to JSON files.",
                "molecule_count": 100,
                "success_count": 100,
                "failure_count": 0,
                "total_runtime_seconds": 0.8,
                "molecules_per_second": 125.0,
                "median_latency_us": 12.0,
                "p95_latency_us": 18.0,
                "peak_rss_mb": 16.0,
            },
            {
                "stage": "json_to_moladt",
                "description": "Decode JSON files back into MolADT objects.",
                "molecule_count": 100,
                "success_count": 100,
                "failure_count": 0,
                "total_runtime_seconds": 1.0,
                "molecules_per_second": 100.0,
                "median_latency_us": 20.0,
                "p95_latency_us": 35.0,
                "peak_rss_mb": 24.0,
            },
        ]
    )

    output = tmp_path / "timing_overview.svg"
    write_timing_stage_overview(timing, output)
    svg = output.read_text(encoding="utf-8")
    caption = (tmp_path / "caption.txt").read_text(encoding="utf-8")
    assert "Timing Throughput" in svg
    assert "SMILES CSV -&gt; string" in svg
    assert "SMILES -&gt; JSON" in svg
    assert "SDF -&gt; MolADT" in svg
    assert "SDF -&gt; SMILES" in svg
    assert "MolADT -&gt; JSON" in svg
    assert "JSON -&gt; MolADT" in svg
    assert "matched local timing corpus" in caption
    assert "SMILES to JSON" in caption
    assert "SDF to SMILES" in caption


def test_results_csv_combines_summary_metric_and_timing_rows(tmp_path, monkeypatch) -> None:
    import scripts.run_all as run_all

    monkeypatch.setattr(run_all, "RESULTS_DIR", tmp_path)
    review = pd.DataFrame(
        [
            {
                "task": "freesolv / moladt",
                "dataset": "freesolv",
                "representation": "moladt",
                "model": "bayes_hierarchical_shrinkage",
                "method": "optimize",
                "split_scheme": "subset:fractional_0.8/0.1/0.1",
                "source_row_count": 642,
                "used_row_count": 642,
                "train_n_eval": 512,
                "valid_n_eval": 64,
                "test_n_eval": 66,
                "train_rmse": 1.00,
                "test_rmse": 1.10,
                "test_minus_train_rmse": 0.10,
                "train_mae": 0.80,
                "test_mae": 0.88,
                "train_r2": 0.80,
                "test_r2": 0.78,
                "fit_runtime_seconds": 2.0,
                "literature_display": "MPNN RMSE 1.150",
                "literature_rmse": 1.15,
                "literature_metric": "RMSE",
                "directly_comparable": "partial",
                "note": "MoleculeNet Table 3 context.",
                "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/",
            }
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "moladt",
                "model": "bayes_hierarchical_shrinkage",
                "method": "optimize",
                "split": "test",
                "split_scheme": "subset:fractional_0.8/0.1/0.1",
                "source_row_count": 642,
                "used_row_count": 642,
                "n_train": 512,
                "n_eval": 66,
                "feature_count": 10,
                "draw_count": 300,
                "runtime_seconds": 2.0,
                "rmse": 1.10,
                "mae": 0.88,
                "r2": 0.78,
                "mean_log_predictive_density": -1.1,
                "coverage_90": 0.8,
                "predictive_sd_mean": 0.3,
                "student_df": 4.0,
            }
        ]
    )
    timing = pd.DataFrame(
        [
            {
                "stage": "smiles_csv_to_string",
                "description": "Read source SMILES rows.",
                "molecule_count": 100,
                "success_count": 100,
                "failure_count": 0,
                "total_runtime_seconds": 0.5,
                "molecules_per_second": 200.0,
                "median_latency_us": 10.0,
                "p95_latency_us": 20.0,
                "peak_rss_mb": 12.0,
            }
        ]
    )

    _write_results_csv(command="benchmark", generated_at="20260409_120000", metrics=metrics, review_frame=review, timing=timing)

    frame = pd.read_csv(tmp_path / "results.csv")
    assert set(frame["row_type"]) == {"predictive_summary", "predictive_metric", "timing_stage"}
    summary_row = frame.loc[frame["row_type"] == "predictive_summary"].iloc[0]
    metric_row = frame.loc[frame["row_type"] == "predictive_metric"].iloc[0]
    timing_row = frame.loc[frame["row_type"] == "timing_stage"].iloc[0]
    assert summary_row["task"] == "freesolv / moladt"
    assert summary_row["literature_context"] == "MPNN RMSE 1.150"
    assert metric_row["representation"] == "moladt"
    assert metric_row["rmse"] == pytest.approx(1.10)
    assert timing_row["stage"] == "smiles_csv_to_string"


def test_remove_legacy_report_artifacts_cleans_old_files(tmp_path, monkeypatch) -> None:
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
        "caption.txt",
        "timing_result_files.txt",
        "split_rmse_overview.svg",
        "inference_sweep_overview.svg",
        "predicted_vs_actual_overview.svg",
        "coverage_calibration.svg",
        "freesolv_rmse_vs_moleculenet.svg",
        "qm9_mae_vs_moleculenet.svg",
    ):
        (tmp_path / name).write_text("legacy\n", encoding="utf-8")
    (tmp_path / "stan_output").mkdir()
    (tmp_path / "results.csv").write_text("keep\n", encoding="utf-8")

    _remove_legacy_report_artifacts()

    assert (tmp_path / "results.csv").exists()
    assert not (tmp_path / "summary.md").exists()
    assert not (tmp_path / "freesolv_rmse_vs_moleculenet.svg").exists()
    assert not (tmp_path / "stan_output").exists()


def test_selected_prediction_rows_matches_generalization_keys() -> None:
    generalization = pd.DataFrame(
        [
            {"dataset": "demo", "representation": "moladt", "model": "m1", "method": "fast"},
        ]
    )
    predictions = pd.DataFrame(
        [
            {"dataset": "demo", "representation": "moladt", "model": "m1", "method": "fast", "split": "test", "mol_id": "a", "actual": 1.0, "predicted_mean": 1.0, "predictive_sd": 0.1},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "fast", "split": "test", "mol_id": "b", "actual": 1.0, "predicted_mean": 1.0, "predictive_sd": 0.1},
        ]
    )

    selected = _selected_prediction_rows(predictions, generalization)

    assert list(selected["representation"]) == ["moladt"]
    assert list(selected["mol_id"]) == ["a"]
