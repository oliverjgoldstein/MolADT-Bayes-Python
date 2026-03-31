from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.literature_baselines import literature_baselines_frame
from scripts.report_graphs import (
    write_calibration_overview,
    write_inference_sweep_overview,
    write_metric_comparison_overviews,
    write_predicted_vs_actual_overview,
    write_residual_vs_uncertainty_overview,
    write_review_rmse_overview,
    write_timing_stage_overview,
)
from scripts.run_all import (
    _build_metric_comparison_frame,
    _build_generalization_frame,
    _build_literature_comparison_rows,
    _build_simple_review_frame,
    _remove_legacy_report_artifacts,
    _selected_prediction_rows,
    _write_literature_comparison,
    _write_model_folders,
    _write_results_csv,
)


def test_build_generalization_frame_selects_lowest_test_rmse_per_representation() -> None:
    metrics = pd.DataFrame(
        [
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "fast", "split": "train", "n_eval": 8, "rmse": 0.8, "mae": 0.6, "r2": 0.9, "mean_log_predictive_density": -1.0, "runtime_seconds": 1.0, "split_scheme": "subset:fractional_0.8/0.1/0.1", "source_row_count": 20, "used_row_count": 20},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "fast", "split": "valid", "n_eval": 2, "rmse": 1.0, "mae": 0.8, "r2": 0.7, "mean_log_predictive_density": -1.1, "runtime_seconds": 1.0, "split_scheme": "subset:fractional_0.8/0.1/0.1", "source_row_count": 20, "used_row_count": 20},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "fast", "split": "test", "n_eval": 2, "rmse": 1.2, "mae": 0.9, "r2": 0.6, "mean_log_predictive_density": -1.2, "runtime_seconds": 1.0, "split_scheme": "subset:fractional_0.8/0.1/0.1", "source_row_count": 20, "used_row_count": 20},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "slow", "split": "train", "n_eval": 8, "rmse": 0.7, "mae": 0.5, "r2": 0.92, "mean_log_predictive_density": -0.9, "runtime_seconds": 2.0, "split_scheme": "subset:fractional_0.8/0.1/0.1", "source_row_count": 20, "used_row_count": 20},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "slow", "split": "valid", "n_eval": 2, "rmse": 0.9, "mae": 0.7, "r2": 0.75, "mean_log_predictive_density": -1.0, "runtime_seconds": 2.0, "split_scheme": "subset:fractional_0.8/0.1/0.1", "source_row_count": 20, "used_row_count": 20},
            {"dataset": "demo", "representation": "smiles", "model": "m1", "method": "slow", "split": "test", "n_eval": 2, "rmse": 1.0, "mae": 0.8, "r2": 0.65, "mean_log_predictive_density": -1.1, "runtime_seconds": 2.0, "split_scheme": "subset:fractional_0.8/0.1/0.1", "source_row_count": 20, "used_row_count": 20},
            {"dataset": "demo", "representation": "moladt", "model": "m2", "method": "fast", "split": "train", "n_eval": 8, "rmse": 0.5, "mae": 0.4, "r2": 0.95, "mean_log_predictive_density": -0.8, "runtime_seconds": 1.5, "split_scheme": "paper:10/5/5", "source_row_count": 25, "used_row_count": 20},
            {"dataset": "demo", "representation": "moladt", "model": "m2", "method": "fast", "split": "valid", "n_eval": 2, "rmse": 0.7, "mae": 0.5, "r2": 0.8, "mean_log_predictive_density": -0.9, "runtime_seconds": 1.5, "split_scheme": "paper:10/5/5", "source_row_count": 25, "used_row_count": 20},
            {"dataset": "demo", "representation": "moladt", "model": "m2", "method": "fast", "split": "test", "n_eval": 2, "rmse": 0.9, "mae": 0.6, "r2": 0.7, "mean_log_predictive_density": -1.0, "runtime_seconds": 1.5, "split_scheme": "paper:10/5/5", "source_row_count": 25, "used_row_count": 20},
        ]
    )

    generalization = _build_generalization_frame(metrics)

    assert list(generalization["representation"]) == ["moladt", "smiles"]
    smiles_row = generalization.loc[generalization["representation"] == "smiles"].iloc[0]
    moladt_row = generalization.loc[generalization["representation"] == "moladt"].iloc[0]
    assert smiles_row["method"] == "slow"
    assert smiles_row["test_rmse"] == 1.0
    assert smiles_row["test_minus_train_rmse"] == pytest.approx(0.3)
    assert moladt_row["split_scheme"] == "paper:10/5/5"
    assert moladt_row["source_row_count"] == 25
    assert moladt_row["used_row_count"] == 20


def test_build_simple_review_frame_keeps_qm9_as_partial_context() -> None:
    generalization = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "smiles",
                "model": "m1",
                "method": "fast",
                "split_scheme": "subset:fractional_0.8/0.1/0.1",
                "source_row_count": 642,
                "used_row_count": 642,
                "train_n_eval": 512,
                "valid_n_eval": 64,
                "test_n_eval": 66,
                "train_rmse": 1.3,
                "test_rmse": 1.5,
                "test_minus_train_rmse": 0.2,
            },
            {
                "dataset": "qm9",
                "representation": "moladt",
                "model": "m2",
                "method": "slow",
                "split_scheme": "paper:110462/10000/10000",
                "source_row_count": 133885,
                "used_row_count": 130462,
                "train_n_eval": 110462,
                "valid_n_eval": 10000,
                "test_n_eval": 10000,
                "train_rmse": 0.8,
                "test_rmse": 0.9,
                "test_minus_train_rmse": 0.1,
            },
        ]
    )

    review = _build_simple_review_frame(generalization, baselines_frame=literature_baselines_frame())

    freesolv = review.loc[review["dataset"] == "freesolv"].iloc[0]
    qm9 = review.loc[review["dataset"] == "qm9"].iloc[0]
    assert freesolv["literature_rmse"] == pytest.approx(1.15)
    assert "MPNN RMSE 1.150" in freesolv["literature_display"]
    assert pd.isna(qm9["literature_rmse"])
    assert "MPNN MAE ratio 0.300" in qm9["literature_display"]
    assert qm9["directly_comparable"] == "partial"
    assert "Paper-sized split counts are matched locally" in qm9["note"]


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
    metrics = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "smiles",
                "model": "m1",
                "method": "fast",
                "split": "test",
                "split_scheme": "subset:fractional_0.8/0.1/0.1",
                "n_train": 8,
                "n_eval": 2,
                "rmse": 1.4,
                "mae": 1.0,
                "runtime_seconds": 0.5,
                "coverage_90": 0.5,
            },
            {
                "dataset": "freesolv",
                "representation": "moladt",
                "model": "m2",
                "method": "slow",
                "split": "test",
                "split_scheme": "subset:fractional_0.8/0.1/0.1",
                "n_train": 8,
                "n_eval": 2,
                "rmse": 1.1,
                "mae": 0.9,
                "runtime_seconds": 0.9,
                "coverage_90": 0.8,
            },
        ]
    )
    timing = pd.DataFrame(
        [
            {
                "stage": "raw_file_read",
                "description": "Read SMILES strings from disk only.",
                "molecule_count": 100,
                "success_count": 100,
                "failure_count": 0,
                "total_runtime_seconds": 0.5,
                "molecules_per_second": 200.0,
                "median_latency_us": 10.0,
            },
            {
                "stage": "moladt_file_parse",
                "description": "Read each MolADT JSON file and parsed it into the local Molecule ADT using the fast JSON loader when available.",
                "molecule_count": 100,
                "success_count": 100,
                "failure_count": 0,
                "total_runtime_seconds": 1.0,
                "molecules_per_second": 100.0,
                "median_latency_us": 20.0,
            },
        ]
    )

    rmse_path = tmp_path / "rmse_vs_literature_context.svg"
    inference_path = tmp_path / "inference_sweep_overview.svg"
    timing_path = tmp_path / "timing_overview.svg"
    scatter_path = tmp_path / "predicted_vs_actual.svg"
    residual_path = tmp_path / "residual_vs_uncertainty.svg"
    calibration_path = tmp_path / "calibration.svg"
    predictions = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "smiles",
                "model": "m1",
                "method": "fast",
                "split": "test",
                "actual": 1.0,
                "predicted_mean": 1.2,
                "predictive_sd": 0.3,
            },
            {
                "dataset": "freesolv",
                "representation": "smiles",
                "model": "m1",
                "method": "fast",
                "split": "test",
                "actual": 2.0,
                "predicted_mean": 1.8,
                "predictive_sd": 0.2,
            },
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "smiles",
                "model": "m1",
                "method": "fast",
                "split": "test",
                "nominal_coverage": 0.8,
                "empirical_coverage": 0.75,
            },
            {
                "dataset": "freesolv",
                "representation": "smiles",
                "model": "m1",
                "method": "fast",
                "split": "test",
                "nominal_coverage": 0.9,
                "empirical_coverage": 0.85,
            },
        ]
    )
    write_review_rmse_overview(review, rmse_path)
    write_inference_sweep_overview(metrics, inference_path)
    write_timing_stage_overview(timing, timing_path)
    write_predicted_vs_actual_overview(predictions, scatter_path)
    write_residual_vs_uncertainty_overview(predictions, residual_path)
    write_calibration_overview(calibration, calibration_path)

    rmse_svg = rmse_path.read_text(encoding="utf-8")
    inference_svg = inference_path.read_text(encoding="utf-8")
    timing_svg = timing_path.read_text(encoding="utf-8")
    scatter_svg = scatter_path.read_text(encoding="utf-8")
    residual_svg = residual_path.read_text(encoding="utf-8")
    calibration_svg = calibration_path.read_text(encoding="utf-8")
    assert "<svg" in rmse_svg
    assert "freesolv / smiles" in rmse_svg
    assert "Blue=train local RMSE" in rmse_svg
    assert "<svg" in inference_svg
    assert "Inference sweep on local test splits" in inference_svg
    assert "freesolv / moladt" in inference_svg
    assert "<svg" in timing_svg
    assert "raw_file_read" in timing_svg
    assert "MolADT JSON file" in timing_svg
    assert "Predicted vs actual" in scatter_svg
    assert "Residual vs uncertainty" in residual_svg
    assert "Coverage calibration" in calibration_svg


def test_metric_comparison_frame_prefers_shared_catboost_rows_and_attaches_paper_context() -> None:
    metrics = pd.DataFrame(
        [
            {"dataset": "freesolv", "representation": "smiles", "model": "catboost_uncertainty", "method": "native_uncertainty", "split": "test", "rmse": 1.20, "mae": 0.92, "r2": 0.72, "coverage_90": 0.80, "runtime_seconds": 2.0},
            {"dataset": "freesolv", "representation": "moladt", "model": "catboost_uncertainty", "method": "native_uncertainty", "split": "test", "rmse": 1.02, "mae": 0.81, "r2": 0.79, "coverage_90": 0.84, "runtime_seconds": 2.1},
            {"dataset": "freesolv", "representation": "smiles", "model": "bayes_linear_student_t", "method": "sample", "split": "test", "rmse": 1.35, "mae": 1.01, "r2": 0.66, "coverage_90": 0.77, "runtime_seconds": 10.0},
            {"dataset": "freesolv", "representation": "moladt", "model": "bayes_linear_student_t", "method": "sample", "split": "test", "rmse": 1.10, "mae": 0.88, "r2": 0.75, "coverage_90": 0.81, "runtime_seconds": 10.2},
            {"dataset": "qm9", "representation": "smiles", "model": "catboost_uncertainty", "method": "native_uncertainty", "split": "test", "rmse": 0.12, "mae": 0.041, "r2": 0.91, "coverage_90": 0.78, "runtime_seconds": 5.0},
            {"dataset": "qm9", "representation": "moladt", "model": "catboost_uncertainty", "method": "native_uncertainty", "split": "test", "rmse": 0.10, "mae": 0.033, "r2": 0.94, "coverage_90": 0.82, "runtime_seconds": 5.1},
        ]
    )

    comparison = _build_metric_comparison_frame(metrics, baselines_frame=literature_baselines_frame())

    freesolv_rmse = comparison.loc[(comparison["dataset"] == "freesolv") & (comparison["metric_key"] == "rmse")]
    qm9_mae = comparison.loc[(comparison["dataset"] == "qm9") & (comparison["metric_key"] == "mae")]
    assert set(freesolv_rmse["series_key"]) == {"smiles", "moladt", "paper"}
    assert "catboost_uncertainty / native_uncertainty" in freesolv_rmse.iloc[0]["comparison_context"]
    assert set(qm9_mae["series_key"]) == {"smiles", "moladt", "paper"}
    assert "PaiNN" in qm9_mae["series_label"].tolist()


def test_metric_comparison_overviews_write_svg_files(tmp_path) -> None:
    comparison = pd.DataFrame(
        [
            {"dataset": "freesolv", "metric_key": "rmse", "metric_label": "Test RMSE", "series_key": "smiles", "series_label": "smiles", "value": 1.20, "comparison_context": "Local shared family: catboost_uncertainty / native_uncertainty", "paper_source_title": "", "paper_note": ""},
            {"dataset": "freesolv", "metric_key": "rmse", "metric_label": "Test RMSE", "series_key": "moladt", "series_label": "moladt", "value": 1.02, "comparison_context": "Local shared family: catboost_uncertainty / native_uncertainty", "paper_source_title": "", "paper_note": ""},
            {"dataset": "freesolv", "metric_key": "rmse", "metric_label": "Test RMSE", "series_key": "paper", "series_label": "MPNN", "value": 1.15, "comparison_context": "Local shared family: catboost_uncertainty / native_uncertainty", "paper_source_title": "MoleculeNet: a benchmark for molecular machine learning", "paper_note": "Useful external context only."},
            {"dataset": "qm9", "metric_key": "mae", "metric_label": "Test MAE", "series_key": "smiles", "series_label": "smiles", "value": 0.041, "comparison_context": "Local shared family: catboost_uncertainty / native_uncertainty", "paper_source_title": "", "paper_note": ""},
            {"dataset": "qm9", "metric_key": "mae", "metric_label": "Test MAE", "series_key": "moladt", "series_label": "moladt", "value": 0.033, "comparison_context": "Local shared family: catboost_uncertainty / native_uncertainty", "paper_source_title": "", "paper_note": ""},
            {"dataset": "qm9", "metric_key": "mae", "metric_label": "Test MAE", "series_key": "paper", "series_label": "PaiNN", "value": 0.012, "comparison_context": "Local shared family: catboost_uncertainty / native_uncertainty", "paper_source_title": "Equivariant message passing for the prediction of tensorial properties and molecular spectra", "paper_note": "ICML 2021 equivariant message-passing baseline."},
        ]
    )

    write_metric_comparison_overviews(comparison, tmp_path)

    rmse_svg = (tmp_path / "rmse_comparison.svg").read_text(encoding="utf-8")
    mae_svg = (tmp_path / "mae_comparison.svg").read_text(encoding="utf-8")
    assert "Test RMSE comparison" in rmse_svg
    assert "MoleculeNet: a benchmark for molecular machine learning" in rmse_svg
    assert "smiles" in rmse_svg
    assert "moladt" in rmse_svg
    assert "PaiNN" in mae_svg
    assert "Test MAE comparison" in mae_svg


def test_results_csv_combines_summary_metric_and_timing_rows(tmp_path, monkeypatch) -> None:
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
                "split_scheme": "subset:fractional_0.8/0.1/0.1",
                "source_row_count": 642,
                "used_row_count": 642,
                "train_n_eval": 512,
                "valid_n_eval": 64,
                "test_n_eval": 66,
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
    metrics = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "smiles",
                "model": "m1",
                "method": "fast",
                "split": "test",
                "split_scheme": "subset:fractional_0.8/0.1/0.1",
                "source_row_count": 642,
                "used_row_count": 642,
                "n_train": 512,
                "n_eval": 66,
                "feature_count": 10,
                "draw_count": 300,
                "runtime_seconds": 2.5,
                "rmse": 1.4,
                "mae": 1.0,
                "r2": 0.7,
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
                "stage": "raw_file_read",
                "description": "Read SMILES strings from disk only.",
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

    _write_results_csv(command="benchmark", generated_at="20260330_170000", metrics=metrics, review_frame=review, timing=timing)

    frame = pd.read_csv(tmp_path / "results.csv")
    assert set(frame["row_type"]) == {"predictive_summary", "predictive_metric", "timing_stage"}
    predictive = frame.loc[frame["row_type"] == "predictive_summary"].iloc[0]
    metric_row = frame.loc[frame["row_type"] == "predictive_metric"].iloc[0]
    timing_row = frame.loc[frame["row_type"] == "timing_stage"].iloc[0]
    assert predictive["task"] == "freesolv / smiles"
    assert predictive["train_rmse"] == pytest.approx(1.2)
    assert np.isnan(float(predictive["seed"]))
    assert metric_row["split"] == "test"
    assert metric_row["rmse"] == pytest.approx(1.4)
    assert np.isnan(float(metric_row["seed"]))
    assert timing_row["stage"] == "raw_file_read"
    assert timing_row["stage_description"] == "Read SMILES strings from disk only."


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


def test_literature_comparison_rows_separate_noncomparable_context(tmp_path) -> None:
    review = pd.DataFrame(
        [
            {
                "task": "qm9 / moladt",
                "dataset": "qm9",
                "representation": "moladt",
                "model": "catboost_uncertainty",
                "method": "native_uncertainty",
                "split_scheme": "paper:110462/10000/10000",
                "train_rmse": 0.10,
                "test_rmse": 0.12,
                "test_minus_train_rmse": 0.02,
                "train_mae": 0.08,
                "test_mae": 0.09,
                "train_r2": 0.95,
                "test_r2": 0.94,
                "fit_runtime_seconds": 1.5,
                "train_n_eval": 110462,
                "valid_n_eval": 10000,
                "test_n_eval": 10000,
                "source_row_count": 133885,
                "used_row_count": 130462,
                "literature_display": "MPNN MAE ratio 0.300",
                "literature_rmse": pd.NA,
                "literature_metric": "MAE ratio",
                "directly_comparable": "partial",
                "note": "Different metric.",
                "source": "https://example.com",
            }
        ]
    )
    aggregated = pd.DataFrame(
        [
            {
                "dataset": "qm9",
                "representation": "moladt",
                "model": "catboost_uncertainty",
                "method": "native_uncertainty",
                "split": "test",
                "split_scheme": "paper:110462/10000/10000",
                "rmse_mean": 0.12,
                "rmse_std": 0.01,
            }
        ]
    )
    rows = _build_literature_comparison_rows(review, literature_baselines_frame(), aggregated)
    assert rows
    assert all(not row["directly_comparable"] for row in rows)

    import scripts.run_all as run_all

    run_all.RESULTS_DIR = tmp_path
    _write_literature_comparison(review_frame=review, baselines_frame=literature_baselines_frame(), aggregated_metrics=aggregated)
    markdown = (tmp_path / "literature_comparison.md").read_text(encoding="utf-8")
    assert "Directly comparable" in markdown
    assert "None in the current configuration." in markdown
    assert "different split/training protocol" in markdown


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


def test_write_model_folders_creates_per_model_readmes(tmp_path, monkeypatch) -> None:
    import scripts.run_all as run_all

    monkeypatch.setattr(run_all, "RESULTS_DIR", tmp_path)
    metrics = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "smiles",
                "model": "catboost_uncertainty",
                "method": "native_uncertainty",
                "split": "test",
                "rmse": 1.1,
                "mae": 0.9,
                "coverage_90": 0.8,
            },
            {
                "dataset": "freesolv",
                "representation": "moladt",
                "model": "catboost_uncertainty",
                "method": "native_uncertainty",
                "split": "test",
                "rmse": 1.0,
                "mae": 0.8,
                "coverage_90": 0.82,
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "smiles",
                "model": "catboost_uncertainty",
                "method": "native_uncertainty",
                "split": "test",
                "mol_id": "mol_1",
                "actual": 1.0,
                "predicted_mean": 1.1,
                "predictive_sd": 0.2,
            }
        ]
    )
    run_all._write_model_folders(
        metrics_frame=metrics,
        predictions_frame=predictions,
        coefficients_frame=pd.DataFrame(),
        model_artifacts_frame=pd.DataFrame(),
    )

    index_path = tmp_path / "models" / "README.md"
    model_readme = tmp_path / "models" / "catboost_uncertainty" / "README.md"
    assert index_path.exists()
    assert model_readme.exists()
    assert "smiles" in model_readme.read_text(encoding="utf-8")
    assert "MolADT" in model_readme.read_text(encoding="utf-8")
