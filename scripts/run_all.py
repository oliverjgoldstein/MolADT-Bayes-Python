from __future__ import annotations

import argparse
import math
import shutil
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .benchmark_zinc import run_zinc_benchmark
from .common import DEFAULT_SEED, RESULTS_DIR, display_path, ensure_directory, log
from .geometry_runner import GeometryRunConfig
from .literature_baselines import literature_baselines_frame
from .model_errors import OptionalModelDependencyError
from .model_registry import GEOMETRIC_MODEL_REGISTRY, TABULAR_MODEL_REGISTRY
from .process_freesolv import FreeSolvArtifacts, process_freesolv_dataset
from .process_qm9 import QM9Artifacts, process_qm9_dataset
from .predictive_metrics import aggregate_seed_metrics, build_calibration_rows
from .report_graphs import (
    write_calibration_overview,
    write_inference_sweep_overview,
    write_metric_comparison_overviews,
    write_predicted_vs_actual_overview,
    write_residual_vs_uncertainty_overview,
    write_review_rmse_overview,
    write_timing_stage_overview,
)
from .stan_runner import ALL_METHODS, StanRunConfig, run_model_suite, write_stan_data_json
from .tabular_runner import CatBoostRunConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.run_all")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke-test", help="Run the FreeSolv smoke test benchmark")
    _add_common_benchmark_args(smoke)
    smoke.add_argument("--include-moladt-predictive", action="store_true")
    smoke.add_argument("--skip-moladt", dest="skip_moladt", action="store_true")
    smoke.add_argument("--skip-sdf", dest="skip_moladt", action="store_true", help=argparse.SUPPRESS)

    qm9 = subparsers.add_parser("qm9", help="Run the QM9 benchmark")
    _add_common_benchmark_args(qm9)
    qm9.add_argument("--include-moladt-predictive", action="store_true")
    qm9.add_argument("--limit", type=int, default=None, help="Deterministic QM9 source-row limit before feature export; omit for the full local download")
    qm9.add_argument("--split-mode", choices=("subset", "paper"), default="subset")

    zinc = subparsers.add_parser("zinc-timing", help="Run the ZINC timing benchmark")
    zinc.add_argument("--dataset-size", default="250K")
    zinc.add_argument("--dataset-dimension", default="2D")
    zinc.add_argument("--limit", type=int, default=None)
    zinc.add_argument("--include-moladt", action="store_true")
    zinc.add_argument("--force", action="store_true")
    zinc.add_argument("--verbose", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="Run FreeSolv, QM9, and ZINC in order")
    _add_common_benchmark_args(benchmark)
    benchmark.add_argument("--qm9-limit", type=int, default=None)
    benchmark.add_argument("--qm9-split-mode", choices=("subset", "paper"), default="subset")
    benchmark.add_argument("--zinc-dataset-size", default="250K")
    benchmark.add_argument("--zinc-dataset-dimension", default="2D")
    benchmark.add_argument("--zinc-limit", type=int, default=None)
    benchmark.add_argument("--include-moladt-predictive", action="store_true")
    benchmark.add_argument("--skip-moladt", dest="skip_moladt", action="store_true")
    benchmark.add_argument("--skip-sdf", dest="skip_moladt", action="store_true", help=argparse.SUPPRESS)
    benchmark.add_argument("--include-moladt", action="store_true")

    models = subparsers.add_parser("models", help="Run the predictive model suite and write per-model folders")
    _add_common_benchmark_args(models)
    models.add_argument("--qm9-limit", type=int, default=None)
    models.add_argument("--qm9-split-mode", choices=("subset", "paper"), default="subset")
    models.add_argument("--include-moladt-predictive", action="store_true", default=True)
    models.add_argument("--skip-moladt", dest="skip_moladt", action="store_true")
    models.add_argument("--skip-sdf", dest="skip_moladt", action="store_true", help=argparse.SUPPRESS)
    return parser


def _add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--methods",
        default=",".join(ALL_METHODS),
        help="Comma-separated inference methods: sample,variational,pathfinder,optimize,laplace",
    )
    parser.add_argument(
        "--models",
        default="bayes_linear_student_t,bayes_hierarchical_shrinkage",
        help="Comma-separated Stan models to run",
    )
    parser.add_argument("--sample-chains", type=int, default=2)
    parser.add_argument("--sample-warmup", type=int, default=200)
    parser.add_argument("--sample-draws", type=int, default=200)
    parser.add_argument("--approximation-draws", type=int, default=500)
    parser.add_argument("--variational-iterations", type=int, default=5000)
    parser.add_argument("--optimize-iterations", type=int, default=2000)
    parser.add_argument("--pathfinder-paths", type=int, default=4)
    parser.add_argument("--predictive-draws", type=int, default=500)
    parser.add_argument("--extra-models", default="", help="Comma-separated optional model families: catboost_uncertainty,visnet_ensemble,dimenetpp_ensemble")
    parser.add_argument("--paper-mode", action="store_true", help="Use paper-sized splits and repeated seeds for optional extra models")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds or ensemble members for optional extra models in paper mode")
    parser.add_argument("--full-qm9", action="store_true", help="Run QM9 without a local source-row limit")
    parser.add_argument("--geom-model", choices=("visnet", "dimenetpp"), default=None, help="Convenience flag that appends the matching geometry ensemble model")
    parser.add_argument("--skip-geom", action="store_true", help="Disable optional geometry-model runs even if they are listed elsewhere")
    parser.add_argument("--verbose", action="store_true")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    extra_models = _parse_extra_models(args)
    _remove_legacy_report_artifacts()
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    training_curve_rows: list[dict[str, object]] = []
    model_artifact_rows: list[dict[str, object]] = []

    if args.command == "smoke-test":
        if args.verbose:
            log("Starting FreeSolv smoke benchmark")
        artifacts = process_freesolv_dataset(
            seed=args.seed,
            force=args.force,
            include_moladt=not args.skip_moladt or args.include_moladt_predictive,
        )
        _extend_with_property_results(
            artifacts,
            metrics_rows,
            prediction_rows,
            coefficient_rows,
            training_curve_rows,
            model_artifact_rows,
            extra_models,
            args,
        )
    elif args.command == "qm9":
        qm9_limit = None if args.full_qm9 or args.paper_mode else args.limit
        qm9_split_mode = "paper" if args.paper_mode else args.split_mode
        if args.verbose:
            log(f"Starting QM9 benchmark with limit={qm9_limit}, split_mode={qm9_split_mode}")
        artifacts = process_qm9_dataset(seed=args.seed, force=args.force, limit=qm9_limit, split_mode=qm9_split_mode)
        _extend_with_property_results(
            artifacts,
            metrics_rows,
            prediction_rows,
            coefficient_rows,
            training_curve_rows,
            model_artifact_rows,
            extra_models,
            args,
        )
    elif args.command == "zinc-timing":
        run_zinc_benchmark(
            dataset_size=args.dataset_size,
            dataset_dimension=args.dataset_dimension,
            limit=args.limit,
            include_moladt=args.include_moladt,
            force=args.force,
            verbose=args.verbose,
        )
    elif args.command == "benchmark":
        qm9_limit = None if args.full_qm9 or args.paper_mode else args.qm9_limit
        qm9_split_mode = "paper" if args.paper_mode else args.qm9_split_mode
        if args.verbose:
            log(
                "Starting full benchmark "
                f"(qm9_limit={qm9_limit}, qm9_split_mode={qm9_split_mode}, "
                f"zinc_dataset_size={args.zinc_dataset_size}, "
                f"zinc_dataset_dimension={args.zinc_dataset_dimension}, zinc_limit={args.zinc_limit}, "
                f"include_moladt={args.include_moladt}, extra_models={','.join(extra_models) or '(none)'})"
            )
            log(f"Results directory: {display_path(RESULTS_DIR)}")
        freesolv = process_freesolv_dataset(
            seed=args.seed,
            force=args.force,
            include_moladt=not args.skip_moladt or args.include_moladt_predictive,
        )
        _extend_with_property_results(
            freesolv,
            metrics_rows,
            prediction_rows,
            coefficient_rows,
            training_curve_rows,
            model_artifact_rows,
            extra_models,
            args,
        )
        qm9 = process_qm9_dataset(seed=args.seed, force=args.force, limit=qm9_limit, split_mode=qm9_split_mode)
        _extend_with_property_results(
            qm9,
            metrics_rows,
            prediction_rows,
            coefficient_rows,
            training_curve_rows,
            model_artifact_rows,
            extra_models,
            args,
        )
        run_zinc_benchmark(
            dataset_size=args.zinc_dataset_size,
            dataset_dimension=args.zinc_dataset_dimension,
            limit=args.zinc_limit,
            include_moladt=args.include_moladt,
            force=args.force,
            verbose=args.verbose,
        )
    elif args.command == "models":
        qm9_limit = None if args.full_qm9 or args.paper_mode else args.qm9_limit
        qm9_split_mode = "paper" if args.paper_mode else args.qm9_split_mode
        if args.verbose:
            log(
                "Starting predictive model suite "
                f"(qm9_limit={qm9_limit}, qm9_split_mode={qm9_split_mode}, "
                f"extra_models={','.join(extra_models) or '(none)'})"
            )
            log(f"Results directory: {display_path(RESULTS_DIR)}")
        freesolv = process_freesolv_dataset(
            seed=args.seed,
            force=args.force,
            include_moladt=not args.skip_moladt or args.include_moladt_predictive,
        )
        _extend_with_property_results(
            freesolv,
            metrics_rows,
            prediction_rows,
            coefficient_rows,
            training_curve_rows,
            model_artifact_rows,
            extra_models,
            args,
        )
        qm9 = process_qm9_dataset(seed=args.seed, force=args.force, limit=qm9_limit, split_mode=qm9_split_mode)
        _extend_with_property_results(
            qm9,
            metrics_rows,
            prediction_rows,
            coefficient_rows,
            training_curve_rows,
            model_artifact_rows,
            extra_models,
            args,
        )
    else:
        raise RuntimeError(f"Unsupported command {args.command}")

    if metrics_rows:
        details_dir = _details_dir()
        metrics_path = details_dir / "predictive_metrics.csv"
        predictions_path = details_dir / "predictions.csv"
        coefficients_path = details_dir / "model_coefficients.csv"
        training_curves_path = details_dir / "training_curves.csv"
        calibration_path = RESULTS_DIR / "calibration.csv"
        aggregated_metrics_path = details_dir / "aggregated_predictive_metrics.csv"
        model_artifacts_path = details_dir / "model_artifacts.csv"
        baselines_path = RESULTS_DIR / "literature_baselines.csv"
        metrics_frame = pd.DataFrame(metrics_rows)
        predictions_frame = pd.DataFrame(prediction_rows)
        coefficients_frame = pd.DataFrame(coefficient_rows)
        training_curves_frame = pd.DataFrame(training_curve_rows)
        if "seed" not in metrics_frame.columns:
            metrics_frame["seed"] = args.seed
        else:
            metrics_frame["seed"] = metrics_frame["seed"].fillna(args.seed)
        if "seed" not in predictions_frame.columns:
            predictions_frame["seed"] = args.seed
        else:
            predictions_frame["seed"] = predictions_frame["seed"].fillna(args.seed)
        metrics_frame.to_csv(metrics_path, index=False)
        predictions_frame.to_csv(predictions_path, index=False)
        coefficients_frame.to_csv(coefficients_path, index=False)
        training_curves_frame.to_csv(training_curves_path, index=False)
        aggregated_metrics = aggregate_seed_metrics(metrics_frame)
        aggregated_metrics.to_csv(aggregated_metrics_path, index=False)
        calibration_frame = pd.DataFrame(build_calibration_rows(predictions_frame))
        calibration_frame.to_csv(calibration_path, index=False)
        pd.DataFrame(model_artifact_rows).to_csv(model_artifacts_path, index=False)
        baselines_frame = literature_baselines_frame()
        baselines_frame.to_csv(baselines_path, index=False)
        generalization = _write_generalization_artifacts(metrics_frame)
        review_frame = _build_simple_review_frame(generalization, baselines_frame=baselines_frame)
        metric_comparisons = _build_metric_comparison_frame(metrics_frame, baselines_frame=baselines_frame)
        _write_summary_report(review_frame, timing=_load_timing_results() if args.command in {"zinc-timing", "benchmark"} else pd.DataFrame())
        _write_model_report(
            metrics_frame=metrics_frame,
            coefficients_frame=coefficients_frame,
            model_artifacts_frame=pd.DataFrame(model_artifact_rows),
            training_curves_frame=training_curves_frame,
        )
        _write_model_folders(
            metrics_frame=metrics_frame,
            predictions_frame=predictions_frame,
            coefficients_frame=coefficients_frame,
            model_artifacts_frame=pd.DataFrame(model_artifact_rows),
        )
        _write_literature_comparison(
            review_frame=review_frame,
            baselines_frame=baselines_frame,
            aggregated_metrics=aggregated_metrics,
        )
        write_review_rmse_overview(review_frame, RESULTS_DIR / "rmse_train_test_vs_literature.svg")
        write_inference_sweep_overview(metrics_frame, RESULTS_DIR / "inference_sweep_overview.svg")
        figures_dir = ensure_directory(RESULTS_DIR / "figures")
        write_metric_comparison_overviews(metric_comparisons, figures_dir / "metric_comparisons")
        selected_predictions = _selected_prediction_rows(predictions_frame, generalization)
        write_predicted_vs_actual_overview(selected_predictions, figures_dir / "predicted_vs_actual_scatter.svg")
        write_residual_vs_uncertainty_overview(selected_predictions, figures_dir / "residual_vs_uncertainty.svg")
        write_calibration_overview(calibration_frame, figures_dir / "coverage_calibration.svg")
    else:
        metrics_frame = pd.DataFrame()
        generalization = pd.DataFrame()
        review_frame = pd.DataFrame()
        training_curves_frame = pd.DataFrame()
        calibration_frame = pd.DataFrame()
    timing = _load_timing_results() if args.command in {"zinc-timing", "benchmark"} else pd.DataFrame()
    if not timing.empty:
        write_timing_stage_overview(timing, RESULTS_DIR / "timing_overview.svg")
        _write_timing_report(timing)
    _write_results_csv(
        command=args.command,
        generated_at=datetime.now().strftime("%Y%m%d_%H%M%S"),
        metrics=metrics_frame,
        review_frame=review_frame,
        timing=timing,
    )
    return 0


def _extend_with_property_results(
    artifacts: FreeSolvArtifacts | QM9Artifacts,
    metrics_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    coefficient_rows: list[dict[str, object]],
    training_curve_rows: list[dict[str, object]],
    model_artifact_rows: list[dict[str, object]],
    extra_models: tuple[str, ...],
    args: argparse.Namespace,
) -> None:
    methods = tuple(method.strip() for method in args.methods.split(",") if method.strip())
    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    config = StanRunConfig(
        methods=methods,
        seed=args.seed,
        sample_chains=args.sample_chains,
        sample_warmup=args.sample_warmup,
        sample_draws=args.sample_draws,
        approximation_draws=args.approximation_draws,
        variational_iterations=args.variational_iterations,
        optimize_iterations=args.optimize_iterations,
        pathfinder_paths=args.pathfinder_paths,
        predictive_draws=args.predictive_draws,
        verbose=args.verbose,
    )
    tabular_exports = dict(getattr(artifacts, "tabular_exports", {}))
    if not tabular_exports:
        tabular_exports["smiles"] = artifacts.smiles_export
        if getattr(artifacts, "moladt_export", None) is not None:
            tabular_exports["moladt"] = artifacts.moladt_export
    for bundle in tabular_exports.values():
        write_stan_data_json(bundle, student_df=config.student_df)
        for model_name in models:
            rows, predictions, coefficients = run_model_suite(bundle, model_name=model_name, config=config)
            metrics_rows.extend(rows)
            prediction_rows.extend(predictions)
            coefficient_rows.extend(coefficients)
    extra_seeds = _extra_model_seeds(args.seed, args.num_seeds if args.paper_mode else 1)
    for model_name in extra_models:
        if model_name in TABULAR_MODEL_REGISTRY:
            runner = TABULAR_MODEL_REGISTRY[model_name].runner
            try:
                for bundle in tabular_exports.values():
                    rows, predictions, artifact_rows = runner(
                        bundle,
                        config=CatBoostRunConfig(seeds=extra_seeds, verbose=args.verbose),
                    )
                    metrics_rows.extend(rows)
                    prediction_rows.extend(predictions)
                    model_artifact_rows.extend(artifact_rows)
            except OptionalModelDependencyError as exc:
                log(f"Skipping optional model `{model_name}`: {exc}")
        elif model_name in GEOMETRIC_MODEL_REGISTRY:
            runner = GEOMETRIC_MODEL_REGISTRY[model_name].runner
            try:
                for bundle in getattr(artifacts, "geometric_exports", {}).values():
                    rows, predictions, training_curves, artifact_manifest = runner(
                        bundle,
                        config=GeometryRunConfig(model_name=model_name, seeds=extra_seeds, verbose=args.verbose),
                    )
                    metrics_rows.extend(rows)
                    prediction_rows.extend(predictions)
                    training_curve_rows.extend(training_curves)
                    model_artifact_rows.extend(artifact_manifest)
            except OptionalModelDependencyError as exc:
                log(f"Skipping optional model `{model_name}`: {exc}")


def _details_dir() -> Any:
    return ensure_directory(RESULTS_DIR / "details")


def _remove_legacy_report_artifacts() -> None:
    legacy_files = (
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
        "calibration.csv",
        "literature_baselines.csv",
        "literature_comparison.md",
    )
    for name in legacy_files:
        path = RESULTS_DIR / name
        if path.exists():
            path.unlink()
    for path in RESULTS_DIR.iterdir() if RESULTS_DIR.exists() else ():
        if path.is_dir() and (path.name in {"stan_output", "models", "figures", "model_artifacts"} or path.name.startswith("review_")):
            shutil.rmtree(path, ignore_errors=True)


def _select_reviewer_rows(test_rows: pd.DataFrame) -> pd.DataFrame:
    ranked = test_rows.sort_values(["dataset", "representation", "rmse", "mae", "runtime_seconds", "model", "method"])
    return ranked.groupby(["dataset", "representation"], sort=True, as_index=False).head(1)




def _best_test_row(test_rows: pd.DataFrame, *, dataset: str, metric: str) -> dict[str, Any] | None:
    subset = test_rows.loc[test_rows["dataset"] == dataset].copy()
    if subset.empty:
        return None
    ordered = subset.sort_values([metric, "rmse", "mae", "runtime_seconds", "representation", "model", "method"])
    return ordered.iloc[0].to_dict()


def _selected_run_keys(metrics: pd.DataFrame) -> pd.DataFrame:
    test_rows = metrics.loc[metrics["split"] == "test"].copy()
    selected = _select_reviewer_rows(test_rows)
    columns = ["dataset", "representation", "model", "method"]
    if "seed" in selected.columns:
        columns.append("seed")
    return selected.loc[:, columns].drop_duplicates().reset_index(drop=True)


def _build_generalization_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    key_frame = _selected_run_keys(metrics)
    rows: list[dict[str, Any]] = []
    for _, key_row in key_frame.iterrows():
        mask = (
            (metrics["dataset"] == key_row["dataset"])
            & (metrics["representation"] == key_row["representation"])
            & (metrics["model"] == key_row["model"])
            & (metrics["method"] == key_row["method"])
        )
        if "seed" in key_row and "seed" in metrics.columns:
            mask &= metrics["seed"] == key_row["seed"]
        subset = metrics.loc[mask].copy()
        split_rows: dict[str, pd.Series] = {}
        for split_name in ("train", "valid", "test"):
            split_subset = subset.loc[subset["split"] == split_name]
            if split_subset.empty:
                break
            split_rows[split_name] = split_subset.iloc[0]
        if len(split_rows) != 3:
            continue
        train_row = split_rows["train"]
        valid_row = split_rows["valid"]
        test_row = split_rows["test"]
        rows.append(
            {
                "dataset": key_row["dataset"],
                "representation": key_row["representation"],
                "model": key_row["model"],
                "method": key_row["method"],
                "split_scheme": str(test_row.get("split_scheme", "")),
                "source_row_count": int(test_row.get("source_row_count", len(subset))),
                "used_row_count": int(test_row.get("used_row_count", len(subset))),
                "train_n_eval": int(train_row["n_eval"]),
                "valid_n_eval": int(valid_row["n_eval"]),
                "test_n_eval": int(test_row["n_eval"]),
                "train_rmse": float(train_row["rmse"]),
                "valid_rmse": float(valid_row["rmse"]),
                "test_rmse": float(test_row["rmse"]),
                "test_minus_train_rmse": float(test_row["rmse"] - train_row["rmse"]),
                "train_mae": float(train_row["mae"]),
                "valid_mae": float(valid_row["mae"]),
                "test_mae": float(test_row["mae"]),
                "test_minus_train_mae": float(test_row["mae"] - train_row["mae"]),
                "train_r2": float(train_row["r2"]),
                "valid_r2": float(valid_row["r2"]),
                "test_r2": float(test_row["r2"]),
                "train_mean_log_predictive_density": float(train_row["mean_log_predictive_density"]),
                "test_mean_log_predictive_density": float(test_row["mean_log_predictive_density"]),
                "fit_runtime_seconds": float(test_row["runtime_seconds"]),
            }
        )
    return pd.DataFrame(rows)


def _write_generalization_artifacts(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    generalization = _build_generalization_frame(metrics)
    if generalization.empty:
        return pd.DataFrame()
    generalization_metrics_path = _details_dir() / "generalization_metrics.csv"
    generalization.to_csv(generalization_metrics_path, index=False)
    return generalization


def _build_simple_review_frame(generalization: pd.DataFrame, *, baselines_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in generalization.iterrows():
        dataset = str(row["dataset"])
        representation = str(row["representation"])
        baseline = _select_review_baseline(baselines_frame, dataset)
        if baseline is None:
            literature_display = "No external context attached"
            literature_rmse = None
            literature_metric = ""
            source = ""
            note = "Local result only."
        else:
            metric_name = str(baseline.get("metric_name", "")).strip()
            metric_value = baseline.get("metric_value", pd.NA)
            metric_bits = []
            if metric_name:
                metric_bits.append(metric_name)
            if pd.notna(metric_value):
                metric_bits.append(f"{float(metric_value):.3f}")
            literature_display = f"{baseline['model_name']} {(' '.join(metric_bits)).strip()}".strip()
            literature_rmse = float(metric_value) if metric_name == "RMSE" and pd.notna(metric_value) else None
            literature_metric = metric_name
            source = str(baseline.get("source_url", ""))
            note = _context_note_for_row(
                dataset=dataset,
                representation=representation,
                split_scheme=str(row.get("split_scheme", "")),
                train_n_eval=int(row.get("train_n_eval", 0)),
                valid_n_eval=int(row.get("valid_n_eval", 0)),
                test_n_eval=int(row.get("test_n_eval", 0)),
                baseline_note=str(baseline.get("note", "")),
            )
        rows.append(
            {
                "task": f"{dataset} / {representation}",
                "dataset": dataset,
                "representation": representation,
                "model": str(row["model"]),
                "method": str(row["method"]),
                "train_rmse": float(row["train_rmse"]),
                "test_rmse": float(row["test_rmse"]),
                "test_minus_train_rmse": float(row["test_minus_train_rmse"]),
                "train_mae": float(row.get("train_mae", float("nan"))),
                "test_mae": float(row.get("test_mae", float("nan"))),
                "train_r2": float(row.get("train_r2", float("nan"))),
                "test_r2": float(row.get("test_r2", float("nan"))),
                "fit_runtime_seconds": float(row.get("fit_runtime_seconds", float("nan"))),
                "split_scheme": str(row.get("split_scheme", "")),
                "source_row_count": float(row.get("source_row_count", float("nan"))),
                "used_row_count": float(row.get("used_row_count", float("nan"))),
                "train_n_eval": float(row.get("train_n_eval", float("nan"))),
                "valid_n_eval": float(row.get("valid_n_eval", float("nan"))),
                "test_n_eval": float(row.get("test_n_eval", float("nan"))),
                "literature_display": literature_display,
                "literature_rmse": literature_rmse,
                "literature_metric": literature_metric,
                "directly_comparable": "partial",
                "note": note,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def _select_review_baseline(baselines_frame: pd.DataFrame, dataset: str) -> pd.Series | None:
    subset = baselines_frame.loc[baselines_frame["dataset"] == dataset].copy()
    if subset.empty:
        return None
    preferred_rmse = subset.loc[(subset["metric_name"] == "RMSE") & subset["metric_value"].notna()]
    if not preferred_rmse.empty:
        return preferred_rmse.sort_values(["metric_value", "model_name"]).iloc[0]
    preferred_numeric = subset.loc[subset["metric_value"].notna()]
    if not preferred_numeric.empty:
        return preferred_numeric.sort_values(["model_name"]).iloc[0]
    return subset.sort_values(["model_name"]).iloc[0]


def _context_note_for_row(
    *,
    dataset: str,
    representation: str,
    split_scheme: str,
    train_n_eval: int,
    valid_n_eval: int,
    test_n_eval: int,
    baseline_note: str,
) -> str:
    if dataset == "freesolv":
        return _freesolv_context_note(representation, baseline_note=baseline_note)
    if dataset == "qm9":
        return _qm9_review_note(
            split_scheme=split_scheme,
            train_n_eval=train_n_eval,
            valid_n_eval=valid_n_eval,
            test_n_eval=test_n_eval,
            baseline_note=baseline_note,
        )
    return baseline_note or "Local result only."


def _selected_prediction_rows(predictions: pd.DataFrame, generalization: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or generalization.empty:
        return pd.DataFrame()
    selected = generalization.loc[:, ["dataset", "representation", "model", "method"]].drop_duplicates()
    merged = predictions.merge(selected, on=["dataset", "representation", "model", "method"], how="inner")
    return merged.loc[merged["split"] == "test"].copy()


_METRIC_COMPARISON_SPECS: tuple[dict[str, str], ...] = (
    {"metric_key": "rmse", "column": "rmse", "label": "Test RMSE", "baseline_metric": "RMSE"},
    {"metric_key": "mae", "column": "mae", "label": "Test MAE", "baseline_metric": "MAE"},
    {"metric_key": "r2", "column": "r2", "label": "Test R2", "baseline_metric": ""},
    {"metric_key": "coverage_90", "column": "coverage_90", "label": "90% Interval Coverage", "baseline_metric": ""},
)
_LOCAL_COMPARISON_REPRESENTATIONS = ("smiles", "moladt")
_LOCAL_MODEL_PREFERENCE = (
    "catboost_uncertainty",
    "bayes_hierarchical_shrinkage",
    "bayes_linear_student_t",
    "visnet_ensemble",
    "dimenetpp_ensemble",
)


def _build_metric_comparison_frame(metrics: pd.DataFrame, *, baselines_frame: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    local_rows = _selected_local_comparison_rows(metrics)
    if local_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, spec in enumerate(_METRIC_COMPARISON_SPECS):
        metric_key = spec["metric_key"]
        column = spec["column"]
        metric_label = spec["label"]
        baseline_metric = spec["baseline_metric"]
        for dataset, dataset_frame in local_rows.groupby("dataset", sort=True):
            if set(dataset_frame["representation"].astype(str)) != set(_LOCAL_COMPARISON_REPRESENTATIONS):
                continue
            local_context = str(dataset_frame.iloc[0]["comparison_context"])
            for representation in _LOCAL_COMPARISON_REPRESENTATIONS:
                representation_row = dataset_frame.loc[dataset_frame["representation"] == representation]
                if representation_row.empty or pd.isna(representation_row.iloc[0].get(column, pd.NA)):
                    continue
                rows.append(
                    {
                        "dataset": str(dataset),
                        "metric_key": metric_key,
                        "metric_label": metric_label,
                        "series_key": representation,
                        "series_label": representation,
                        "value": float(representation_row.iloc[0][column]),
                        "comparison_context": local_context,
                        "paper_source_title": "",
                        "paper_note": "",
                    }
                )
            baseline = _select_metric_baseline(baselines_frame, dataset=str(dataset), metric_name=baseline_metric)
            if baseline is not None:
                rows.append(
                    {
                        "dataset": str(dataset),
                        "metric_key": metric_key,
                        "metric_label": metric_label,
                        "series_key": "paper",
                        "series_label": str(baseline["model_name"]),
                        "value": float(baseline["metric_value"]),
                        "comparison_context": local_context,
                        "paper_source_title": str(baseline["source_title"]),
                        "paper_note": str(baseline["note"]),
                    }
                )
    return pd.DataFrame(rows)


def _selected_local_comparison_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    test_rows = metrics.loc[
        (metrics["split"] == "test") & metrics["representation"].isin(_LOCAL_COMPARISON_REPRESENTATIONS)
    ].copy()
    if test_rows.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for dataset, dataset_frame in test_rows.groupby("dataset", sort=True):
        shared_rows = _best_shared_local_rows(dataset_frame)
        if shared_rows.empty:
            independent_rows = _select_reviewer_rows(dataset_frame).copy()
            if set(independent_rows["representation"].astype(str)) != set(_LOCAL_COMPARISON_REPRESENTATIONS):
                continue
            independent_rows["comparison_context"] = (
                "Local rows chosen independently because no shared smiles/MolADT model family was present."
            )
            rows.append(independent_rows)
            continue
        rows.append(shared_rows)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _best_shared_local_rows(test_rows: pd.DataFrame) -> pd.DataFrame:
    candidates: list[tuple[int, float, float, pd.DataFrame]] = []
    for (_, _), frame in test_rows.groupby(["model", "method"], sort=False):
        chosen = (
            frame.sort_values(["representation", "rmse", "mae", "runtime_seconds"])
            .groupby("representation", sort=False, as_index=False)
            .head(1)
            .copy()
        )
        if set(chosen["representation"].astype(str)) != set(_LOCAL_COMPARISON_REPRESENTATIONS):
            continue
        model_name = str(chosen.iloc[0]["model"])
        method_name = str(chosen.iloc[0]["method"])
        chosen["comparison_context"] = f"Local shared family: {model_name} / {method_name}"
        candidates.append(
            (
                _LOCAL_MODEL_PREFERENCE.index(model_name) if model_name in _LOCAL_MODEL_PREFERENCE else len(_LOCAL_MODEL_PREFERENCE),
                float(chosen["rmse"].mean()),
                float(chosen["runtime_seconds"].mean()),
                chosen,
            )
        )
    if not candidates:
        return pd.DataFrame()
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3].reset_index(drop=True)


def _select_metric_baseline(baselines_frame: pd.DataFrame, *, dataset: str, metric_name: str) -> pd.Series | None:
    if not metric_name:
        return None
    subset = baselines_frame.loc[
        (baselines_frame["dataset"] == dataset)
        & (baselines_frame["metric_name"] == metric_name)
        & baselines_frame["metric_value"].notna()
    ].copy()
    if subset.empty:
        return None
    return subset.sort_values(["metric_value", "model_name"]).iloc[0]


def _load_timing_results() -> pd.DataFrame:
    timing_path = _details_dir() / "zinc_timing.csv"
    if not timing_path.exists():
        return pd.DataFrame()
    return pd.read_csv(timing_path)


def _write_results_csv(
    *,
    command: str,
    generated_at: str,
    metrics: pd.DataFrame,
    review_frame: pd.DataFrame,
    timing: pd.DataFrame,
) -> None:
    rows: list[dict[str, object]] = []
    for _, row in review_frame.iterrows():
        rows.append(
            {
                "generated_at": generated_at,
                "command": command,
                "row_type": "predictive_summary",
                "task": row["task"],
                "dataset": row["dataset"],
                "representation": row["representation"],
                "model": row["model"],
                "method": row["method"],
                "split": "",
                "split_scheme": row.get("split_scheme", ""),
                "source_row_count": row.get("source_row_count", pd.NA),
                "used_row_count": row.get("used_row_count", pd.NA),
                "n_train": row.get("train_n_eval", pd.NA),
                "n_eval": row.get("test_n_eval", pd.NA),
                "feature_count": pd.NA,
                "draw_count": pd.NA,
                "seed": pd.NA,
                "parameter_count": pd.NA,
                "rmse": pd.NA,
                "mae": pd.NA,
                "r2": pd.NA,
                "mean_log_predictive_density": pd.NA,
                "coverage_90": pd.NA,
                "predictive_sd_mean": pd.NA,
                "student_df": pd.NA,
                "train_rmse": row["train_rmse"],
                "test_rmse": row["test_rmse"],
                "test_minus_train_rmse": row["test_minus_train_rmse"],
                "train_mae": row["train_mae"],
                "test_mae": row["test_mae"],
                "train_r2": row["train_r2"],
                "test_r2": row["test_r2"],
                "fit_runtime_seconds": row["fit_runtime_seconds"],
                "literature_context": row["literature_display"],
                "literature_rmse": row["literature_rmse"],
                "literature_metric": row["literature_metric"],
                "directly_comparable": row["directly_comparable"],
                "note": row["note"],
                "source": row["source"],
                "stage": "",
                "stage_description": "",
                "molecule_count": pd.NA,
                "success_count": pd.NA,
                "failure_count": pd.NA,
                "total_runtime_seconds": pd.NA,
                "molecules_per_second": pd.NA,
                "median_latency_us": pd.NA,
                "p95_latency_us": pd.NA,
                "peak_rss_mb": pd.NA,
            }
        )
    for _, row in metrics.iterrows():
        rows.append(
            {
                "generated_at": generated_at,
                "command": command,
                "row_type": "predictive_metric",
                "task": f"{row['dataset']} / {row['representation']} / {row['split']}",
                "dataset": row["dataset"],
                "representation": row["representation"],
                "model": row["model"],
                "method": row["method"],
                "split": row["split"],
                "split_scheme": row.get("split_scheme", ""),
                "source_row_count": row.get("source_row_count", pd.NA),
                "used_row_count": row.get("used_row_count", pd.NA),
                "n_train": row["n_train"],
                "n_eval": row["n_eval"],
                "feature_count": row["feature_count"],
                "draw_count": row["draw_count"],
                "seed": row.get("seed", pd.NA),
                "parameter_count": row.get("parameter_count", pd.NA),
                "rmse": row["rmse"],
                "mae": row["mae"],
                "r2": row["r2"],
                "mean_log_predictive_density": row["mean_log_predictive_density"],
                "coverage_90": row["coverage_90"],
                "predictive_sd_mean": row["predictive_sd_mean"],
                "student_df": row["student_df"],
                "train_rmse": pd.NA,
                "test_rmse": pd.NA,
                "test_minus_train_rmse": pd.NA,
                "train_mae": pd.NA,
                "test_mae": pd.NA,
                "train_r2": pd.NA,
                "test_r2": pd.NA,
                "fit_runtime_seconds": row["runtime_seconds"],
                "literature_context": "",
                "literature_rmse": pd.NA,
                "literature_metric": "",
                "directly_comparable": "",
                "note": "",
                "source": "",
                "stage": "",
                "stage_description": "",
                "molecule_count": pd.NA,
                "success_count": pd.NA,
                "failure_count": pd.NA,
                "total_runtime_seconds": pd.NA,
                "molecules_per_second": pd.NA,
                "median_latency_us": pd.NA,
                "p95_latency_us": pd.NA,
                "peak_rss_mb": pd.NA,
            }
        )
    for _, row in timing.iterrows():
        rows.append(
            {
                "generated_at": generated_at,
                "command": command,
                "row_type": "timing_stage",
                "task": "",
                "dataset": "",
                "representation": "",
                "model": "",
                "method": "",
                "split": "",
                "split_scheme": "",
                "source_row_count": pd.NA,
                "used_row_count": pd.NA,
                "n_train": pd.NA,
                "n_eval": pd.NA,
                "feature_count": pd.NA,
                "draw_count": pd.NA,
                "seed": pd.NA,
                "parameter_count": pd.NA,
                "rmse": pd.NA,
                "mae": pd.NA,
                "r2": pd.NA,
                "mean_log_predictive_density": pd.NA,
                "coverage_90": pd.NA,
                "predictive_sd_mean": pd.NA,
                "student_df": pd.NA,
                "train_rmse": pd.NA,
                "test_rmse": pd.NA,
                "test_minus_train_rmse": pd.NA,
                "train_mae": pd.NA,
                "test_mae": pd.NA,
                "train_r2": pd.NA,
                "test_r2": pd.NA,
                "fit_runtime_seconds": pd.NA,
                "literature_context": "",
                "literature_rmse": pd.NA,
                "literature_metric": "",
                "directly_comparable": "",
                "note": "",
                "source": "",
                "stage": row["stage"],
                "stage_description": row.get("description", ""),
                "molecule_count": row["molecule_count"],
                "success_count": row["success_count"],
                "failure_count": row["failure_count"],
                "total_runtime_seconds": row["total_runtime_seconds"],
                "molecules_per_second": row["molecules_per_second"],
                "median_latency_us": row["median_latency_us"],
                "p95_latency_us": row["p95_latency_us"],
                "peak_rss_mb": row["peak_rss_mb"],
            }
        )
    frame = pd.DataFrame(rows)
    ensure_directory(RESULTS_DIR)
    frame.to_csv(RESULTS_DIR / "results.csv", index=False)


def _freesolv_context_note(representation: str, *, baseline_note: str = "") -> str:
    if representation == "moladt":
        contextual = (
            "Partial context only: the gray bar is MoleculeNet's neural baseline, while the local MolADT bar "
            "uses descriptor features computed after an RDKit SDF record is parsed into the ADT."
        )
    else:
        contextual = "Partial context only: MoleculeNet uses its own random split; this repo uses the local deterministic split."
    if baseline_note:
        return f"{contextual} {baseline_note}"
    return contextual


def _qm9_context_note(*, n_train: int, n_eval: int) -> str:
    if n_train == 110_462 and n_eval == 10_000:
        return (
            "Gilmer et al. use the same 110462/10000/10000 split sizes, but their model is a 3D MPNN. "
            "This repo run matches the counts with a deterministic local split and reports descriptor-based Bayesian baselines."
        )
    return (
        "Gilmer et al. use the full QM9 split with 110462 train / 10000 validation / 10000 test molecules and a 3D MPNN. "
        "This repo run uses the smaller local subset configuration and reports descriptor-based Bayesian baselines."
    )


def _qm9_review_note(
    *,
    split_scheme: str,
    train_n_eval: int,
    valid_n_eval: int,
    test_n_eval: int,
    baseline_note: str = "",
) -> str:
    if split_scheme.startswith("paper:") or (train_n_eval == 110_462 and valid_n_eval == 10_000 and test_n_eval == 10_000):
        contextual = (
            "Paper-sized split counts are matched locally, but the model family still differs: "
            "these are descriptor-based Bayesian baselines rather than Gilmer's 3D message-passing network."
        )
    else:
        contextual = "Partial context only: the paper reports a different metric and a much larger QM9 split."
    if baseline_note:
        return f"{contextual} {baseline_note}"
    return contextual


def _parse_extra_models(args: argparse.Namespace) -> tuple[str, ...]:
    models = [item.strip() for item in str(getattr(args, "extra_models", "")).split(",") if item.strip()]
    if not models and getattr(args, "command", "") == "models":
        models = ["catboost_uncertainty", "visnet_ensemble"]
    geom_model = getattr(args, "geom_model", None)
    if geom_model is not None:
        models.append(f"{geom_model}_ensemble")
    if getattr(args, "skip_geom", False):
        models = [name for name in models if name not in GEOMETRIC_MODEL_REGISTRY]
    ordered = tuple(dict.fromkeys(models))
    unknown = [name for name in ordered if name not in TABULAR_MODEL_REGISTRY and name not in GEOMETRIC_MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown extra model(s): {', '.join(unknown)}")
    return ordered


def _extra_model_seeds(base_seed: int, count: int) -> tuple[int, ...]:
    return tuple(base_seed + 101 * index for index in range(max(1, count)))


def _write_summary_report(review_frame: pd.DataFrame, *, timing: pd.DataFrame) -> None:
    lines = ["# Benchmark Summary", ""]
    if review_frame.empty:
        lines.append("No predictive runs were recorded.")
    else:
        lines.append("## Best Local Test Rows")
        lines.append("")
        for _, row in review_frame.iterrows():
            lines.append(
                f"- `{row['dataset']}/{row['representation']}`: test RMSE {float(row['test_rmse']):.4f}, "
                f"test MAE {float(row['test_mae']):.4f}, model `{row['model']}` via `{row['method']}`."
            )
        lines.append("")
    if not timing.empty:
        lines.append("## Timing")
        lines.append("")
        for _, row in timing.iterrows():
            lines.append(
                f"- `{row['stage']}`: {float(row['molecules_per_second']):.1f} mol/s, "
                f"failures {int(row['failure_count'])}, runtime {float(row['total_runtime_seconds']):.3f}s."
            )
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_model_report(
    *,
    metrics_frame: pd.DataFrame,
    coefficients_frame: pd.DataFrame,
    model_artifacts_frame: pd.DataFrame,
    training_curves_frame: pd.DataFrame,
) -> None:
    lines = ["# Model Report", ""]
    coefficient_columns = {"dataset_name", "representation", "method", "parameter_type", "parameter_name", "median"}
    if not coefficients_frame.empty and coefficient_columns.issubset(coefficients_frame.columns):
        lines.append("## Stan Coefficients")
        lines.append("")
        top_rows = coefficients_frame.sort_values(["dataset_name", "representation", "method", "parameter_type", "parameter_name"]).head(20)
        for _, row in top_rows.iterrows():
            lines.append(
                f"- `{row['dataset_name']}/{row['representation']}/{row['method']}` `{row['parameter_name']}` "
                f"median {float(row['median']):.4f}."
            )
        lines.append("")
    nonlinear = metrics_frame.loc[metrics_frame["model"].isin(["catboost_uncertainty", "visnet_ensemble", "dimenetpp_ensemble"])]
    if not nonlinear.empty:
        lines.append("## Non-Linear Models")
        lines.append("")
        lines.append("CatBoost and geometry rows do not emit fake coefficient tables. Use the saved artifacts instead.")
        lines.append("")
        if not model_artifacts_frame.empty:
            for _, row in model_artifacts_frame.iterrows():
                lines.append(f"- `{row['model']}` `{row['artifact_type']}`: `{row['path']}`")
        if not training_curves_frame.empty:
            lines.append("")
            lines.append(f"- Training curves recorded: `{display_path(_details_dir() / 'training_curves.csv')}`")
    (RESULTS_DIR / "model_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_model_folders(
    *,
    metrics_frame: pd.DataFrame,
    predictions_frame: pd.DataFrame,
    coefficients_frame: pd.DataFrame,
    model_artifacts_frame: pd.DataFrame,
) -> None:
    if metrics_frame.empty:
        return
    models_dir = ensure_directory(RESULTS_DIR / "models")
    index_lines = ["# Models", "", "Each folder contains rows filtered to one model family plus a short explanation of how to read that model against `smiles` and `MolADT`.", ""]
    model_names = sorted(metrics_frame["model"].dropna().astype(str).unique().tolist())
    for model_name in model_names:
        model_dir = ensure_directory(models_dir / model_name)
        model_metrics = metrics_frame.loc[metrics_frame["model"] == model_name].copy()
        model_predictions = predictions_frame.loc[predictions_frame["model"] == model_name].copy()
        model_metrics.to_csv(model_dir / "predictive_metrics.csv", index=False)
        model_predictions.to_csv(model_dir / "predictions.csv", index=False)
        if not coefficients_frame.empty and "model_name" in coefficients_frame.columns:
            coefficients_frame.loc[coefficients_frame["model_name"] == model_name].to_csv(model_dir / "coefficients.csv", index=False)
        if not model_artifacts_frame.empty:
            model_artifacts_frame.loc[model_artifacts_frame["model"] == model_name].to_csv(model_dir / "artifacts.csv", index=False)
        readme_path = model_dir / "README.md"
        readme_path.write_text(_model_folder_readme(model_name, model_metrics), encoding="utf-8")
        index_lines.append(f"- `{model_name}`: see `{display_path(readme_path)}`")
    (models_dir / "README.md").write_text("\n".join(index_lines).rstrip() + "\n", encoding="utf-8")


def _model_folder_readme(model_name: str, model_metrics: pd.DataFrame) -> str:
    representations = sorted(model_metrics["representation"].dropna().astype(str).unique().tolist())
    methods = sorted(model_metrics["method"].dropna().astype(str).unique().tolist())
    lines = [f"# {model_name}", ""]
    lines.extend(_model_explanation_lines(model_name))
    lines.append("")
    lines.append("## Present representations")
    lines.append("")
    for representation in representations:
        lines.append(f"- `{representation}`")
    lines.append("")
    lines.append("## Present methods")
    lines.append("")
    for method in methods:
        lines.append(f"- `{method}`")
    lines.append("")
    lines.append("## Best test rows")
    lines.append("")
    test_rows = model_metrics.loc[model_metrics["split"] == "test"].sort_values(["dataset", "representation", "rmse", "mae"])
    for _, row in test_rows.groupby(["dataset", "representation"], sort=True).head(1).iterrows():
        lines.append(
            f"- `{row['dataset']}/{row['representation']}`: RMSE {float(row['rmse']):.4f}, "
            f"MAE {float(row['mae']):.4f}, coverage {float(row['coverage_90']):.3f}."
        )
    lines.append("")
    lines.append("Supporting files in this folder:")
    lines.append("")
    lines.append("- `predictive_metrics.csv`")
    lines.append("- `predictions.csv`")
    if model_name in {"bayes_linear_student_t", "bayes_hierarchical_shrinkage"}:
        lines.append("- `coefficients.csv`")
    else:
        lines.append("- `artifacts.csv`")
    return "\n".join(lines).rstrip() + "\n"


def _model_explanation_lines(model_name: str) -> list[str]:
    if model_name in {"bayes_linear_student_t", "bayes_hierarchical_shrinkage"}:
        return [
            "These are the Stan descriptor baselines.",
            "",
            "They compare `smiles` and `MolADT` fairly because both branches use standardized tabular feature matrices under the same model family.",
            "Use these folders to answer whether the ADT-derived feature table helps before introducing a different learner.",
        ]
    if model_name == "catboost_uncertainty":
        return [
            "This is the shared non-linear tabular comparison model.",
            "",
            "It is the fairest `smiles` vs `MolADT` comparison in the repo because the learner is held fixed and only the representation changes.",
            "If `MolADT` beats `smiles` here, that is evidence about the representation rather than about a model-family switch.",
        ]
    if model_name in {"visnet_ensemble", "dimenetpp_ensemble"}:
        return [
            "This is a geometry-aware model family.",
            "",
            "These rows should be read against `sdf_geom` and `moladt_geom`, not against plain `smiles`.",
            "They answer a different question from the tabular models: what happens when the model can use coordinates and optional MolADT global descriptors.",
        ]
    return [
        "Model-specific explanation unavailable.",
        "",
        "Read this folder together with the representations listed below.",
    ]


def _write_literature_comparison(
    *,
    review_frame: pd.DataFrame,
    baselines_frame: pd.DataFrame,
    aggregated_metrics: pd.DataFrame,
) -> None:
    comparison_rows = _build_literature_comparison_rows(review_frame, baselines_frame, aggregated_metrics)
    details_path = RESULTS_DIR / "literature_comparison.md"
    direct_rows = [row for row in comparison_rows if row["directly_comparable"]]
    indirect_rows = [row for row in comparison_rows if not row["directly_comparable"]]
    lines = ["# Literature Comparison", ""]
    lines.append("## Directly comparable")
    lines.append("")
    if direct_rows:
        for row in direct_rows:
            lines.append(f"- {row['local_result']} vs {row['baseline_result']} ({row['source_url']})")
    else:
        lines.append("- None in the current configuration.")
    lines.append("")
    lines.append("## Same dataset/target but different split/training protocol")
    lines.append("")
    if indirect_rows:
        for row in indirect_rows:
            lines.append(f"- {row['local_result']} vs {row['baseline_result']}. {row['note']} Source: {row['source_url']}")
    else:
        lines.append("- None.")
    details_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    pd.DataFrame(comparison_rows).to_csv(_details_dir() / "literature_comparison.csv", index=False)


def _build_literature_comparison_rows(
    review_frame: pd.DataFrame,
    baselines_frame: pd.DataFrame,
    aggregated_metrics: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    aggregate_test = aggregated_metrics.loc[aggregated_metrics["split"] == "test"].copy() if not aggregated_metrics.empty else pd.DataFrame()
    for _, review_row in review_frame.iterrows():
        dataset = str(review_row["dataset"])
        local_metric_row = aggregate_test.loc[
            (aggregate_test["dataset"] == dataset)
            & (aggregate_test["representation"] == review_row["representation"])
            & (aggregate_test["model"] == review_row["model"])
            & (aggregate_test["method"] == review_row["method"])
        ]
        if local_metric_row.empty:
            local_result = (
                f"{dataset}/{review_row['representation']} local best test RMSE {float(review_row['test_rmse']):.4f}"
            )
            local_split = str(review_row.get("split_scheme", ""))
        else:
            metric_row = local_metric_row.iloc[0]
            local_result = (
                f"{dataset}/{review_row['representation']} aggregated test RMSE {float(metric_row['rmse_mean']):.4f}"
                + (f" ± {float(metric_row['rmse_std']):.4f}" if pd.notna(metric_row.get("rmse_std", pd.NA)) else "")
            )
            local_split = str(metric_row.get("split_scheme", ""))
        dataset_baselines = baselines_frame.loc[baselines_frame["dataset"] == dataset]
        for _, baseline in dataset_baselines.iterrows():
            metric_name = str(baseline.get("metric_name", "")).strip()
            metric_value = baseline.get("metric_value", pd.NA)
            if metric_name and pd.notna(metric_value):
                baseline_result = f"{baseline['model_name']} {metric_name} {float(metric_value):.4f}"
            else:
                baseline_result = f"{baseline['model_name']} (context row only)"
            rows.append(
                {
                    "dataset": dataset,
                    "representation": review_row["representation"],
                    "model": review_row["model"],
                    "method": review_row["method"],
                    "split_scheme": local_split,
                    "local_result": local_result,
                    "baseline_result": baseline_result,
                    "baseline_split_protocol": baseline["split_protocol"],
                    "source_url": baseline["source_url"],
                    "directly_comparable": bool(baseline["directly_comparable"]) and local_split == baseline["split_protocol"],
                    "note": baseline["note"],
                }
            )
    return rows


def _write_timing_report(timing: pd.DataFrame) -> None:
    lines = ["# ZINC Timing", ""]
    for _, row in timing.iterrows():
        lines.append(
            f"- `{row['stage']}`: {row.get('description', '')} "
            f"{float(row['molecules_per_second']):.1f} mol/s with {int(row['failure_count'])} failures."
        )
    items_path = _details_dir() / "zinc_timing_items.csv"
    if items_path.exists():
        lines.append("")
        lines.append(f"Detailed per-item parse timings: `{display_path(items_path)}`")
        items = pd.read_csv(items_path)
        if not items.empty:
            lines.append("")
            lines.append("## Slowest Parse Items")
            lines.append("")
            for stage_name in ("smiles_library_parse", "moladt_file_parse"):
                stage_items = items.loc[items["stage"] == stage_name].sort_values("latency_us", ascending=False).head(5)
                if stage_items.empty:
                    continue
                lines.append(f"### {stage_name}")
                lines.append("")
                for _, item in stage_items.iterrows():
                    lines.append(
                        f"- `{item['mol_id']}` `{item['item_path']}`: "
                        f"{float(item['latency_us']):.1f} us, success={bool(item['success'])}."
                    )
                lines.append("")
    manifest_path = _details_dir() / "zinc_timing_library_manifest.csv"
    if manifest_path.exists():
        lines.append(f"Matched timing-library manifest: `{display_path(manifest_path)}`")
    (RESULTS_DIR / "zinc_timing.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
