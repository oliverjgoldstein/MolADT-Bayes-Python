from __future__ import annotations

import argparse
import math
import shutil
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .benchmark_zinc import run_zinc_benchmark
from .common import DEFAULT_SEED, RESULTS_DIR, display_path, ensure_directory, log, log_stage
from .literature_baselines import literature_baselines_frame
from .model_errors import OptionalModelDependencyError
from .model_registry import GEOMETRIC_MODEL_REGISTRY, TABULAR_MODEL_REGISTRY
from .process_freesolv import FreeSolvArtifacts, process_freesolv_dataset
from .process_qm9 import QM9Artifacts, process_qm9_dataset
from .predictive_metrics import aggregate_seed_metrics, build_calibration_rows
from .report_graphs import (
    write_moleculenet_comparison_overviews,
    write_timing_stage_overview,
)
from .stan_runner import ALL_METHODS, StanRunConfig, run_model_suite, write_stan_data_json
from .tabular_runner import CatBoostRunConfig
from .geometry_runner import GeometryRunConfig

DEFAULT_STAN_METHODS = tuple(ALL_METHODS)
DEFAULT_STAN_METHODS_ARG = ",".join(DEFAULT_STAN_METHODS)
DEFAULT_STAN_MODELS = ("bayes_linear_student_t", "bayes_hierarchical_shrinkage")
DEFAULT_STAN_MODELS_ARG = ",".join(DEFAULT_STAN_MODELS)
DEFAULT_FREESOLV_STAN_METHODS = ("laplace",)
DEFAULT_FREESOLV_STAN_MODELS = ("bayes_gp_rbf_screened",)
DEFAULT_QM9_STAN_METHODS = ("optimize",)
DEFAULT_QM9_STAN_MODELS = ("bayes_linear_student_t",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.run_all")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freesolv = subparsers.add_parser("freesolv", help="Run the FreeSolv benchmark")
    _add_common_benchmark_args(freesolv)
    freesolv.add_argument("--include-moladt-predictive", action="store_true")
    freesolv.add_argument("--skip-moladt", dest="skip_moladt", action="store_true")
    freesolv.add_argument("--skip-sdf", dest="skip_moladt", action="store_true", help=argparse.SUPPRESS)

    qm9 = subparsers.add_parser("qm9", help="Run the QM9 benchmark")
    _add_common_benchmark_args(qm9)
    qm9.add_argument("--include-moladt-predictive", action="store_true")
    qm9.add_argument("--limit", type=int, default=None, help="Deterministic QM9 source-row limit before feature export; omit for the full local download")
    qm9.add_argument("--split-mode", choices=("subset", "paper", "long"), default="long")

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
    benchmark.add_argument("--qm9-split-mode", choices=("subset", "paper", "long"), default="long")
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
    models.add_argument("--qm9-split-mode", choices=("subset", "paper", "long"), default="long")
    models.add_argument("--include-moladt-predictive", action="store_true", default=True)
    models.add_argument("--skip-moladt", dest="skip_moladt", action="store_true")
    models.add_argument("--skip-sdf", dest="skip_moladt", action="store_true", help=argparse.SUPPRESS)
    return parser


def _add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--methods",
        default=DEFAULT_STAN_METHODS_ARG,
        help="Comma-separated inference methods: sample,variational,pathfinder,optimize,laplace",
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_STAN_MODELS_ARG,
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
    parser.add_argument(
        "--preferred-qm9-geometry-representation",
        choices=("sdf_geom", "moladt_geom", "moladt_featurized_geom"),
        default=None,
        help="Restrict QM9 geometry runs to one exported representation",
    )
    parser.add_argument("--verbose", action="store_true")


def _parsed_models_arg(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(model.strip() for model in str(args.models).split(",") if model.strip())


def _uses_fixed_freesolv_contract(args: argparse.Namespace) -> bool:
    requested = _parsed_models_arg(args)
    return requested in (DEFAULT_FREESOLV_STAN_MODELS, DEFAULT_STAN_MODELS)


def _uses_fixed_qm9_contract(args: argparse.Namespace) -> bool:
    requested = _parsed_models_arg(args)
    return requested in (DEFAULT_QM9_STAN_MODELS, DEFAULT_STAN_MODELS)


def _uses_sdf_only_qm9_predictive_contract(
    args: argparse.Namespace,
    extra_models: tuple[str, ...],
) -> bool:
    return bool(extra_models) and bool(getattr(args, "include_moladt_predictive", False))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    extra_models = _parse_extra_models(args)
    _remove_legacy_report_artifacts()
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    training_curve_rows: list[dict[str, object]] = []
    model_artifact_rows: list[dict[str, object]] = []

    if args.command == "freesolv":
        if args.verbose:
            log("Starting FreeSolv benchmark")
            log(f"Results directory: {display_path(RESULTS_DIR)}")
            log_stage("benchmark", 1, 3, "Preparing FreeSolv exports")
        artifacts = process_freesolv_dataset(
            seed=args.seed,
            force=args.force,
            include_moladt=True,
            include_legacy_tabular=not _uses_fixed_freesolv_contract(args),
            verbose=args.verbose,
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
            log("QM9 target: mu (dipole moment)")
            log(f"Results directory: {display_path(RESULTS_DIR)}")
            log_stage("benchmark", 1, 3, "Preparing QM9 exports")
        artifacts = process_qm9_dataset(
            seed=args.seed,
            force=args.force,
            limit=qm9_limit,
            split_mode=qm9_split_mode,
            include_legacy_tabular=not (
                _uses_fixed_qm9_contract(args)
                or _uses_sdf_only_qm9_predictive_contract(args, extra_models)
            ),
            verbose=args.verbose,
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
                f"MolADT-only Stan comparison, extra_models={','.join(extra_models) or '(none)'})"
            )
            log("QM9 target: mu (dipole moment)")
            log(f"Results directory: {display_path(RESULTS_DIR)}")
            log_stage("benchmark", 1, 3, "Running FreeSolv benchmark flow")
        freesolv = process_freesolv_dataset(
            seed=args.seed,
            force=args.force,
            include_moladt=True,
            include_legacy_tabular=not _uses_fixed_freesolv_contract(args),
            verbose=args.verbose,
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
        if args.verbose:
            log_stage("benchmark", 2, 3, "Running QM9 benchmark flow")
        qm9 = process_qm9_dataset(
            seed=args.seed,
            force=args.force,
            limit=qm9_limit,
            split_mode=qm9_split_mode,
            include_legacy_tabular=not (
                _uses_fixed_qm9_contract(args)
                or _uses_sdf_only_qm9_predictive_contract(args, extra_models)
            ),
            verbose=args.verbose,
        )
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
            log_stage("benchmark", 1, 3, "Running FreeSolv model flow")
        freesolv = process_freesolv_dataset(
            seed=args.seed,
            force=args.force,
            include_moladt=True,
            include_legacy_tabular=not _uses_fixed_freesolv_contract(args),
            verbose=args.verbose,
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
        if args.verbose:
            log_stage("benchmark", 2, 3, "Running QM9 model flow")
        qm9 = process_qm9_dataset(
            seed=args.seed,
            force=args.force,
            limit=qm9_limit,
            split_mode=qm9_split_mode,
            include_legacy_tabular=not (
                _uses_fixed_qm9_contract(args)
                or _uses_sdf_only_qm9_predictive_contract(args, extra_models)
            ),
            verbose=args.verbose,
        )
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
        if args.verbose:
            total_stages = 2 if args.command in {"freesolv", "qm9", "models"} else 3 if args.command == "benchmark" else 1
            log_stage(
                "benchmark",
                total_stages,
                total_stages,
                f"Writing reports (metrics={len(metrics_rows)} predictions={len(prediction_rows)} "
                f"coefficients={len(coefficient_rows)} curves={len(training_curve_rows)} artifacts={len(model_artifact_rows)})",
            )
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
        molecule_net_comparison = _build_moleculenet_comparison_frame(review_frame)
        molecule_net_comparison = _attach_moleculenet_uncertainty(
            molecule_net_comparison,
            predictions_frame=predictions_frame,
        )
        molecule_net_comparison.to_csv(_details_dir() / "moleculenet_comparison.csv", index=False)
        write_moleculenet_comparison_overviews(molecule_net_comparison, RESULTS_DIR)
        _write_freesolv_bayesian_artifacts(
            review_frame=review_frame,
            metrics_frame=metrics_frame,
            coefficients_frame=coefficients_frame,
            echo_to_console=args.command == "freesolv",
        )
    else:
        metrics_frame = pd.DataFrame()
        generalization = pd.DataFrame()
        review_frame = pd.DataFrame()
        training_curves_frame = pd.DataFrame()
        calibration_frame = pd.DataFrame()
    timing = _load_timing_results() if args.command == "zinc-timing" else pd.DataFrame()
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
    methods = _stan_methods_for_artifacts(artifacts, args)
    models = _stan_models_for_artifacts(artifacts, args)
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
    moladt_featurized_bundle = getattr(artifacts, "moladt_featurized_export", None)
    moladt_bundle = getattr(artifacts, "moladt_export", None)
    if isinstance(artifacts, FreeSolvArtifacts) and models == DEFAULT_FREESOLV_STAN_MODELS:
        if moladt_featurized_bundle is None:
            raise RuntimeError("FreeSolv featurized export is required for the reviewer benchmark")
        bundles: list[ExportedDataset] = [moladt_featurized_bundle]
    elif isinstance(artifacts, QM9Artifacts) and models == DEFAULT_QM9_STAN_MODELS:
        if moladt_featurized_bundle is None:
            raise RuntimeError("QM9 featurized export is required for the reviewer benchmark")
        bundles = [moladt_featurized_bundle]
    else:
        if moladt_bundle is None:
            raise RuntimeError("MolADT export is required for the Stan benchmark")
        bundles = [moladt_bundle]
        if moladt_featurized_bundle is not None:
            bundles.append(moladt_featurized_bundle)
    for bundle in bundles:
        write_stan_data_json(bundle, student_df=config.student_df)
        if args.verbose:
            log(
                f"[{bundle.dataset_name}/{bundle.representation}] tabular_rows="
                f"train={len(bundle.y_train)} valid={len(bundle.y_valid)} test={len(bundle.y_test)} "
                f"features={len(bundle.feature_names)}"
            )
        for model_name in models:
            if model_name == "bayes_gp_rbf_screened" and bundle.representation != "moladt_featurized":
                if args.verbose:
                    log(
                        f"[{bundle.dataset_name}/{bundle.representation}] "
                        f"Skipping `{model_name}` because it is reserved for the SDF-backed featurized FreeSolv path"
                    )
                continue
            if args.verbose:
                log(f"[{bundle.dataset_name}/{bundle.representation}] Running Stan model `{model_name}`")
            rows, predictions, coefficients = run_model_suite(bundle, model_name=model_name, config=config)
            metrics_rows.extend(rows)
            prediction_rows.extend(predictions)
            coefficient_rows.extend(coefficients)
    extra_seeds = _extra_model_seeds(args.seed, args.num_seeds if args.paper_mode else 1)
    for model_name in extra_models:
        if model_name in TABULAR_MODEL_REGISTRY:
            runner = TABULAR_MODEL_REGISTRY[model_name].runner
            try:
                for bundle in getattr(artifacts, "tabular_exports", {}).values():
                    if getattr(bundle, "representation", "") == "smiles":
                        continue
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
                geometric_exports = getattr(artifacts, "geometric_exports", {})
                preferred_qm9_geometry = getattr(args, "preferred_qm9_geometry_representation", None)
                geometric_bundles = list(geometric_exports.values())
                if preferred_qm9_geometry is not None:
                    geometric_bundles = [
                        bundle for bundle in geometric_bundles if getattr(bundle, "representation", "") == preferred_qm9_geometry
                    ]
                    if not geometric_bundles:
                        raise RuntimeError(
                            f"Preferred QM9 geometry representation `{preferred_qm9_geometry}` was not exported"
                        )
                for bundle in geometric_bundles:
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


def _stan_models_for_artifacts(
    artifacts: FreeSolvArtifacts | QM9Artifacts,
    args: argparse.Namespace,
) -> tuple[str, ...]:
    requested = tuple(model.strip() for model in args.models.split(",") if model.strip())
    if isinstance(artifacts, FreeSolvArtifacts) and args.models == DEFAULT_STAN_MODELS_ARG:
        return DEFAULT_FREESOLV_STAN_MODELS
    if isinstance(artifacts, QM9Artifacts) and args.models == DEFAULT_STAN_MODELS_ARG:
        return DEFAULT_QM9_STAN_MODELS
    return requested


def _stan_methods_for_artifacts(
    artifacts: FreeSolvArtifacts | QM9Artifacts,
    args: argparse.Namespace,
) -> tuple[str, ...]:
    requested = tuple(method.strip() for method in args.methods.split(",") if method.strip())
    if isinstance(artifacts, FreeSolvArtifacts) and args.methods == DEFAULT_STAN_METHODS_ARG:
        return DEFAULT_FREESOLV_STAN_METHODS
    if isinstance(artifacts, QM9Artifacts) and args.methods == DEFAULT_STAN_METHODS_ARG:
        return DEFAULT_QM9_STAN_METHODS
    return requested


def _details_dir() -> Any:
    return ensure_directory(RESULTS_DIR / "details")


def _remove_legacy_report_artifacts() -> None:
    legacy_files = (
        "summary.md",
        "model_report.md",
        "generalization_report.md",
        "literature_context.md",
        "zinc_timing.md",
        "timing_result_files.txt",
        "caption.txt",
        "freesolv_rmse_vs_moleculenet.caption.txt",
        "qm9_mae_vs_moleculenet.caption.txt",
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
        "rmse_train_test_vs_literature.svg",
        "inference_sweep_overview.svg",
        "predicted_vs_actual_overview.svg",
        "calibration.csv",
        "coverage_calibration.svg",
        "rmse_comparison.svg",
        "mae_comparison.svg",
        "r2_comparison.svg",
        "coverage_90_comparison.svg",
        "rmse_frontier_comparison.svg",
        "mae_frontier_comparison.svg",
        "r2_frontier_comparison.svg",
        "coverage_90_frontier_comparison.svg",
        "freesolv_rmse_vs_moleculenet.svg",
        "qm9_mae_vs_moleculenet.svg",
    )
    for name in legacy_files:
        path = RESULTS_DIR / name
        if path.exists():
            path.unlink()
    for path in RESULTS_DIR.iterdir() if RESULTS_DIR.exists() else ():
        if path.is_dir() and (path.name in {"stan_output", "models", "figures", "model_artifacts"} or path.name.startswith("review_")):
            shutil.rmtree(path, ignore_errors=True)


def _select_reviewer_rows(selection_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (dataset, representation), frame in selection_rows.groupby(["dataset", "representation"], sort=True):
        primary_metric, secondary_metric = _metric_priority_for_dataset(str(dataset))
        ranked = frame.sort_values(
            [primary_metric, secondary_metric, "runtime_seconds", "model", "method"],
            kind="stable",
        )
        rows.append(ranked.head(1))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)




def _best_test_row(test_rows: pd.DataFrame, *, dataset: str, metric: str) -> dict[str, Any] | None:
    subset = test_rows.loc[test_rows["dataset"] == dataset].copy()
    if subset.empty:
        return None
    ordered = subset.sort_values([metric, "rmse", "mae", "runtime_seconds", "representation", "model", "method"])
    return ordered.iloc[0].to_dict()


def _primary_metric_for_dataset(dataset: str) -> tuple[str, str]:
    if dataset == "qm9":
        return ("MAE", "test_mae")
    return ("RMSE", "test_rmse")


def _metric_priority_for_dataset(dataset: str) -> tuple[str, str]:
    if dataset == "qm9":
        return ("mae", "rmse")
    return ("rmse", "mae")


def _selected_run_keys(metrics: pd.DataFrame) -> pd.DataFrame:
    valid_rows = metrics.loc[metrics["split"] == "valid"].copy()
    selected = _select_reviewer_rows(valid_rows)
    # `seed` is not a stable run identifier here because the benchmark records
    # different split-level predictive seeds for train/valid/test rows.
    return selected.loc[:, ["dataset", "representation", "model", "method"]].drop_duplicates().reset_index(drop=True)


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
        subset = metrics.loc[mask].copy()
        split_rows: dict[str, pd.Series] = {}
        for split_name in ("train", "valid", "test"):
            split_subset = subset.loc[subset["split"] == split_name].sort_values(
                ["rmse", "mae", "runtime_seconds"],
                kind="stable",
            )
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


def _write_freesolv_bayesian_artifacts(
    *,
    review_frame: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    coefficients_frame: pd.DataFrame,
    echo_to_console: bool,
) -> None:
    artifact = _build_freesolv_bayesian_artifact(review_frame, metrics_frame, coefficients_frame)
    if artifact is None:
        return
    model_path = RESULTS_DIR / "freesolv_bayesian_model.txt"
    uncertainty_path = _details_dir() / "freesolv_train_test_uncertainty.csv"
    model_path.write_text(artifact["model_text"], encoding="utf-8")
    artifact["uncertainty_frame"].to_csv(uncertainty_path, index=False)
    if echo_to_console:
        for line in artifact["summary_lines"]:
            log(line)
    else:
        log(
            "Wrote FreeSolv Bayesian summary artifacts: "
            f"{display_path(model_path)}, {display_path(uncertainty_path)}"
        )


def _build_freesolv_bayesian_artifact(
    review_frame: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    coefficients_frame: pd.DataFrame,
) -> dict[str, Any] | None:
    if review_frame.empty or metrics_frame.empty or coefficients_frame.empty:
        return None
    freesolv_rows = review_frame.loc[review_frame["dataset"] == "freesolv"].copy()
    if freesolv_rows.empty:
        return None
    selected = freesolv_rows.iloc[0]
    key_mask = (
        (metrics_frame["dataset"] == "freesolv")
        & (metrics_frame["representation"] == selected["representation"])
        & (metrics_frame["model"] == selected["model"])
        & (metrics_frame["method"] == selected["method"])
    )
    uncertainty_frame = (
        metrics_frame.loc[key_mask & metrics_frame["split"].isin(["train", "test"])]
        .loc[:, ["split", "n_eval", "rmse", "mae", "r2", "predictive_sd_mean", "coverage_90", "mean_log_predictive_density", "draw_count", "runtime_seconds"]]
        .copy()
    )
    if uncertainty_frame.empty:
        return None
    uncertainty_frame["split"] = pd.Categorical(uncertainty_frame["split"], categories=["train", "test"], ordered=True)
    uncertainty_frame = uncertainty_frame.sort_values("split").reset_index(drop=True)
    coefficient_mask = (
        (coefficients_frame["dataset"] == "freesolv")
        & (coefficients_frame["representation"] == selected["representation"])
        & (coefficients_frame["model"] == selected["model"])
        & (coefficients_frame["method"] == selected["method"])
    )
    coefficient_rows = coefficients_frame.loc[coefficient_mask].copy()
    if coefficient_rows.empty:
        return None
    model_text = _format_freesolv_bayesian_model_text(selected, uncertainty_frame, coefficient_rows)
    summary_lines = [
        "FreeSolv Bayesian summary",
        f"  selected run: {selected['representation']} / {selected['model']} / {selected['method']}",
    ]
    for _, row in uncertainty_frame.iterrows():
        split_label = str(row["split"]).title()
        summary_lines.append(
            f"  {split_label}: rmse={float(row['rmse']):.3f}, mae={float(row['mae']):.3f}, "
            f"mean_predictive_sd={float(row['predictive_sd_mean']):.3f}, coverage_90={float(row['coverage_90']):.3f}"
        )
    summary_lines.append(f"  model: {display_path(RESULTS_DIR / 'freesolv_bayesian_model.txt')}")
    summary_lines.append(f"  uncertainty: {display_path(_details_dir() / 'freesolv_train_test_uncertainty.csv')}")
    return {
        "model_text": model_text,
        "uncertainty_frame": uncertainty_frame,
        "summary_lines": summary_lines,
    }


def _format_freesolv_bayesian_model_text(
    selected: pd.Series,
    uncertainty_frame: pd.DataFrame,
    coefficient_rows: pd.DataFrame,
) -> str:
    parameter_lookup = {
        str(row["parameter_name"]): row
        for _, row in coefficient_rows.iterrows()
    }
    alpha_row = parameter_lookup.get("alpha")
    signal_row = parameter_lookup.get("signal_scale")
    length_row = parameter_lookup.get("lengthscale")
    sigma_row = parameter_lookup.get("sigma")
    lines = [
        "FreeSolv Bayesian model summary",
        f"Selected run: {selected['representation']} / {selected['model']} / {selected['method']}",
        f"Split scheme: {selected.get('split_scheme', '')}",
        (
            "Rows: "
            f"train={int(float(selected.get('train_n_eval', 0)))} "
            f"valid={int(float(selected.get('valid_n_eval', 0)))} "
            f"test={int(float(selected.get('test_n_eval', 0)))}"
        ),
        "",
        "Train/test predictive uncertainty",
    ]
    for _, row in uncertainty_frame.iterrows():
        lines.extend(
            [
                f"- {str(row['split']).title()}",
                f"  n_eval = {int(row['n_eval'])}",
                f"  rmse = {float(row['rmse']):.6f}",
                f"  mae = {float(row['mae']):.6f}",
                f"  r2 = {float(row['r2']):.6f}",
                f"  mean_predictive_sd = {float(row['predictive_sd_mean']):.6f}",
                f"  empirical_90pct_coverage = {float(row['coverage_90']):.6f}",
                f"  mean_log_predictive_density = {float(row['mean_log_predictive_density']):.6f}",
            ]
        )
    lines.extend(["", "Posterior hyperparameters"])
    for name in ("alpha", "signal_scale", "lengthscale", "sigma"):
        row = parameter_lookup.get(name)
        if row is None:
            continue
        lines.append(f"- {name} = {_posterior_summary(row)}")
    if alpha_row is not None and signal_row is not None and length_row is not None and sigma_row is not None:
        alpha = float(alpha_row["posterior_mean"])
        signal_scale = float(signal_row["posterior_mean"])
        lengthscale = float(length_row["posterior_mean"])
        sigma = float(sigma_row["posterior_mean"])
        lines.extend(
            [
                "",
                "Posterior-mean predictive model",
                (
                    f"mu(x*) = {alpha:.6f} + "
                    f"K_rbf(x*, X_train; signal_scale={signal_scale:.6f}, lengthscale={lengthscale:.6f}) @ "
                    f"[K_rbf(X_train, X_train; signal_scale={signal_scale:.6f}, lengthscale={lengthscale:.6f}) + "
                    f"{sigma:.6f}^2 I]^-1 @ (y_train - {alpha:.6f})"
                ),
                "k_rbf(a, b; signal_scale=s, lengthscale=l) = s^2 * exp(-||a - b||^2 / (2 * l^2))",
                f"y(x*) | data is summarized by Normal(mu(x*), predictive_sd(x*)^2) with observation noise sigma={sigma:.6f}",
            ]
        )
    return "\n".join(lines) + "\n"


def _posterior_summary(row: pd.Series) -> str:
    return (
        f"{float(row['posterior_mean']):.6f} "
        f"(sd {float(row['posterior_sd']):.6f}; "
        f"p05 {float(row['posterior_p05']):.6f}; "
        f"p95 {float(row['posterior_p95']):.6f})"
    )


def _build_simple_review_frame(generalization: pd.DataFrame, *, baselines_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, dataset_rows in generalization.groupby("dataset", sort=True):
        primary_metric, secondary_metric = _metric_priority_for_dataset(str(dataset))
        ordered_rows = dataset_rows.sort_values(
            [f"valid_{primary_metric}", f"valid_{secondary_metric}", f"test_{primary_metric}", "fit_runtime_seconds", "representation", "model", "method"],
            kind="stable",
        )
        row = ordered_rows.iloc[0]
        dataset = str(row["dataset"])
        representation = str(row["representation"])
        local_metric_name, local_metric_column = _primary_metric_for_dataset(dataset)
        train_metric_column = "train_mae" if dataset == "qm9" else "train_rmse"
        valid_metric_column = "valid_mae" if dataset == "qm9" else "valid_rmse"
        train_metric_value = float(row[train_metric_column])
        valid_metric_value = float(row[valid_metric_column])
        test_metric_value = float(row[local_metric_column])
        local_metric_value = test_metric_value
        baseline = _select_review_baseline(baselines_frame, dataset)
        if baseline is None:
            literature_display = "No external context attached"
            literature_rmse = None
            literature_metric = ""
            paper_value = float("nan")
            paper_model_name = ""
            paper_source_title = ""
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
            paper_value = float(metric_value) if pd.notna(metric_value) else float("nan")
            paper_model_name = str(baseline.get("model_name", ""))
            paper_source_title = str(baseline.get("source_title", ""))
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
                "dataset_label": "FreeSolv" if dataset == "freesolv" else "QM9" if dataset == "qm9" else dataset,
                "representation": representation,
                "model": str(row["model"]),
                "method": str(row["method"]),
                "local_metric_name": local_metric_name,
                "local_metric_value": local_metric_value,
                "train_metric_value": train_metric_value,
                "valid_metric_value": valid_metric_value,
                "test_metric_value": test_metric_value,
                "selection_split": "valid",
                "train_rmse": float(row["train_rmse"]),
                "test_rmse": float(row["test_rmse"]),
                "test_minus_train_rmse": float(row["test_minus_train_rmse"]),
                "train_mae": float(row.get("train_mae", float("nan"))),
                "valid_mae": float(row.get("valid_mae", float("nan"))),
                "test_mae": float(row.get("test_mae", float("nan"))),
                "train_r2": float(row.get("train_r2", float("nan"))),
                "valid_r2": float(row.get("valid_r2", float("nan"))),
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
                "paper_metric_name": literature_metric,
                "paper_metric_value": paper_value,
                "paper_model_name": paper_model_name,
                "paper_source_title": paper_source_title,
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
    metric_name, _ = _primary_metric_for_dataset(dataset)
    preferred = subset.loc[(subset["metric_name"] == metric_name) & subset["metric_value"].notna()]
    if not preferred.empty:
        return preferred.sort_values(["metric_value", "model_name"]).iloc[0]
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
        return _freesolv_context_note(baseline_note=baseline_note)
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


def _build_moleculenet_comparison_frame(review_frame: pd.DataFrame) -> pd.DataFrame:
    if review_frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in review_frame.iterrows():
        paper_value = row.get("paper_metric_value", pd.NA)
        if pd.isna(paper_value):
            continue
        rows.append(
            {
                "dataset": str(row["dataset"]),
                "dataset_label": str(row.get("dataset_label", row["dataset"])),
                "representation": str(row.get("representation", "moladt")),
                "metric_name": str(row["local_metric_name"]),
                "train_value": float(row.get("train_metric_value", row["local_metric_value"])),
                "valid_value": float(row.get("valid_metric_value", row["local_metric_value"])),
                "test_value": float(row.get("test_metric_value", row["local_metric_value"])),
                "local_value": float(row["local_metric_value"]),
                "paper_value": float(paper_value),
                "model": str(row["model"]),
                "method": str(row["method"]),
                "selection_split": str(row.get("selection_split", "valid")),
                "paper_model_name": str(row.get("paper_model_name", "")),
                "paper_source_title": str(row.get("paper_source_title", "")),
                "note": str(row.get("note", "")),
            }
        )
    return pd.DataFrame(rows)


def _attach_moleculenet_uncertainty(
    comparison_frame: pd.DataFrame,
    *,
    predictions_frame: pd.DataFrame,
) -> pd.DataFrame:
    if comparison_frame.empty or predictions_frame.empty:
        return comparison_frame
    frame = comparison_frame.copy()
    required = {"dataset", "representation", "model", "method", "split", "actual", "predicted_mean", "predictive_sd"}
    if not required.issubset(predictions_frame.columns):
        return frame
    for row_index, row in frame.iterrows():
        if str(row["dataset"]) != "freesolv":
            continue
        selected = predictions_frame.loc[
            (predictions_frame["dataset"] == row["dataset"])
            & (predictions_frame["representation"] == row["representation"])
            & (predictions_frame["model"] == row["model"])
            & (predictions_frame["method"] == row["method"])
        ].copy()
        if selected.empty:
            continue
        for split_name, prefix in (("train", "train"), ("valid", "valid"), ("test", "test")):
            interval = _posterior_rmse_interval(selected.loc[selected["split"] == split_name].copy())
            if interval is None:
                continue
            frame.loc[row_index, f"{prefix}_interval_low"] = interval[0]
            frame.loc[row_index, f"{prefix}_interval_high"] = interval[1]
    return frame


def _posterior_rmse_interval(
    predictions: pd.DataFrame,
    *,
    draws: int = 4000,
) -> tuple[float, float] | None:
    if predictions.empty:
        return None
    actual = predictions["actual"].to_numpy(dtype=float)
    predicted_mean = predictions["predicted_mean"].to_numpy(dtype=float)
    predictive_sd = np.clip(predictions["predictive_sd"].to_numpy(dtype=float), 1e-9, None)
    if actual.size == 0:
        return None
    split_name = str(predictions["split"].iloc[0])
    seed = DEFAULT_SEED + sum(ord(char) for char in split_name) + actual.size
    rng = np.random.default_rng(seed)
    sampled_predictions = rng.normal(
        loc=predicted_mean[np.newaxis, :],
        scale=predictive_sd[np.newaxis, :],
        size=(draws, actual.size),
    )
    sampled_rmse = np.sqrt(np.mean(np.square(sampled_predictions - actual[np.newaxis, :]), axis=1))
    return float(np.quantile(sampled_rmse, 0.05)), float(np.quantile(sampled_rmse, 0.95))


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


def _freesolv_context_note(*, baseline_note: str = "") -> str:
    contextual = (
        "MoleculeNet Table 3 is the paper baseline. The local result is the fixed MolADT benchmark run on the repo split, "
        "so the metric matches but the split and model family still differ."
    )
    if baseline_note:
        return f"{contextual} {baseline_note}"
    return contextual


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
            "The local run uses paper-sized split counts, but it still compares against MoleculeNet's DTNN MAE row "
            "rather than reproducing the original neural training recipe exactly."
        )
    else:
        contextual = (
            "MoleculeNet Table 3 is the paper baseline. The metric matches MAE, but the local split and training recipe "
            "still differ from the original DTNN benchmark."
        )
    if baseline_note:
        return f"{contextual} {baseline_note}"
    return contextual


def _parse_extra_models(args: argparse.Namespace) -> tuple[str, ...]:
    models = [item.strip() for item in str(getattr(args, "extra_models", "")).split(",") if item.strip()]
    if not models and getattr(args, "command", "") == "models":
        models = ["catboost_uncertainty", "visnet_ensemble", "dimenetpp_ensemble"]
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
        lines.append(f"Detailed per-item timings: `{display_path(items_path)}`")
        items = pd.read_csv(items_path)
        if not items.empty:
            lines.append("")
            lines.append("## Slowest Timed Items")
            lines.append("")
            for stage_name in ("smiles_to_json", "sdf_to_moladt", "sdf_to_smiles", "moladt_to_json", "json_to_moladt"):
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
    manifest_path = _details_dir() / "zinc_timing_corpus_manifest.csv"
    if manifest_path.exists():
        lines.append(f"Matched timing-library manifest: `{display_path(manifest_path)}`")
    result_files_path = RESULTS_DIR / "timing_result_files.txt"
    if result_files_path.exists():
        lines.append(f"Result-file index: `{display_path(result_files_path)}`")
    (RESULTS_DIR / "zinc_timing.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
