from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from typing import Any

import pandas as pd

from .benchmark_zinc import run_zinc_benchmark
from .common import DEFAULT_SEED, RESULTS_DIR, display_path, ensure_directory, log
from .process_freesolv import FreeSolvArtifacts, process_freesolv_dataset
from .process_qm9 import QM9Artifacts, process_qm9_dataset
from .report_graphs import (
    write_review_rmse_overview,
    write_timing_stage_overview,
)
from .stan_runner import ALL_METHODS, StanRunConfig, run_model_suite, write_stan_data_json

MOLECULENET_SOURCE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/"
GILMER_SOURCE_URL = "https://proceedings.mlr.press/v70/gilmer17a.html"
GILMER_SUPPLEMENT_URL = "https://proceedings.mlr.press/v70/gilmer17a/gilmer17a-supp.pdf"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.run_all")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke-test", help="Run the FreeSolv smoke test benchmark")
    _add_common_benchmark_args(smoke)
    smoke.add_argument("--skip-sdf", action="store_true")

    qm9 = subparsers.add_parser("qm9", help="Run the QM9 benchmark")
    _add_common_benchmark_args(qm9)
    qm9.add_argument("--limit", type=int, default=2000, help="Deterministic QM9 subset size for the first benchmark run")

    zinc = subparsers.add_parser("zinc-timing", help="Run the ZINC timing benchmark")
    zinc.add_argument("--dataset-size", default="250K")
    zinc.add_argument("--dataset-dimension", default="2D")
    zinc.add_argument("--limit", type=int, default=None)
    zinc.add_argument("--include-moladt", action="store_true")
    zinc.add_argument("--force", action="store_true")
    zinc.add_argument("--verbose", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="Run FreeSolv, QM9, and ZINC in order")
    _add_common_benchmark_args(benchmark)
    benchmark.add_argument("--qm9-limit", type=int, default=2000)
    benchmark.add_argument("--zinc-dataset-size", default="250K")
    benchmark.add_argument("--zinc-dataset-dimension", default="2D")
    benchmark.add_argument("--zinc-limit", type=int, default=None)
    benchmark.add_argument("--skip-sdf", action="store_true")
    benchmark.add_argument("--include-moladt", action="store_true")
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
    parser.add_argument("--verbose", action="store_true")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _remove_legacy_report_artifacts()
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    if args.command == "smoke-test":
        if args.verbose:
            log("Starting FreeSolv smoke benchmark")
        artifacts = process_freesolv_dataset(seed=args.seed, force=args.force, include_sdf=not args.skip_sdf)
        _extend_with_property_results(artifacts, metrics_rows, prediction_rows, coefficient_rows, args)
    elif args.command == "qm9":
        if args.verbose:
            log(f"Starting QM9 benchmark with limit={args.limit}")
        artifacts = process_qm9_dataset(seed=args.seed, force=args.force, limit=args.limit)
        _extend_with_property_results(artifacts, metrics_rows, prediction_rows, coefficient_rows, args)
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
        if args.verbose:
            log(
                "Starting full benchmark "
                f"(qm9_limit={args.qm9_limit}, zinc_dataset_size={args.zinc_dataset_size}, "
                f"zinc_dataset_dimension={args.zinc_dataset_dimension}, zinc_limit={args.zinc_limit}, "
                f"include_moladt={args.include_moladt})"
            )
            log(f"Results directory: {display_path(RESULTS_DIR)}")
        freesolv = process_freesolv_dataset(seed=args.seed, force=args.force, include_sdf=not args.skip_sdf)
        _extend_with_property_results(freesolv, metrics_rows, prediction_rows, coefficient_rows, args)
        qm9 = process_qm9_dataset(seed=args.seed, force=args.force, limit=args.qm9_limit)
        _extend_with_property_results(qm9, metrics_rows, prediction_rows, coefficient_rows, args)
        run_zinc_benchmark(
            dataset_size=args.zinc_dataset_size,
            dataset_dimension=args.zinc_dataset_dimension,
            limit=args.zinc_limit,
            include_moladt=args.include_moladt,
            force=args.force,
            verbose=args.verbose,
        )
    else:
        raise RuntimeError(f"Unsupported command {args.command}")

    if metrics_rows:
        details_dir = _details_dir()
        metrics_path = details_dir / "predictive_metrics.csv"
        predictions_path = details_dir / "predictions.csv"
        coefficients_path = details_dir / "model_coefficients.csv"
        metrics_frame = pd.DataFrame(metrics_rows)
        predictions_frame = pd.DataFrame(prediction_rows)
        coefficients_frame = pd.DataFrame(coefficient_rows)
        metrics_frame.to_csv(metrics_path, index=False)
        predictions_frame.to_csv(predictions_path, index=False)
        coefficients_frame.to_csv(coefficients_path, index=False)
        generalization = _write_generalization_artifacts(metrics_frame)
        _write_literature_context()
        review_frame = _build_simple_review_frame(generalization)
        write_review_rmse_overview(review_frame, RESULTS_DIR / "rmse_train_test_vs_literature.svg")
    else:
        generalization = pd.DataFrame()
        review_frame = pd.DataFrame()
    timing = _load_timing_results() if args.command in {"zinc-timing", "benchmark"} else pd.DataFrame()
    if not timing.empty:
        write_timing_stage_overview(timing, RESULTS_DIR / "timing_overview.svg")
    _write_results_csv(
        command=args.command,
        generated_at=datetime.now().strftime("%Y%m%d_%H%M%S"),
        review_frame=review_frame,
        timing=timing,
    )
    return 0


def _extend_with_property_results(
    artifacts: FreeSolvArtifacts | QM9Artifacts,
    metrics_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    coefficient_rows: list[dict[str, object]],
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
    bundles = [artifacts.smiles_export]
    if getattr(artifacts, "sdf_export", None) is not None:
        bundles.append(artifacts.sdf_export)
    for bundle in bundles:
        write_stan_data_json(bundle, student_df=config.student_df)
        for model_name in models:
            rows, predictions, coefficients = run_model_suite(bundle, model_name=model_name, config=config)
            metrics_rows.extend(rows)
            prediction_rows.extend(predictions)
            coefficient_rows.extend(coefficients)


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
    )
    for name in legacy_files:
        path = RESULTS_DIR / name
        if path.exists():
            path.unlink()
    for path in RESULTS_DIR.iterdir() if RESULTS_DIR.exists() else ():
        if path.is_dir() and (path.name == "stan_output" or path.name.startswith("review_")):
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


def _build_literature_context_rows(test_rows: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    freesolv = _best_test_row(test_rows, dataset="freesolv", metric="rmse")
    if freesolv is not None:
        rows.append(
            {
                "task": "FreeSolv",
                "local_result": (
                    f"RMSE {float(freesolv['rmse']):.4f} "
                    f"({freesolv['representation']}, {freesolv['model']}, {freesolv['method']}, test)"
                ),
                "literature_result": "RMSE 1.15 (MPNN, MoleculeNet Table 3 test subset); MoleculeNet also lists XGBoost at 1.74",
                "metric": "RMSE",
                "directly_comparable": "partial",
                "note": "Same benchmark family and error direction, but MoleculeNet uses its own random-split test subset while this repo run uses the local deterministic smoke split.",
                "source": f"[MoleculeNet](<{MOLECULENET_SOURCE_URL}>)",
            }
        )
    qm9 = _best_test_row(test_rows, dataset="qm9", metric="mae")
    if qm9 is not None:
        rows.append(
            {
                "task": "QM9 mu",
                "local_result": (
                    f"MAE {float(qm9['mae']):.4f}, RMSE {float(qm9['rmse']):.4f} "
                    f"({qm9['representation']}, {qm9['model']}, {qm9['method']}, test)"
                ),
                "literature_result": "Error ratio 0.30 for mu (Gilmer supplementary Table 2 at N=110k), equivalent to about 0.030 Debye MAE using the paper's 0.1 Debye chemical-accuracy table",
                "metric": "MAE (paper reports MAE-to-chemical-accuracy ratio)",
                "directly_comparable": "partial",
                "note": "Gilmer et al. use the full QM9 split with 110462 train / 10000 validation / 10000 test molecules and a 3D MPNN. This repo run uses a deterministic 2000-molecule subset and reports descriptor-based Bayesian baselines.",
                "source": f"[Gilmer et al.](<{GILMER_SOURCE_URL}>); [supplementary tables](<{GILMER_SUPPLEMENT_URL}>)",
            }
        )
    return rows


def _write_generalization_artifacts(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    generalization = _build_generalization_frame(metrics)
    if generalization.empty:
        return pd.DataFrame()
    generalization_metrics_path = _details_dir() / "generalization_metrics.csv"
    generalization.to_csv(generalization_metrics_path, index=False)
    return generalization


def _write_literature_context() -> pd.DataFrame:
    predictive_metrics_path = _details_dir() / "predictive_metrics.csv"
    if not predictive_metrics_path.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(predictive_metrics_path)
    test_rows = metrics.loc[metrics["split"] == "test"].copy()
    if test_rows.empty:
        return pd.DataFrame()
    literature_rows = _build_literature_context_rows(test_rows)
    if not literature_rows:
        return pd.DataFrame()
    frame = pd.DataFrame(literature_rows)
    frame.to_csv(_details_dir() / "literature_context.csv", index=False)
    return frame


def _build_simple_review_frame(generalization: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in generalization.iterrows():
        dataset = str(row["dataset"])
        representation = str(row["representation"])
        if dataset == "freesolv":
            literature_display = "MoleculeNet MPNN RMSE 1.15"
            literature_rmse = 1.15
            literature_metric = "RMSE"
            note = "Partial context only: MoleculeNet uses its own random split; this repo uses the local deterministic split."
            source = MOLECULENET_SOURCE_URL
        elif dataset == "qm9":
            literature_display = "Gilmer supplementary Table 2: mu ratio 0.30 at N=110k (about 0.030 Debye MAE)"
            literature_rmse = None
            literature_metric = "MAE ratio"
            note = "Partial context only: the paper reports a different metric and a much larger QM9 split."
            source = f"{GILMER_SOURCE_URL} ; {GILMER_SUPPLEMENT_URL}"
        else:
            literature_display = "No external context attached"
            literature_rmse = None
            literature_metric = ""
            note = "Local result only."
            source = ""
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
                "literature_display": literature_display,
                "literature_rmse": literature_rmse,
                "literature_metric": literature_metric,
                "directly_comparable": "partial",
                "note": note,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def _load_timing_results() -> pd.DataFrame:
    timing_path = _details_dir() / "zinc_timing.csv"
    if not timing_path.exists():
        return pd.DataFrame()
    return pd.read_csv(timing_path)


def _write_results_csv(
    *,
    command: str,
    generated_at: str,
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


if __name__ == "__main__":
    raise SystemExit(main())
