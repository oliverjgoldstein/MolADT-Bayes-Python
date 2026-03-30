from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from .benchmark_zinc import run_zinc_benchmark
from .common import DEFAULT_SEED, RESULTS_DIR, display_path, ensure_directory, render_markdown_table
from .process_freesolv import FreeSolvArtifacts, process_freesolv_dataset
from .process_qm9 import QM9Artifacts, process_qm9_dataset
from .report_graphs import write_predicted_vs_actual_overview, write_split_rmse_overview
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    if args.command == "smoke-test":
        artifacts = process_freesolv_dataset(seed=args.seed, force=args.force, include_sdf=not args.skip_sdf)
        _extend_with_property_results(artifacts, metrics_rows, prediction_rows, coefficient_rows, args)
    elif args.command == "qm9":
        artifacts = process_qm9_dataset(seed=args.seed, force=args.force, limit=args.limit)
        _extend_with_property_results(artifacts, metrics_rows, prediction_rows, coefficient_rows, args)
    elif args.command == "zinc-timing":
        run_zinc_benchmark(
            dataset_size=args.dataset_size,
            dataset_dimension=args.dataset_dimension,
            limit=args.limit,
            include_moladt=args.include_moladt,
            force=args.force,
        )
    elif args.command == "benchmark":
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
        )
    else:
        raise RuntimeError(f"Unsupported command {args.command}")

    if metrics_rows:
        ensure_directory(RESULTS_DIR)
        metrics_path = RESULTS_DIR / "predictive_metrics.csv"
        predictions_path = RESULTS_DIR / "predictions.csv"
        coefficients_path = RESULTS_DIR / "model_coefficients.csv"
        metrics_frame = pd.DataFrame(metrics_rows)
        predictions_frame = pd.DataFrame(prediction_rows)
        coefficients_frame = pd.DataFrame(coefficient_rows)
        metrics_frame.to_csv(metrics_path, index=False)
        predictions_frame.to_csv(predictions_path, index=False)
        coefficients_frame.to_csv(coefficients_path, index=False)
        _write_model_report(metrics_rows, coefficient_rows, args=args)
        _write_generalization_artifacts(metrics_frame, predictions_frame)
        _write_literature_context()
    _write_summary(include_predictive=bool(metrics_rows), include_zinc=args.command in {"zinc-timing", "benchmark"})
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


def _write_model_report(
    metrics_rows: list[dict[str, object]],
    coefficient_rows: list[dict[str, object]],
    *,
    args: argparse.Namespace,
) -> None:
    ensure_directory(RESULTS_DIR)
    metrics = pd.DataFrame(metrics_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    if metrics.empty or coefficients.empty:
        return

    sections: list[str] = [
        "# Fitted Model Report",
        "",
        "Predictors are standardized with train-split means and standard deviations only. Coefficients therefore multiply z-scored predictors, while the response stays in its original units.",
        "",
        "## Run Configuration",
        "",
        f"- command: `{args.command}`",
        f"- methods: `{args.methods}`",
        f"- models: `{args.models}`",
        f"- sample chains: `{args.sample_chains}`",
        f"- sample warmup: `{args.sample_warmup}`",
        f"- sample draws: `{args.sample_draws}`",
        f"- approximation draws: `{args.approximation_draws}`",
        f"- variational iterations: `{args.variational_iterations}`",
        f"- optimize iterations: `{args.optimize_iterations}`",
        f"- pathfinder paths: `{args.pathfinder_paths}`",
        f"- predictive draws: `{args.predictive_draws}`",
        "",
    ]

    grouped_metrics = metrics.groupby(["dataset", "representation", "model", "method"], sort=True)
    for (dataset, representation, model, method), run_metrics in grouped_metrics:
        coeff_subset = coefficients.loc[
            (coefficients["dataset"] == dataset)
            & (coefficients["representation"] == representation)
            & (coefficients["model"] == model)
            & (coefficients["method"] == method)
        ].copy()
        if coeff_subset.empty:
            continue
        target = str(coeff_subset["target"].iloc[0])
        sections.extend(
            [
                f"## {dataset} / {representation} / {model} / {method}",
                "",
                "### Equation",
                "",
                "```text",
                _equation_text(target=target, model_name=str(model), feature_names=tuple(coeff_subset.loc[coeff_subset["parameter_type"] == "coefficient", "parameter_name"].astype(str))),
                "```",
                "",
            ]
        )
        performance_rows = []
        for split_name in ("train", "valid", "test"):
            split_metrics = run_metrics.loc[run_metrics["split"] == split_name]
            if split_metrics.empty:
                continue
            row = split_metrics.iloc[0]
            performance_rows.append(
                [
                    row["split"],
                    row["n_train"],
                    row["n_eval"],
                    row["draw_count"],
                    row["runtime_seconds"],
                    row["rmse"],
                    row["mae"],
                    row["r2"],
                    row["mean_log_predictive_density"],
                    row["coverage_90"],
                ]
            )
        if performance_rows:
            sections.extend(
                [
                    "### Predictive Metrics",
                    "",
                    render_markdown_table(
                        headers=["split", "n_train", "n_eval", "draws", "runtime_s", "rmse", "mae", "r2", "mlpd", "coverage_90"],
                        rows=performance_rows,
                    ),
                    "",
                ]
            )

        scalar_rows = []
        scalar_subset = coeff_subset.loc[coeff_subset["parameter_type"].isin(["intercept", "noise_scale", "global_scale"])]
        for _, row in scalar_subset.iterrows():
            scalar_rows.append(
                [
                    row["parameter_name"],
                    row["posterior_mean"],
                    row["posterior_sd"],
                    row["posterior_median"],
                    row["posterior_p05"],
                    row["posterior_p95"],
                ]
            )
        if scalar_rows:
            sections.extend(
                [
                    "### Scalar Parameters",
                    "",
                    render_markdown_table(
                        headers=["parameter", "mean", "sd", "median", "p05", "p95"],
                        rows=scalar_rows,
                    ),
                    "",
                ]
            )

        group_scale_rows = []
        group_scale_subset = coeff_subset.loc[coeff_subset["parameter_type"] == "group_scale"].sort_values("feature_group")
        for _, row in group_scale_subset.iterrows():
            group_scale_rows.append(
                [
                    row["feature_group"],
                    row["posterior_mean"],
                    row["posterior_sd"],
                    row["posterior_median"],
                    row["posterior_p05"],
                    row["posterior_p95"],
                ]
            )
        if group_scale_rows:
            sections.extend(
                [
                    "### Group Scales",
                    "",
                    render_markdown_table(
                        headers=["group", "mean", "sd", "median", "p05", "p95"],
                        rows=group_scale_rows,
                    ),
                    "",
                ]
            )

        coefficient_table_rows = []
        coefficient_subset = coeff_subset.loc[coeff_subset["parameter_type"] == "coefficient"].sort_values(["importance_rank", "parameter_name"])
        for _, row in coefficient_subset.iterrows():
            coefficient_table_rows.append(
                [
                    row["importance_rank"],
                    row["parameter_name"],
                    row["feature_group"],
                    row["posterior_mean"],
                    row["posterior_sd"],
                    row["posterior_median"],
                    row["posterior_p05"],
                    row["posterior_p95"],
                ]
            )
        sections.extend(
            [
                "### Coefficients",
                "",
                render_markdown_table(
                    headers=["rank", "feature", "group", "mean", "sd", "median", "p05", "p95"],
                    rows=coefficient_table_rows,
                ),
                "",
            ]
        )
    (RESULTS_DIR / "model_report.md").write_text("\n".join(sections), encoding="utf-8")


def _equation_text(*, target: str, model_name: str, feature_names: tuple[str, ...]) -> str:
    predictor_terms = "\n  + ".join(f"beta[{feature_name}] * z({feature_name})" for feature_name in feature_names)
    lines = []
    if model_name == "bayes_hierarchical_shrinkage":
        lines.append("beta[k] = beta_raw[k] * global_scale * group_scale[group_id[k]]")
    lines.append("eta = alpha")
    if predictor_terms:
        lines.append(f"  + {predictor_terms}")
    lines.append(f"{target} ~ StudentT(nu=4.0, eta, sigma)")
    return "\n".join(lines)


def _write_summary(*, include_predictive: bool, include_zinc: bool) -> None:
    ensure_directory(RESULTS_DIR)
    sections: list[str] = [
        "# Benchmark Summary",
        "",
        "Local benchmark outputs from this repository are shown first. Literature context is kept separate and marked as partial unless split, subset size, metric, and units line up closely enough for a direct comparison.",
    ]
    coefficient_report_path = RESULTS_DIR / "model_report.md"
    coefficients_csv_path = RESULTS_DIR / "model_coefficients.csv"
    predictive_metrics_path = RESULTS_DIR / "predictive_metrics.csv"
    generalization_report_path = RESULTS_DIR / "generalization_report.md"
    generalization_metrics_path = RESULTS_DIR / "generalization_metrics.csv"
    split_rmse_graph_path = RESULTS_DIR / "split_rmse_overview.svg"
    predicted_vs_actual_graph_path = RESULTS_DIR / "predicted_vs_actual_overview.svg"
    if include_predictive and predictive_metrics_path.exists():
        metrics = pd.read_csv(predictive_metrics_path)
        test_rows = metrics.loc[metrics["split"] == "test"].copy()
        if not test_rows.empty:
            selected_rows = _select_reviewer_rows(test_rows)
            sections.extend(
                [
                    "",
                    "## Local Predictive Results",
                    "",
                    f"Reviewer-facing selection rule: the lowest test RMSE within each dataset/representation from this run. The full local grid remains in `{display_path(predictive_metrics_path)}`.",
                    "",
                ]
            )
            sections.append(
                render_markdown_table(
                    headers=[
                        "dataset",
                        "representation",
                        "model",
                        "method",
                        "rmse",
                        "mae",
                        "r2",
                        "mean_log_predictive_density",
                        "coverage_90",
                        "runtime_seconds",
                    ],
                    rows=[
                        [
                            row["dataset"],
                            row["representation"],
                            row["model"],
                            row["method"],
                            row["rmse"],
                            row["mae"],
                            row["r2"],
                            row["mean_log_predictive_density"],
                            row["coverage_90"],
                            row["runtime_seconds"],
                        ]
                        for _, row in selected_rows.iterrows()
                    ],
                )
            )
            split_notes = _split_note_lines(test_rows)
            if split_notes:
                sections.extend(["", *split_notes])
            literature_rows = _build_literature_context_rows(test_rows)
            if literature_rows:
                sections.extend(
                    [
                        "",
                        "## Tiny Literature Context",
                        "",
                        f"These rows are context only. Full source links and notes are in `{display_path(RESULTS_DIR / 'literature_context.md')}`.",
                        "",
                        render_markdown_table(
                            headers=["task", "literature_result", "directly_comparable", "note"],
                            rows=[
                                [
                                    row["task"],
                                    row["literature_result"],
                                    row["directly_comparable"],
                                    row["note"],
                                ]
                                for row in literature_rows
                            ],
                        ),
                    ]
                )
            if generalization_metrics_path.exists():
                generalization = pd.read_csv(generalization_metrics_path)
                if not generalization.empty:
                    sections.extend(
                        [
                            "",
                            "## Train vs Test Overview",
                            "",
                            f"Positive `test_minus_train_rmse` values mean the held-out error is worse than the training error. Full detail: `{display_path(generalization_report_path)}` and `{display_path(generalization_metrics_path)}`.",
                            "",
                            render_markdown_table(
                                headers=[
                                    "dataset",
                                    "representation",
                                    "model",
                                    "method",
                                    "train_rmse",
                                    "valid_rmse",
                                    "test_rmse",
                                    "test_minus_train_rmse",
                                    "train_mae",
                                    "test_mae",
                                    "train_r2",
                                    "test_r2",
                                ],
                                rows=[
                                    [
                                        row["dataset"],
                                        row["representation"],
                                        row["model"],
                                        row["method"],
                                        row["train_rmse"],
                                        row["valid_rmse"],
                                        row["test_rmse"],
                                        row["test_minus_train_rmse"],
                                        row["train_mae"],
                                        row["test_mae"],
                                        row["train_r2"],
                                        row["test_r2"],
                                    ]
                                    for _, row in generalization.iterrows()
                                ],
                            ),
                        ]
                    )
                    if split_rmse_graph_path.exists():
                        sections.extend(["", "![Split RMSE overview](split_rmse_overview.svg)"])
                    if predicted_vs_actual_graph_path.exists():
                        sections.extend(["", "![Predicted versus actual overview](predicted_vs_actual_overview.svg)"])
    zinc_timing_path = RESULTS_DIR / "zinc_timing.csv"
    if include_zinc and zinc_timing_path.exists():
        timing = pd.read_csv(zinc_timing_path)
        sections.extend(["", "## Local ZINC Timing", ""])
        sections.append(
            render_markdown_table(
                headers=[
                    "stage",
                    "molecule_count",
                    "success_count",
                    "failure_count",
                    "total_runtime_seconds",
                    "molecules_per_second",
                    "median_latency_us",
                    "p95_latency_us",
                    "peak_rss_mb",
                ],
                rows=[
                    [
                        row["stage"],
                        row["molecule_count"],
                        row["success_count"],
                        row["failure_count"],
                        row["total_runtime_seconds"],
                        row["molecules_per_second"],
                        row["median_latency_us"],
                        row["p95_latency_us"],
                        row["peak_rss_mb"],
                    ]
                    for _, row in timing.iterrows()
                ],
            )
        )
    if include_predictive and (coefficient_report_path.exists() or coefficients_csv_path.exists()):
        sections.extend(
            [
                "",
                "## Fitted Model Outputs",
                "",
                f"- `{display_path(coefficient_report_path)}` contains the fitted equations, predictive metrics, and posterior summaries.",
                f"- `{display_path(coefficients_csv_path)}` contains the full posterior summary table for coefficients and scales.",
            ]
        )
    (RESULTS_DIR / "summary.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


def _select_reviewer_rows(test_rows: pd.DataFrame) -> pd.DataFrame:
    ranked = test_rows.sort_values(["dataset", "representation", "rmse", "mae", "runtime_seconds", "model", "method"])
    return ranked.groupby(["dataset", "representation"], sort=True, as_index=False).head(1)


def _split_note_lines(test_rows: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    for dataset in ("freesolv", "qm9"):
        dataset_rows = test_rows.loc[test_rows["dataset"] == dataset]
        if dataset_rows.empty:
            continue
        reference = dataset_rows.sort_values(["n_train", "n_eval", "representation", "model", "method"]).iloc[0]
        if dataset == "freesolv":
            notes.append(
                f"- FreeSolv local split in this run: {int(reference['n_train'])} train / {int(reference['n_eval'])} test molecules."
            )
        elif dataset == "qm9":
            notes.append(
                f"- QM9 local split in this run: {int(reference['n_train'])} train / {int(reference['n_eval'])} test molecules from the repo's benchmark subset."
            )
    return notes


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
                "literature_result": "Error ratio 0.20 for mu (enn-s2s-ens5, Gilmer Table 2), equivalent to about 0.020 Debye MAE using the paper's 0.1 Debye chemical-accuracy table",
                "metric": "MAE (paper reports MAE-to-chemical-accuracy ratio)",
                "directly_comparable": "partial",
                "note": "Gilmer et al. use the full QM9 split with 110462 train / 10000 validation / 10000 test molecules and a 3D MPNN. This repo run uses a deterministic 2000-molecule subset and reports descriptor-based Bayesian baselines.",
                "source": f"[Gilmer et al.](<{GILMER_SOURCE_URL}>); [supplementary tables](<{GILMER_SUPPLEMENT_URL}>)",
            }
        )
    return rows


def _write_generalization_artifacts(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    if metrics.empty or predictions.empty:
        return
    generalization = _build_generalization_frame(metrics)
    if generalization.empty:
        return
    generalization_metrics_path = RESULTS_DIR / "generalization_metrics.csv"
    generalization_report_path = RESULTS_DIR / "generalization_report.md"
    split_rmse_graph_path = RESULTS_DIR / "split_rmse_overview.svg"
    predicted_vs_actual_graph_path = RESULTS_DIR / "predicted_vs_actual_overview.svg"
    generalization.to_csv(generalization_metrics_path, index=False)

    key_frame = generalization.loc[:, ["dataset", "representation", "model", "method"]].drop_duplicates()
    selected_metrics = metrics.merge(key_frame, on=["dataset", "representation", "model", "method"], how="inner")
    selected_predictions = predictions.merge(key_frame, on=["dataset", "representation", "model", "method"], how="inner")
    write_split_rmse_overview(selected_metrics, split_rmse_graph_path)
    write_predicted_vs_actual_overview(selected_predictions, predicted_vs_actual_graph_path)

    sections = [
        "# Train vs Test Overview",
        "",
        "These rows follow the same reviewer-facing selection rule as `summary.md`: the lowest test RMSE within each dataset/representation from this run.",
        "",
        f"- numeric table: `{display_path(generalization_metrics_path)}`",
        f"- RMSE graph: `{split_rmse_graph_path.name}`",
        f"- predicted-vs-actual graph: `{predicted_vs_actual_graph_path.name}`",
        "",
        render_markdown_table(
            headers=[
                "dataset",
                "representation",
                "model",
                "method",
                "train_rmse",
                "valid_rmse",
                "test_rmse",
                "test_minus_train_rmse",
                "train_mae",
                "test_mae",
                "train_r2",
                "test_r2",
                "fit_runtime_seconds",
            ],
            rows=[
                [
                    row["dataset"],
                    row["representation"],
                    row["model"],
                    row["method"],
                    row["train_rmse"],
                    row["valid_rmse"],
                    row["test_rmse"],
                    row["test_minus_train_rmse"],
                    row["train_mae"],
                    row["test_mae"],
                    row["train_r2"],
                    row["test_r2"],
                    row["fit_runtime_seconds"],
                ]
                for _, row in generalization.iterrows()
            ],
        ),
        "",
        "## Graphs",
        "",
        "![Split RMSE overview](split_rmse_overview.svg)",
        "",
        "![Predicted versus actual overview](predicted_vs_actual_overview.svg)",
        "",
    ]
    generalization_report_path.write_text("\n".join(sections), encoding="utf-8")


def _write_literature_context() -> None:
    predictive_metrics_path = RESULTS_DIR / "predictive_metrics.csv"
    if not predictive_metrics_path.exists():
        return
    metrics = pd.read_csv(predictive_metrics_path)
    test_rows = metrics.loc[metrics["split"] == "test"].copy()
    if test_rows.empty:
        return
    literature_rows = _build_literature_context_rows(test_rows)
    if not literature_rows:
        return
    sections = [
        "# Literature Context",
        "",
        "This file keeps literature context separate from the local repo outputs. A row is only directly comparable when split, subset size, metric, and units align closely enough; otherwise it is marked as partial context.",
        "",
        render_markdown_table(
            headers=["task", "local_result", "literature_result", "metric", "directly_comparable", "note", "source"],
            rows=[
                [
                    row["task"],
                    row["local_result"],
                    row["literature_result"],
                    row["metric"],
                    row["directly_comparable"],
                    row["note"],
                    row["source"],
                ]
                for row in literature_rows
            ],
        ),
        "",
    ]
    (RESULTS_DIR / "literature_context.md").write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
