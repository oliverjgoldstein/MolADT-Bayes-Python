from __future__ import annotations

import argparse

import pandas as pd

from .benchmark_zinc import run_zinc_benchmark
from .common import DEFAULT_SEED, RESULTS_DIR, ensure_directory, render_markdown_table
from .process_freesolv import FreeSolvArtifacts, process_freesolv_dataset
from .process_qm9 import QM9Artifacts, process_qm9_dataset
from .stan_runner import ALL_METHODS, StanRunConfig, run_model_suite, write_stan_data_json


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
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
        pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
        pd.DataFrame(coefficient_rows).to_csv(coefficients_path, index=False)
        _write_model_report(metrics_rows, coefficient_rows, args=args)
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
        for split_name in ("valid", "test"):
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
    sections: list[str] = ["# Benchmark Summary"]
    coefficient_report_path = RESULTS_DIR / "model_report.md"
    coefficients_csv_path = RESULTS_DIR / "model_coefficients.csv"
    if include_predictive and (coefficient_report_path.exists() or coefficients_csv_path.exists()):
        sections.extend(
            [
                "",
                "## Fitted Model Outputs",
                "",
                "- `results/model_report.md` contains the fitted equations, predictive metrics, and posterior summaries.",
                "- `results/model_coefficients.csv` contains the full posterior summary table for coefficients and scales.",
            ]
        )
    predictive_metrics_path = RESULTS_DIR / "predictive_metrics.csv"
    if include_predictive and predictive_metrics_path.exists():
        metrics = pd.read_csv(predictive_metrics_path)
        test_rows = metrics.loc[metrics["split"] == "test"].copy()
        if not test_rows.empty:
            test_rows = test_rows.sort_values(["dataset", "representation", "model", "method"])
            sections.append("\n## Predictive Benchmarks\n")
            sections.append(
                render_markdown_table(
                    headers=["dataset", "repr", "model", "method", "rmse", "mae", "r2", "mlpd", "coverage_90", "runtime_s"],
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
                        for _, row in test_rows.iterrows()
                    ],
                )
            )
    zinc_timing_path = RESULTS_DIR / "zinc_timing.csv"
    if include_zinc and zinc_timing_path.exists():
        timing = pd.read_csv(zinc_timing_path)
        sections.append("\n## ZINC Timing\n")
        sections.append(
            render_markdown_table(
                headers=["stage", "count", "success", "failure", "runtime_s", "mol_per_s", "median_us", "p95_us", "peak_rss_mb"],
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
    (RESULTS_DIR / "summary.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
