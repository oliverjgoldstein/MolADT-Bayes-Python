from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

from .common import DEFAULT_SEED, LOCAL_CMDSTAN_DIR, PROJECT_ROOT, RESULTS_DIR, display_path, ensure_directory, log, write_json
from .predictive_metrics import build_metric_row, build_prediction_rows
from .splits import ExportedDataset
from .toolchain import cmdstan_build_environment

MODEL_FILES = {
    "bayes_linear_student_t": PROJECT_ROOT / "stan" / "bayes_linear_student_t.stan",
    "bayes_hierarchical_shrinkage": PROJECT_ROOT / "stan" / "bayes_hierarchical_shrinkage.stan",
    "bayes_gp_rbf_screened": PROJECT_ROOT / "stan" / "bayes_gp_rbf_screened.stan",
}
ALL_METHODS = ("sample", "variational", "pathfinder", "optimize", "laplace")
GP_MODEL_NAMES = frozenset({"bayes_gp_rbf_screened"})
GP_ALLOWED_METHODS = ("variational", "pathfinder", "optimize", "laplace")
GP_SCREENED_FEATURE_COUNT = 24


@dataclass(frozen=True, slots=True)
class StanRunConfig:
    methods: tuple[str, ...] = ALL_METHODS
    seed: int = DEFAULT_SEED
    student_df: float = 4.0
    sample_chains: int = 2
    sample_warmup: int = 200
    sample_draws: int = 200
    approximation_draws: int = 500
    variational_iterations: int = 5000
    optimize_iterations: int = 2000
    pathfinder_paths: int = 4
    predictive_draws: int = 500
    verbose: bool = False


def ensure_cmdstan_ready() -> None:
    import cmdstanpy

    candidates = sorted(LOCAL_CMDSTAN_DIR.glob("cmdstan-*"))
    if candidates:
        cmdstanpy.set_cmdstan_path(str(candidates[-1]))
    try:
        cmdstanpy.cmdstan_path()
    except Exception as exc:
        raise RuntimeError(
            "CmdStan is not installed. From the repo root, run `make python-cmdstan-install` once, "
            "or install it manually with "
            "`python -c \"import cmdstanpy; cmdstanpy.install_cmdstan(dir='MolADT-Bayes-Python/.cmdstan')\"`, "
            "or point CmdStanPy at an existing installation via set_cmdstan_path()."
        ) from exc


def build_stan_data(bundle: ExportedDataset, *, student_df: float) -> dict[str, Any]:
    X_eval = np.vstack([bundle.X_valid, bundle.X_test])
    y_mean = float(np.mean(bundle.y_train))
    y_scale = max(float(np.std(bundle.y_train)), 1e-3)
    return {
        "N": int(bundle.X_train.shape[0]),
        "K": int(bundle.X_train.shape[1]),
        "X": bundle.X_train,
        "y": bundle.y_train,
        "N_eval": int(X_eval.shape[0]),
        "X_eval": X_eval,
        "G": len(bundle.group_names),
        "group_id": np.array(bundle.group_ids, dtype=int),
        "nu": float(student_df),
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def build_gp_stan_data(bundle: ExportedDataset) -> dict[str, Any]:
    y_mean = float(np.mean(bundle.y_train))
    y_scale = max(float(np.std(bundle.y_train)), 1e-3)
    return {
        "N": int(bundle.X_train.shape[0]),
        "K": int(bundle.X_train.shape[1]),
        "X": bundle.X_train,
        "y": bundle.y_train,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def write_stan_data_json(bundle: ExportedDataset, *, student_df: float) -> Path:
    payload = _json_ready(build_stan_data(bundle, student_df=student_df))
    target = bundle.metadata_path.with_name(bundle.metadata_path.stem.replace("_metadata", "_stan_data") + ".json")
    return write_json(target, payload)


def run_model_suite(
    bundle: ExportedDataset,
    *,
    model_name: str,
    config: StanRunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import cmdstanpy

    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model {model_name}")
    if model_name in GP_MODEL_NAMES:
        return _run_gp_model_suite(bundle, model_name=model_name, config=config)
    ensure_cmdstan_ready()
    output_dir = ensure_directory(RESULTS_DIR / "details" / "stan_output" / bundle.dataset_name / bundle.representation / model_name)
    data = build_stan_data(bundle, student_df=config.student_df)
    with cmdstan_build_environment(verbose=config.verbose):
        model = cmdstanpy.CmdStanModel(stan_file=str(MODEL_FILES[model_name]))
    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    total_methods = len(config.methods)
    for method_index, method in enumerate(config.methods, start=1):
        if config.verbose:
            log(
                f"[stan {method_index}/{total_methods}] "
                f"{bundle.dataset_name}/{bundle.representation}/{model_name}/{method} "
                f"output={display_path(output_dir / method)}"
            )
        fit, runtime_seconds = _run_method(model, data, method=method, config=config, output_dir=output_dir)
        if config.verbose:
            log(
                f"[stan {method_index}/{total_methods}] "
                f"{bundle.dataset_name}/{bundle.representation}/{model_name}/{method} "
                f"runtime_s={runtime_seconds:.2f}"
            )
        summary_rows, prediction_rows, coefficient_rows = _evaluate_fit(
            bundle,
            model_name=model_name,
            method=method,
            fit=fit,
            runtime_seconds=runtime_seconds,
            config=config,
        )
        results.extend(summary_rows)
        predictions.extend(prediction_rows)
        coefficients.extend(coefficient_rows)
    return results, predictions, coefficients


def _run_gp_model_suite(
    bundle: ExportedDataset,
    *,
    model_name: str,
    config: StanRunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import cmdstanpy

    ensure_cmdstan_ready()
    screened_bundle = _screen_bundle_top_correlation_features(bundle, top_k=GP_SCREENED_FEATURE_COUNT)
    output_dir = ensure_directory(
        RESULTS_DIR / "details" / "stan_output" / screened_bundle.dataset_name / screened_bundle.representation / model_name
    )
    data = build_gp_stan_data(screened_bundle)
    methods = tuple(method for method in config.methods if method in GP_ALLOWED_METHODS)
    if not methods:
        raise ValueError(
            f"{model_name} supports only {', '.join(GP_ALLOWED_METHODS)}; "
            f"received {', '.join(config.methods) or '(none)'}"
        )
    skipped_methods = [method for method in config.methods if method not in GP_ALLOWED_METHODS]
    if skipped_methods and config.verbose:
        log(
            f"[{screened_bundle.dataset_name}/{screened_bundle.representation}] "
            f"Skipping unsupported GP methods: {', '.join(skipped_methods)}"
        )
    with cmdstan_build_environment(verbose=config.verbose):
        model = cmdstanpy.CmdStanModel(stan_file=str(MODEL_FILES[model_name]))
    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    total_methods = len(methods)
    for method_index, method in enumerate(methods, start=1):
        if config.verbose:
            log(
                f"[stan {method_index}/{total_methods}] "
                f"{screened_bundle.dataset_name}/{screened_bundle.representation}/{model_name}/{method} "
                f"output={display_path(output_dir / method)}"
            )
        fit, runtime_seconds = _run_method(model, data, method=method, config=config, output_dir=output_dir)
        if config.verbose:
            log(
                f"[stan {method_index}/{total_methods}] "
                f"{screened_bundle.dataset_name}/{screened_bundle.representation}/{model_name}/{method} "
                f"runtime_s={runtime_seconds:.2f}"
            )
        summary_rows, prediction_rows, coefficient_rows = _evaluate_gp_fit(
            screened_bundle,
            model_name=model_name,
            method=method,
            fit=fit,
            runtime_seconds=runtime_seconds,
            config=config,
        )
        results.extend(summary_rows)
        predictions.extend(prediction_rows)
        coefficients.extend(coefficient_rows)
    return results, predictions, coefficients


def _run_method(model: Any, data: dict[str, Any], *, method: str, config: StanRunConfig, output_dir: Path) -> tuple[Any, float]:
    start = time.perf_counter()
    if method == "sample":
        fit = model.sample(
            data=data,
            chains=config.sample_chains,
            parallel_chains=config.sample_chains,
            iter_warmup=config.sample_warmup,
            iter_sampling=config.sample_draws,
            adapt_delta=0.95,
            seed=config.seed,
            show_progress=config.verbose,
            show_console=config.verbose,
            refresh=50 if config.verbose else None,
            output_dir=str(output_dir / method),
        )
    elif method == "variational":
        fit = model.variational(
            data=data,
            seed=config.seed,
            algorithm="fullrank",
            iter=config.variational_iterations,
            draws=config.approximation_draws,
            require_converged=False,
            show_console=config.verbose,
            refresh=100 if config.verbose else None,
            output_dir=str(output_dir / method),
        )
    elif method == "pathfinder":
        fit = model.pathfinder(
            data=data,
            seed=config.seed,
            num_paths=config.pathfinder_paths,
            draws=config.approximation_draws,
            num_elbo_draws=max(50, min(config.approximation_draws, 500)),
            num_single_draws=max(50, min(config.approximation_draws, 500)),
            show_console=config.verbose,
            refresh=100 if config.verbose else None,
            output_dir=str(output_dir / method),
        )
    elif method == "optimize":
        fit = model.optimize(
            data=data,
            seed=config.seed,
            algorithm="lbfgs",
            iter=config.optimize_iterations,
            jacobian=False,
            show_console=config.verbose,
            refresh=100 if config.verbose else None,
            output_dir=str(output_dir / method),
        )
    elif method == "laplace":
        mode = model.optimize(
            data=data,
            seed=config.seed,
            algorithm="lbfgs",
            iter=config.optimize_iterations,
            jacobian=False,
            show_console=config.verbose,
            refresh=100 if config.verbose else None,
            output_dir=str(output_dir / "laplace_optimize"),
        )
        fit = model.laplace_sample(
            data=data,
            mode=mode,
            draws=config.approximation_draws,
            jacobian=False,
            seed=config.seed,
            show_console=config.verbose,
            refresh=100 if config.verbose else None,
            output_dir=str(output_dir / method),
        )
    else:
        raise ValueError(f"Unsupported inference method {method}")
    return fit, time.perf_counter() - start


def _evaluate_fit(
    bundle: ExportedDataset,
    *,
    model_name: str,
    method: str,
    fit: Any,
    runtime_seconds: float,
    config: StanRunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    alpha = _ensure_draw_array(_stan_variable_draws(fit, "alpha"), ndim=1)
    beta = _ensure_draw_array(_stan_variable_draws(fit, "beta"), ndim=2)
    sigma = _ensure_draw_array(_stan_variable_draws(fit, "sigma"), ndim=1)
    global_scale = _optional_draw_array(fit, "global_scale", ndim=1)
    group_scale = _optional_draw_array(fit, "group_scale", ndim=2)
    finite_mask = np.isfinite(alpha) & np.isfinite(sigma) & np.all(np.isfinite(beta), axis=1)
    if global_scale is not None:
        finite_mask &= np.isfinite(global_scale)
    if group_scale is not None:
        finite_mask &= np.all(np.isfinite(group_scale), axis=1)
    if not np.any(finite_mask):
        raise RuntimeError(f"No finite posterior draws were returned for {model_name}/{method}")
    alpha = alpha[finite_mask]
    beta = beta[finite_mask]
    sigma = sigma[finite_mask]
    if global_scale is not None:
        global_scale = global_scale[finite_mask]
    if group_scale is not None:
        group_scale = group_scale[finite_mask]
    alpha, beta, sigma, global_scale, group_scale = _filter_draws_for_design_matrix(
        X=bundle.X_valid,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        global_scale=global_scale,
        group_scale=group_scale,
    )
    alpha, beta, sigma, global_scale, group_scale = _filter_draws_for_design_matrix(
        X=bundle.X_test,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        global_scale=global_scale,
        group_scale=group_scale,
    )
    train_metrics, train_predictions = _evaluate_split(
        X=bundle.X_train,
        y=bundle.y_train,
        mol_ids=bundle.mol_ids_train,
        dataset_name=bundle.dataset_name,
        representation=bundle.representation,
        model_name=model_name,
        method=method,
        split_name="train",
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        student_df=config.student_df,
        runtime_seconds=runtime_seconds,
        feature_count=len(bundle.feature_names),
        n_train=len(bundle.y_train),
        split_scheme=bundle.split_scheme,
        source_row_count=bundle.source_row_count,
        used_row_count=bundle.used_row_count,
        seed=config.seed + 11,
        predictive_draws=config.predictive_draws,
    )
    valid_metrics, valid_predictions = _evaluate_split(
        X=bundle.X_valid,
        y=bundle.y_valid,
        mol_ids=bundle.mol_ids_valid,
        dataset_name=bundle.dataset_name,
        representation=bundle.representation,
        model_name=model_name,
        method=method,
        split_name="valid",
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        student_df=config.student_df,
        runtime_seconds=runtime_seconds,
        feature_count=len(bundle.feature_names),
        n_train=len(bundle.y_train),
        split_scheme=bundle.split_scheme,
        source_row_count=bundle.source_row_count,
        used_row_count=bundle.used_row_count,
        seed=config.seed + 17,
        predictive_draws=config.predictive_draws,
    )
    test_metrics, test_predictions = _evaluate_split(
        X=bundle.X_test,
        y=bundle.y_test,
        mol_ids=bundle.mol_ids_test,
        dataset_name=bundle.dataset_name,
        representation=bundle.representation,
        model_name=model_name,
        method=method,
        split_name="test",
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        student_df=config.student_df,
        runtime_seconds=runtime_seconds,
        feature_count=len(bundle.feature_names),
        n_train=len(bundle.y_train),
        split_scheme=bundle.split_scheme,
        source_row_count=bundle.source_row_count,
        used_row_count=bundle.used_row_count,
        seed=config.seed + 29,
        predictive_draws=config.predictive_draws,
    )
    coefficient_rows = _summarize_parameters(
        bundle=bundle,
        model_name=model_name,
        method=method,
        runtime_seconds=runtime_seconds,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        global_scale=global_scale,
        group_scale=group_scale,
    )
    return [train_metrics, valid_metrics, test_metrics], [*train_predictions, *valid_predictions, *test_predictions], coefficient_rows


def _evaluate_gp_fit(
    bundle: ExportedDataset,
    *,
    model_name: str,
    method: str,
    fit: Any,
    runtime_seconds: float,
    config: StanRunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    alpha = _ensure_draw_array(_stan_variable_draws(fit, "alpha"), ndim=1)
    signal_scale = _ensure_draw_array(_stan_variable_draws(fit, "signal_scale"), ndim=1)
    lengthscale = _ensure_draw_array(_stan_variable_draws(fit, "lengthscale"), ndim=1)
    sigma = _ensure_draw_array(_stan_variable_draws(fit, "sigma"), ndim=1)
    finite_mask = (
        np.isfinite(alpha)
        & np.isfinite(signal_scale)
        & np.isfinite(lengthscale)
        & np.isfinite(sigma)
        & (signal_scale > 0.0)
        & (lengthscale > 0.0)
        & (sigma > 0.0)
    )
    if not np.any(finite_mask):
        raise RuntimeError(f"No finite GP posterior draws were returned for {model_name}/{method}")
    alpha = alpha[finite_mask]
    signal_scale = signal_scale[finite_mask]
    lengthscale = lengthscale[finite_mask]
    sigma = sigma[finite_mask]

    train_metrics, train_predictions = _evaluate_gp_split(
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_eval=bundle.X_train,
        y_eval=bundle.y_train,
        mol_ids=bundle.mol_ids_train,
        dataset_name=bundle.dataset_name,
        representation=bundle.representation,
        model_name=model_name,
        method=method,
        split_name="train",
        alpha=alpha,
        signal_scale=signal_scale,
        lengthscale=lengthscale,
        sigma=sigma,
        runtime_seconds=runtime_seconds,
        feature_count=len(bundle.feature_names),
        n_train=len(bundle.y_train),
        split_scheme=bundle.split_scheme,
        source_row_count=bundle.source_row_count,
        used_row_count=bundle.used_row_count,
        seed=config.seed + 11,
        predictive_draws=config.predictive_draws,
    )
    valid_metrics, valid_predictions = _evaluate_gp_split(
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_eval=bundle.X_valid,
        y_eval=bundle.y_valid,
        mol_ids=bundle.mol_ids_valid,
        dataset_name=bundle.dataset_name,
        representation=bundle.representation,
        model_name=model_name,
        method=method,
        split_name="valid",
        alpha=alpha,
        signal_scale=signal_scale,
        lengthscale=lengthscale,
        sigma=sigma,
        runtime_seconds=runtime_seconds,
        feature_count=len(bundle.feature_names),
        n_train=len(bundle.y_train),
        split_scheme=bundle.split_scheme,
        source_row_count=bundle.source_row_count,
        used_row_count=bundle.used_row_count,
        seed=config.seed + 17,
        predictive_draws=config.predictive_draws,
    )
    test_metrics, test_predictions = _evaluate_gp_split(
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_eval=bundle.X_test,
        y_eval=bundle.y_test,
        mol_ids=bundle.mol_ids_test,
        dataset_name=bundle.dataset_name,
        representation=bundle.representation,
        model_name=model_name,
        method=method,
        split_name="test",
        alpha=alpha,
        signal_scale=signal_scale,
        lengthscale=lengthscale,
        sigma=sigma,
        runtime_seconds=runtime_seconds,
        feature_count=len(bundle.feature_names),
        n_train=len(bundle.y_train),
        split_scheme=bundle.split_scheme,
        source_row_count=bundle.source_row_count,
        used_row_count=bundle.used_row_count,
        seed=config.seed + 29,
        predictive_draws=config.predictive_draws,
    )
    coefficient_rows = _summarize_gp_parameters(
        bundle=bundle,
        model_name=model_name,
        method=method,
        runtime_seconds=runtime_seconds,
        alpha=alpha,
        signal_scale=signal_scale,
        lengthscale=lengthscale,
        sigma=sigma,
    )
    return [train_metrics, valid_metrics, test_metrics], [*train_predictions, *valid_predictions, *test_predictions], coefficient_rows


def _evaluate_gp_split(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    mol_ids: tuple[str, ...],
    dataset_name: str,
    representation: str,
    model_name: str,
    method: str,
    split_name: str,
    alpha: np.ndarray,
    signal_scale: np.ndarray,
    lengthscale: np.ndarray,
    sigma: np.ndarray,
    runtime_seconds: float,
    feature_count: int,
    n_train: int,
    split_scheme: str,
    source_row_count: int,
    used_row_count: int,
    seed: int,
    predictive_draws: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mean_by_draw, sd_by_draw = _gp_predictive_draws(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        alpha=alpha,
        signal_scale=signal_scale,
        lengthscale=lengthscale,
        sigma=sigma,
    )
    predictive_mean = np.mean(mean_by_draw, axis=0)
    predictive_var = np.mean(np.square(sd_by_draw) + np.square(mean_by_draw), axis=0) - np.square(predictive_mean)
    predictive_sd = np.sqrt(np.clip(predictive_var, 1e-9, None))
    metrics = build_metric_row(
        dataset_name=dataset_name,
        representation=representation,
        model_name=model_name,
        method=method,
        split_name=split_name,
        mol_ids=mol_ids,
        actual=y_eval,
        predicted_mean=predictive_mean,
        predictive_sd=predictive_sd,
        runtime_seconds=runtime_seconds,
        feature_count=feature_count,
        n_train=n_train,
        split_scheme=split_scheme,
        source_row_count=source_row_count,
        used_row_count=used_row_count,
        seed=seed,
        draw_count=int(mean_by_draw.shape[0]),
        parameter_count=4,
        extra_metrics={"screened_feature_count": float(feature_count)},
    )
    predictions = build_prediction_rows(
        dataset_name=dataset_name,
        representation=representation,
        model_name=model_name,
        method=method,
        split_name=split_name,
        mol_ids=mol_ids,
        actual=y_eval,
        predicted_mean=predictive_mean,
        predictive_sd=predictive_sd,
        seed=seed,
    )
    return metrics, predictions


def _gp_predictive_draws(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    alpha: np.ndarray,
    signal_scale: np.ndarray,
    lengthscale: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_count = X_train.shape[0]
    eval_count = X_eval.shape[0]
    means: list[np.ndarray] = []
    sds: list[np.ndarray] = []
    centered_cache = y_train
    for draw_alpha, draw_signal, draw_length, draw_sigma in zip(alpha, signal_scale, lengthscale, sigma, strict=True):
        train_kernel = _rbf_kernel(X_train, X_train, lengthscale=draw_length, signal_scale=draw_signal)
        train_kernel[np.diag_indices_from(train_kernel)] += draw_sigma**2 + 1e-8
        try:
            chol = np.linalg.cholesky(train_kernel)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("GP covariance was not numerically positive definite") from exc
        centered_y = centered_cache - draw_alpha
        weight = np.linalg.solve(chol.T, np.linalg.solve(chol, centered_y))
        cross_kernel = _rbf_kernel(X_eval, X_train, lengthscale=draw_length, signal_scale=draw_signal)
        mean = draw_alpha + cross_kernel @ weight
        solve = np.linalg.solve(chol, cross_kernel.T)
        marginal_var = draw_signal**2 + draw_sigma**2 - np.sum(np.square(solve), axis=0)
        means.append(mean.reshape(eval_count))
        sds.append(np.sqrt(np.clip(marginal_var, 1e-9, None)).reshape(eval_count))
    return np.vstack(means), np.vstack(sds)


def _rbf_kernel(X_left: np.ndarray, X_right: np.ndarray, *, lengthscale: float, signal_scale: float) -> np.ndarray:
    left_sq = np.sum(np.square(X_left), axis=1)[:, np.newaxis]
    right_sq = np.sum(np.square(X_right), axis=1)[np.newaxis, :]
    sqdist = np.maximum(left_sq + right_sq - 2.0 * (X_left @ X_right.T), 0.0)
    return (signal_scale**2) * np.exp(-0.5 * sqdist / max(lengthscale**2, 1e-9))


def _screen_bundle_top_correlation_features(bundle: ExportedDataset, *, top_k: int) -> ExportedDataset:
    feature_count = min(int(top_k), bundle.X_train.shape[1])
    if feature_count <= 0:
        raise ValueError("GP feature screening requires at least one feature")
    y_centered = bundle.y_train - np.mean(bundle.y_train)
    scores: list[tuple[float, int]] = []
    for feature_index in range(bundle.X_train.shape[1]):
        column = bundle.X_train[:, feature_index]
        denominator = float(np.linalg.norm(column) * np.linalg.norm(y_centered))
        score = 0.0 if denominator == 0.0 else abs(float(np.dot(column, y_centered) / denominator))
        scores.append((score, feature_index))
    selected_indices = tuple(sorted(index for _, index in sorted(scores, reverse=True)[:feature_count]))
    selected_feature_names = tuple(bundle.feature_names[index] for index in selected_indices)
    selected_feature_groups = {name: bundle.feature_groups[name] for name in selected_feature_names}
    group_names = tuple(dict.fromkeys(selected_feature_groups[name] for name in selected_feature_names))
    group_ids = tuple(group_names.index(selected_feature_groups[name]) + 1 for name in selected_feature_names)
    return ExportedDataset(
        dataset_name=bundle.dataset_name,
        representation=bundle.representation,
        target_name=bundle.target_name,
        split_scheme=bundle.split_scheme,
        source_row_count=bundle.source_row_count,
        used_row_count=bundle.used_row_count,
        feature_names=selected_feature_names,
        feature_groups=selected_feature_groups,
        group_names=group_names,
        group_ids=group_ids,
        rows=bundle.rows,
        X_train=bundle.X_train[:, selected_indices],
        X_valid=bundle.X_valid[:, selected_indices],
        X_test=bundle.X_test[:, selected_indices],
        y_train=bundle.y_train,
        y_valid=bundle.y_valid,
        y_test=bundle.y_test,
        mol_ids_train=bundle.mol_ids_train,
        mol_ids_valid=bundle.mol_ids_valid,
        mol_ids_test=bundle.mol_ids_test,
        metadata_path=bundle.metadata_path,
        feature_csv_path=bundle.feature_csv_path,
    )


def _summarize_gp_parameters(
    *,
    bundle: ExportedDataset,
    model_name: str,
    method: str,
    runtime_seconds: float,
    alpha: np.ndarray,
    signal_scale: np.ndarray,
    lengthscale: np.ndarray,
    sigma: np.ndarray,
) -> list[dict[str, Any]]:
    rows = [
        _parameter_row(
            dataset_name=bundle.dataset_name,
            representation=bundle.representation,
            target_name=bundle.target_name,
            model_name=model_name,
            method=method,
            parameter_type="intercept",
            parameter_name="alpha",
            feature_group="global",
            equation_term="alpha",
            draws=alpha,
            runtime_seconds=runtime_seconds,
        ),
        _parameter_row(
            dataset_name=bundle.dataset_name,
            representation=bundle.representation,
            target_name=bundle.target_name,
            model_name=model_name,
            method=method,
            parameter_type="signal_scale",
            parameter_name="signal_scale",
            feature_group="kernel",
            equation_term="signal_scale",
            draws=signal_scale,
            runtime_seconds=runtime_seconds,
        ),
        _parameter_row(
            dataset_name=bundle.dataset_name,
            representation=bundle.representation,
            target_name=bundle.target_name,
            model_name=model_name,
            method=method,
            parameter_type="lengthscale",
            parameter_name="lengthscale",
            feature_group="kernel",
            equation_term="lengthscale",
            draws=lengthscale,
            runtime_seconds=runtime_seconds,
        ),
        _parameter_row(
            dataset_name=bundle.dataset_name,
            representation=bundle.representation,
            target_name=bundle.target_name,
            model_name=model_name,
            method=method,
            parameter_type="noise_scale",
            parameter_name="sigma",
            feature_group="global",
            equation_term="sigma",
            draws=sigma,
            runtime_seconds=runtime_seconds,
        ),
    ]
    for rank, row in enumerate(rows, start=1):
        row["importance_rank"] = rank
    return rows


def _evaluate_split(
    *,
    X: np.ndarray,
    y: np.ndarray,
    mol_ids: tuple[str, ...],
    dataset_name: str,
    representation: str,
    model_name: str,
    method: str,
    split_name: str,
    alpha: np.ndarray,
    beta: np.ndarray,
    sigma: np.ndarray,
    student_df: float,
    runtime_seconds: float,
    feature_count: int,
    n_train: int,
    split_scheme: str,
    source_row_count: int,
    used_row_count: int,
    seed: int,
    predictive_draws: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(y) == 0:
        raise ValueError(f"Empty split {split_name}")
    mu = alpha[:, np.newaxis] + beta @ X.T
    finite_draw_mask = np.isfinite(sigma) & (sigma > 0.0) & np.all(np.isfinite(mu), axis=1)
    if not np.any(finite_draw_mask):
        raise RuntimeError(f"No finite posterior draws remained for {dataset_name}/{representation}/{model_name}/{method}/{split_name}")
    mu = mu[finite_draw_mask]
    sigma = sigma[finite_draw_mask]
    with np.errstate(over="ignore", invalid="ignore"):
        log_density = student_t.logpdf(y[np.newaxis, :], df=student_df, loc=mu, scale=sigma[:, np.newaxis])
    if not np.all(np.isfinite(log_density)):
        bad_draws = np.all(np.isfinite(log_density), axis=1)
        if not np.any(bad_draws):
            raise RuntimeError(f"All log-density evaluations became non-finite for {dataset_name}/{representation}/{model_name}/{method}/{split_name}")
        mu = mu[bad_draws]
        sigma = sigma[bad_draws]
        log_density = log_density[bad_draws]
    mlpd = float(np.mean(logsumexp(log_density, axis=0) - math.log(log_density.shape[0])))
    predictive_mean = np.mean(mu, axis=0)
    predictive_samples = _sample_predictive(mu, sigma, student_df=student_df, seed=seed, draws=predictive_draws)
    predictive_sd = np.std(predictive_samples, axis=0)
    lower = np.quantile(predictive_samples, 0.05, axis=0)
    upper = np.quantile(predictive_samples, 0.95, axis=0)
    residuals = predictive_mean - y
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    mae = float(np.mean(np.abs(residuals)))
    total_sum_squares = float(np.sum(np.square(y - np.mean(y))))
    residual_sum_squares = float(np.sum(np.square(residuals)))
    r2 = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0.0 else 0.0
    coverage = float(np.mean((y >= lower) & (y <= upper)))
    metrics = {
        "dataset": dataset_name,
        "representation": representation,
        "model": model_name,
        "method": method,
        "split": split_name,
        "split_scheme": split_scheme,
        "source_row_count": source_row_count,
        "used_row_count": used_row_count,
        "n_train": n_train,
        "n_eval": int(len(y)),
        "feature_count": feature_count,
        "draw_count": int(alpha.shape[0]),
        "runtime_seconds": runtime_seconds,
        "seed": seed,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mean_log_predictive_density": mlpd,
        "coverage_90": coverage,
        "predictive_sd_mean": float(np.mean(predictive_sd)),
        "student_df": student_df,
    }
    predictions = [
        {
            "dataset": dataset_name,
            "representation": representation,
            "model": model_name,
            "method": method,
            "split": split_name,
            "mol_id": mol_id,
            "actual": float(actual),
            "predicted_mean": float(mean),
            "predictive_sd": float(sd),
            "seed": seed,
        }
        for mol_id, actual, mean, sd in zip(mol_ids, y, predictive_mean, predictive_sd, strict=True)
    ]
    return metrics, predictions


def _sample_predictive(mu: np.ndarray, sigma: np.ndarray, *, student_df: float, seed: int, draws: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mu.shape[0] >= draws:
        chosen = rng.choice(mu.shape[0], size=draws, replace=False)
        sampled_mu = mu[chosen]
        sampled_sigma = sigma[chosen]
    else:
        chosen = rng.choice(mu.shape[0], size=draws, replace=True)
        sampled_mu = mu[chosen]
        sampled_sigma = sigma[chosen]
    noise = student_t.rvs(df=student_df, size=sampled_mu.shape, random_state=rng)
    samples = sampled_mu + noise * sampled_sigma[:, np.newaxis]
    if not np.all(np.isfinite(samples)):
        raise RuntimeError("Posterior predictive sampling produced non-finite values")
    return samples


def _summarize_parameters(
    *,
    bundle: ExportedDataset,
    model_name: str,
    method: str,
    runtime_seconds: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    sigma: np.ndarray,
    global_scale: np.ndarray | None,
    group_scale: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows = [
        _parameter_row(
            dataset_name=bundle.dataset_name,
            representation=bundle.representation,
            target_name=bundle.target_name,
            model_name=model_name,
            method=method,
            parameter_type="intercept",
            parameter_name="alpha",
            feature_group="global",
            equation_term="alpha",
            draws=alpha,
            runtime_seconds=runtime_seconds,
        ),
        _parameter_row(
            dataset_name=bundle.dataset_name,
            representation=bundle.representation,
            target_name=bundle.target_name,
            model_name=model_name,
            method=method,
            parameter_type="noise_scale",
            parameter_name="sigma",
            feature_group="global",
            equation_term="sigma",
            draws=sigma,
            runtime_seconds=runtime_seconds,
        ),
    ]
    if global_scale is not None:
        rows.append(
            _parameter_row(
                dataset_name=bundle.dataset_name,
                representation=bundle.representation,
                target_name=bundle.target_name,
                model_name=model_name,
                method=method,
                parameter_type="global_scale",
                parameter_name="global_scale",
                feature_group="global",
                equation_term="global_scale",
                draws=global_scale,
                runtime_seconds=runtime_seconds,
            )
        )
    if group_scale is not None:
        for group_index, group_name in enumerate(bundle.group_names):
            rows.append(
                _parameter_row(
                    dataset_name=bundle.dataset_name,
                    representation=bundle.representation,
                    target_name=bundle.target_name,
                    model_name=model_name,
                    method=method,
                    parameter_type="group_scale",
                    parameter_name=f"group_scale[{group_name}]",
                    feature_group=group_name,
                    equation_term=f"group_scale[{group_name}]",
                    draws=group_scale[:, group_index],
                    runtime_seconds=runtime_seconds,
                )
            )

    coefficient_rows = [
        _parameter_row(
            dataset_name=bundle.dataset_name,
            representation=bundle.representation,
            target_name=bundle.target_name,
            model_name=model_name,
            method=method,
            parameter_type="coefficient",
            parameter_name=feature_name,
            feature_group=bundle.feature_groups[feature_name],
            equation_term=f"beta[{feature_name}] * z({feature_name})",
            draws=beta[:, feature_index],
            runtime_seconds=runtime_seconds,
        )
        for feature_index, feature_name in enumerate(bundle.feature_names)
    ]
    coefficient_rows.sort(key=lambda row: (-float(row["posterior_abs_mean"]), str(row["parameter_name"])))
    for rank, row in enumerate(coefficient_rows, start=1):
        row["importance_rank"] = rank
    rows.extend(coefficient_rows)
    return rows


def _parameter_row(
    *,
    dataset_name: str,
    representation: str,
    target_name: str,
    model_name: str,
    method: str,
    parameter_type: str,
    parameter_name: str,
    feature_group: str,
    equation_term: str,
    draws: np.ndarray,
    runtime_seconds: float,
) -> dict[str, Any]:
    summary = _draw_summary(draws)
    return {
        "dataset": dataset_name,
        "representation": representation,
        "target": target_name,
        "model": model_name,
        "method": method,
        "parameter_type": parameter_type,
        "parameter_name": parameter_name,
        "feature_group": feature_group,
        "equation_term": equation_term,
        "draw_count": int(summary["draw_count"]),
        "runtime_seconds": runtime_seconds,
        "posterior_mean": summary["mean"],
        "posterior_abs_mean": abs(float(summary["mean"])),
        "posterior_sd": summary["sd"],
        "posterior_median": summary["median"],
        "posterior_p05": summary["p05"],
        "posterior_p95": summary["p95"],
        "importance_rank": 0,
    }


def _draw_summary(draws: np.ndarray) -> dict[str, float]:
    vector = np.asarray(draws, dtype=float).reshape(-1)
    vector = vector[np.isfinite(vector)]
    if vector.size == 0:
        raise RuntimeError("Tried to summarize an empty or non-finite draw vector")
    return {
        "draw_count": float(vector.shape[0]),
        "mean": float(np.mean(vector)),
        "sd": float(np.std(vector)),
        "median": float(np.median(vector)),
        "p05": float(np.quantile(vector, 0.05)),
        "p95": float(np.quantile(vector, 0.95)),
    }


def _ensure_draw_array(value: Any, *, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim == 1:
        if array.ndim == 0:
            return array.reshape(1)
        if array.ndim == 1:
            return array
    if ndim == 2:
        if array.ndim == 1:
            return array[np.newaxis, :]
        if array.ndim == 2:
            return array
    raise ValueError(f"Unexpected array shape {array.shape} for ndim={ndim}")


def _optional_draw_array(fit: Any, name: str, *, ndim: int) -> np.ndarray | None:
    try:
        value = _stan_variable_draws(fit, name)
    except Exception:
        return None
    return _ensure_draw_array(value, ndim=ndim)


def _stan_variable_draws(fit: Any, name: str) -> Any:
    if fit.__class__.__name__ == "CmdStanMLE":
        return _optimized_parameter_value(fit, name)
    if fit.__class__.__name__ == "CmdStanVB":
        return fit.stan_variable(name, mean=False)
    return fit.stan_variable(name)


def _optimized_parameter_value(fit: Any, name: str) -> Any:
    params = fit.optimized_params_dict
    if name in params:
        return params[name]
    prefix = f"{name}["
    indexed_entries: list[tuple[int, Any]] = []
    for key, value in params.items():
        if not key.startswith(prefix) or not key.endswith("]"):
            continue
        index_text = key[len(prefix) : -1]
        try:
            index = int(index_text)
        except ValueError:
            continue
        indexed_entries.append((index, value))
    if indexed_entries:
        indexed_entries.sort(key=lambda item: item[0])
        return np.asarray([value for _, value in indexed_entries], dtype=float)
    raise KeyError(name)


def _filter_draws_for_design_matrix(
    *,
    X: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    sigma: np.ndarray,
    global_scale: np.ndarray | None,
    group_scale: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if X.size == 0:
        return alpha, beta, sigma, global_scale, group_scale
    mu = alpha[:, np.newaxis] + beta @ X.T
    finite_mask = np.isfinite(sigma) & (sigma > 0.0) & np.all(np.isfinite(mu), axis=1)
    if global_scale is not None:
        finite_mask &= np.isfinite(global_scale)
    if group_scale is not None:
        finite_mask &= np.all(np.isfinite(group_scale), axis=1)
    if not np.any(finite_mask):
        raise RuntimeError("No finite posterior draws remained after validating design-matrix projections")
    alpha = alpha[finite_mask]
    beta = beta[finite_mask]
    sigma = sigma[finite_mask]
    if global_scale is not None:
        global_scale = global_scale[finite_mask]
    if group_scale is not None:
        group_scale = group_scale[finite_mask]
    return alpha, beta, sigma, global_scale, group_scale


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
