from __future__ import annotations

import importlib
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import RESULTS_DIR, ensure_directory
from .model_errors import OptionalModelDependencyError
from .predictive_metrics import build_metric_row, build_prediction_rows
from .splits import ExportedDataset

CATBOOST_METHOD = "native_uncertainty"
CATBOOST_MODEL = "catboost_uncertainty"
CATBOOST_VIRTUAL_ENSEMBLES = 10

_CATBOOST_SEARCH_SPACES: dict[str, dict[str, tuple[float | int, ...]]] = {
    "freesolv": {
        "depth": (4, 6, 8),
        "learning_rate": (0.03, 0.05),
        "l2_leaf_reg": (3.0, 6.0, 10.0),
        "iterations": (1500, 2500, 4000),
        "min_data_in_leaf": (1, 3, 5),
    },
    "qm9": {
        "depth": (4, 6, 8),
        "learning_rate": (0.03, 0.05),
        "l2_leaf_reg": (3.0, 6.0, 10.0),
        "iterations": (1500, 2500, 4000),
        "min_data_in_leaf": (1, 3, 5),
    },
}

_CATBOOST_DEFAULTS: dict[str, dict[str, float | int]] = {
    "freesolv": {
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 6.0,
        "iterations": 2500,
        "min_data_in_leaf": 3,
        "early_stopping_rounds": 200,
    },
    "qm9": {
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 4.0,
        "iterations": 4000,
        "min_data_in_leaf": 5,
        "early_stopping_rounds": 300,
    },
}


@dataclass(frozen=True, slots=True)
class CatBoostRunConfig:
    seeds: tuple[int, ...]
    virtual_ensembles_count: int = CATBOOST_VIRTUAL_ENSEMBLES
    search_hyperparameters: bool = True
    verbose: bool = False


def run_catboost_uncertainty(
    bundle: ExportedDataset,
    *,
    config: CatBoostRunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    catboost = _import_catboost()
    feature_names = list(bundle.feature_names)
    X_train = pd.DataFrame(bundle.X_train, columns=feature_names)
    X_valid = pd.DataFrame(bundle.X_valid, columns=feature_names)
    X_test = pd.DataFrame(bundle.X_test, columns=feature_names)
    y_train = np.asarray(bundle.y_train, dtype=float)
    y_valid = np.asarray(bundle.y_valid, dtype=float)
    y_test = np.asarray(bundle.y_test, dtype=float)

    base_params = dict(_catboost_base_params(bundle.dataset_name, seed=config.seeds[0]))
    if config.search_hyperparameters:
        best_params = _search_best_params(
            catboost=catboost,
            dataset_name=bundle.dataset_name,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            base_params=base_params,
            virtual_ensembles_count=config.virtual_ensembles_count,
        )
    else:
        best_params = dict(_dataset_defaults(bundle.dataset_name))

    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        params = dict(base_params)
        params.update(best_params)
        params["random_seed"] = seed
        early_stopping_rounds = int(params.pop("early_stopping_rounds"))
        model = catboost.CatBoostRegressor(**params)
        start = time.perf_counter()
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            use_best_model=False,
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
        )
        runtime_seconds = time.perf_counter() - start
        artifact_rows.extend(
            _write_catboost_artifacts(
                model=model,
                dataset_name=bundle.dataset_name,
                representation=bundle.representation,
                feature_names=feature_names,
                X_valid=X_valid,
                seed=seed,
            )
        )
        metrics_rows.extend(
            _evaluate_catboost_split(
                model=model,
                X_train=X_train,
                y_train=y_train,
                mol_ids_train=bundle.mol_ids_train,
                X_valid=X_valid,
                y_valid=y_valid,
                mol_ids_valid=bundle.mol_ids_valid,
                X_test=X_test,
                y_test=y_test,
                mol_ids_test=bundle.mol_ids_test,
                bundle=bundle,
                runtime_seconds=runtime_seconds,
                seed=seed,
                virtual_ensembles_count=config.virtual_ensembles_count,
                prediction_rows=prediction_rows,
            )
        )
    return metrics_rows, prediction_rows, artifact_rows


def _evaluate_catboost_split(
    *,
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    mol_ids_train: tuple[str, ...],
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    mol_ids_valid: tuple[str, ...],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    mol_ids_test: tuple[str, ...],
    bundle: ExportedDataset,
    runtime_seconds: float,
    seed: int,
    virtual_ensembles_count: int,
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, X, y, mol_ids in (
        ("train", X_train, y_train, mol_ids_train),
        ("valid", X_valid, y_valid, mol_ids_valid),
        ("test", X_test, y_test, mol_ids_test),
    ):
        mean, predictive_sd, uncertainty_components = _catboost_predictions(
            model=model,
            X=X,
            virtual_ensembles_count=virtual_ensembles_count,
        )
        rows.append(
            build_metric_row(
                dataset_name=bundle.dataset_name,
                representation=bundle.representation,
                model_name=CATBOOST_MODEL,
                method=CATBOOST_METHOD,
                split_name=split_name,
                mol_ids=mol_ids,
                actual=y,
                predicted_mean=mean,
                predictive_sd=predictive_sd,
                runtime_seconds=runtime_seconds,
                feature_count=len(bundle.feature_names),
                n_train=len(bundle.y_train),
                split_scheme=bundle.split_scheme,
                source_row_count=bundle.source_row_count,
                used_row_count=bundle.used_row_count,
                seed=seed,
                draw_count=virtual_ensembles_count,
            )
        )
        prediction_rows.extend(
            build_prediction_rows(
                dataset_name=bundle.dataset_name,
                representation=bundle.representation,
                model_name=CATBOOST_MODEL,
                method=CATBOOST_METHOD,
                split_name=split_name,
                mol_ids=mol_ids,
                actual=y,
                predicted_mean=mean,
                predictive_sd=predictive_sd,
                seed=seed,
                extra_columns=uncertainty_components,
            )
        )
    return rows


def _catboost_predictions(
    *,
    model: Any,
    X: pd.DataFrame,
    virtual_ensembles_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    total_uncertainty = np.asarray(
        model.virtual_ensembles_predict(
            X,
            prediction_type="TotalUncertainty",
            virtual_ensembles_count=virtual_ensembles_count,
        ),
        dtype=float,
    )
    if total_uncertainty.ndim == 1:
        total_uncertainty = total_uncertainty[:, np.newaxis]
    predicted_mean = total_uncertainty[:, 0]
    data_uncertainty = total_uncertainty[:, 1] if total_uncertainty.shape[1] > 1 else np.zeros(len(predicted_mean), dtype=float)
    knowledge_uncertainty = total_uncertainty[:, 2] if total_uncertainty.shape[1] > 2 else np.zeros(len(predicted_mean), dtype=float)
    total_variance = np.clip(data_uncertainty + knowledge_uncertainty, 1e-12, None)
    predictive_sd = np.sqrt(total_variance)
    return predicted_mean, predictive_sd, {
        "data_uncertainty": data_uncertainty,
        "knowledge_uncertainty": knowledge_uncertainty,
        "total_uncertainty": total_variance,
    }


def _search_best_params(
    *,
    catboost: Any,
    dataset_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    base_params: dict[str, Any],
    virtual_ensembles_count: int,
) -> dict[str, Any]:
    defaults = _dataset_defaults(dataset_name)
    search_space = _CATBOOST_SEARCH_SPACES.get(dataset_name, _CATBOOST_SEARCH_SPACES["qm9"])
    search_keys = ("depth", "learning_rate", "l2_leaf_reg", "iterations", "min_data_in_leaf")
    best_score = float("inf")
    best_params = dict(defaults)
    for values in itertools.product(*(search_space[key] for key in search_keys)):
        candidate = dict(defaults)
        candidate.update(dict(zip(search_keys, values, strict=True)))
        params = dict(base_params)
        params.update(candidate)
        early_stopping_rounds = int(params.pop("early_stopping_rounds"))
        model = catboost.CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            use_best_model=False,
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
        )
        predicted_mean, _, _ = _catboost_predictions(
            model=model,
            X=X_valid,
            virtual_ensembles_count=virtual_ensembles_count,
        )
        rmse = float(np.sqrt(np.mean(np.square(predicted_mean - y_valid))))
        if rmse < best_score:
            best_score = rmse
            best_params = candidate
    return best_params


def _catboost_base_params(dataset_name: str, *, seed: int) -> dict[str, Any]:
    params = dict(_dataset_defaults(dataset_name))
    params.update(
        {
            "loss_function": "RMSEWithUncertainty",
            "eval_metric": "RMSE",
            "posterior_sampling": True,
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.5,
            "random_strength": 1.0,
            "thread_count": -1,
        }
    )
    return params


def _dataset_defaults(dataset_name: str) -> dict[str, Any]:
    return dict(_CATBOOST_DEFAULTS.get(dataset_name, _CATBOOST_DEFAULTS["qm9"]))


def _write_catboost_artifacts(
    *,
    model: Any,
    dataset_name: str,
    representation: str,
    feature_names: list[str],
    X_valid: pd.DataFrame,
    seed: int,
) -> list[dict[str, Any]]:
    artifact_dir = ensure_directory(
        RESULTS_DIR / "model_artifacts" / "catboost" / dataset_name / representation / f"seed_{seed}"
    )
    rows: list[dict[str, Any]] = []
    feature_importance = np.asarray(model.get_feature_importance(), dtype=float)
    importance_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": feature_importance,
        }
    ).sort_values("importance", ascending=False)
    importance_path = artifact_dir / "feature_importance.csv"
    importance_frame.to_csv(importance_path, index=False)
    rows.append(
        {
            "dataset": dataset_name,
            "representation": representation,
            "model": CATBOOST_MODEL,
            "seed": seed,
            "artifact_type": "feature_importance",
            "path": str(importance_path.relative_to(RESULTS_DIR)),
        }
    )
    try:
        shap_values = np.asarray(
            model.get_feature_importance(
                data=X_valid,
                type="ShapValues",
            ),
            dtype=float,
        )
        if shap_values.ndim == 2 and shap_values.shape[1] >= len(feature_names):
            shap_summary = np.mean(np.abs(shap_values[:, : len(feature_names)]), axis=0)
            shap_frame = pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_abs_shap": shap_summary,
                }
            ).sort_values("mean_abs_shap", ascending=False)
            shap_path = artifact_dir / "shap_summary.csv"
            shap_frame.to_csv(shap_path, index=False)
            rows.append(
                {
                    "dataset": dataset_name,
                    "representation": representation,
                    "model": CATBOOST_MODEL,
                    "seed": seed,
                    "artifact_type": "shap_summary",
                    "path": str(shap_path.relative_to(RESULTS_DIR)),
                }
            )
    except Exception:
        # SHAP export is optional and depends on CatBoost build capabilities.
        pass
    return rows


def _import_catboost() -> Any:
    try:
        return importlib.import_module("catboost")
    except ModuleNotFoundError as exc:
        raise OptionalModelDependencyError(
            "CatBoost is not installed in the local repo environment. Re-run `make python-setup` "
            "to install the default model stack."
        ) from exc
