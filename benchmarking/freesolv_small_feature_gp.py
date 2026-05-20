from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import minimize

from moladt.chem.molecule import Molecule, molecule_edges
from moladt.chem.molecule_ops import effective_order
from moladt.io.smiles import molecule_to_smiles, parse_smiles

from .common import PROCESSED_DATA_DIR, ensure_directory
from .predictive_metrics import build_metric_row, build_prediction_rows


REPRESENTATION = "moladt_small_descriptors"
METHOD_NAME = "empirical_bayes_exact_gp"
DEFAULT_SPLIT_COUNT = 20
DEFAULT_SINGLE_SPLIT_SEED = 18
MAX_OPTIMIZER_ITERATIONS = 160
SPLIT_SCHEME = "moleculenet_random_like:513/64/65;final_refit=train+valid"
TRAIN_SIZE = 513
VALID_SIZE = 64
TEST_SIZE = 65

SMILES_ATOM_FEATURES: tuple[str, ...] = (
    "smiles_atom_count_c",
    "smiles_atom_count_n",
    "smiles_atom_count_o",
    "smiles_atom_count_f",
    "smiles_atom_count_p",
    "smiles_atom_count_s",
    "smiles_atom_count_cl",
    "smiles_atom_count_br",
    "smiles_atom_count_i",
    "smiles_atom_count_h",
)

ATOM_BAG_FEATURES: tuple[str, ...] = SMILES_ATOM_FEATURES

ADJACENCY_GRAPH_FEATURES: tuple[str, ...] = (
    *SMILES_ATOM_FEATURES,
    "smiles_bond_count_single",
    "smiles_bond_count_aromatic",
    "smiles_bond_count_double",
    "smiles_bond_count_triple",
    "smiles_bond_count_total",
    "smiles_heavy_atom_count",
    "smiles_component_count",
    "smiles_cycle_rank",
    "smiles_heavy_degree_mean",
    "smiles_heavy_degree_max",
)

MOLADT_COMPOSITION_FEATURES: tuple[str, ...] = (
    "weight",
    "polar",
    "surface",
    "donor_count",
    "acceptor_count",
    "heavy_atoms",
    "halogens",
    "atom_count_c",
    "atom_count_n",
    "atom_count_o",
)

MOLADT_MULTIGRAPH_FEATURES: tuple[str, ...] = (
    "sigma_edge_count",
    "effective_bond_order_sum",
    "effective_bond_order_mean",
    "effective_bond_order_max",
    "edge_order_sigma_like_count",
    "edge_order_delocalized_count",
    "edge_order_double_like_count",
    "edge_order_triple_plus_count",
    "bonding_system_count",
    "multicentre_system_count",
    "pi_ring_system_count",
    "system_member_edges_max",
    "system_shared_electrons_sum",
    "system_shared_electrons_mean",
)

MOLADT_GEOMETRY_FEATURES: tuple[str, ...] = (
    "ring_edge_fraction",
    "rotatable_bonds",
    "heavy_atom_degree_mean",
    "heavy_atom_degree_max",
    "aprdf_edge_order_1p5a",
    "aprdf_system_edge_1p5a",
)

FULL_MOLADT_FEATURES: tuple[str, ...] = (
    *MOLADT_COMPOSITION_FEATURES,
    *MOLADT_MULTIGRAPH_FEATURES,
    *MOLADT_GEOMETRY_FEATURES,
)

MODEL_FEATURES: dict[str, tuple[str, ...]] = {
    "atom_bag": ATOM_BAG_FEATURES,
    "adjacency_graph": ADJACENCY_GRAPH_FEATURES,
    "full_moladt": FULL_MOLADT_FEATURES,
}

MODEL_NAMES: dict[str, str] = {
    "atom_bag": "atom_bag10_rbf_gp",
    "adjacency_graph": "adjacency_graph20_rbf_gp",
    "full_moladt": "moladt_full30_rbf_gp",
}


@dataclass(frozen=True, slots=True)
class SmallFeatureResult:
    metric_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]
    coefficient_rows: list[dict[str, Any]]
    artifact_rows: list[dict[str, Any]]
    feature_rows: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _RBFGPFit:
    model_key: str
    feature_names: tuple[str, ...]
    signal_variance: float
    lengthscale: float
    noise_variance: float
    y_mean: float
    y_scale: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    x_train_scaled: np.ndarray
    alpha: np.ndarray
    cholesky: tuple[np.ndarray, bool]
    train_idx: np.ndarray
    full_kernel: np.ndarray
    nll: float


@dataclass(frozen=True, slots=True)
class SmallFeaturePredictorState:
    fit: _RBFGPFit
    seed: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray


def run_freesolv_small_feature_gp_splits(
    *,
    model_keys: Sequence[str],
    seeds: Sequence[int],
    noise_floor: float,
    verbose: bool = False,
) -> SmallFeatureResult:
    rows = _load_feature_rows()
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        feature_names = MODEL_FEATURES[model_key]
        x = _feature_matrix(rows, feature_names)
        y = rows["expt"].to_numpy(dtype=float)
        mol_ids = tuple(rows["mol_id"].astype(str).tolist())
        feature_rows.extend(_feature_manifest_rows(model_key, feature_names))
        for seed in seeds:
            result = _evaluate_model_split(
                model_key=model_key,
                rows=rows,
                mol_ids=mol_ids,
                x=x,
                y=y,
                seed=int(seed),
                noise_floor=float(noise_floor),
                verbose=verbose,
            )
            metric_rows.extend(result.metric_rows)
            prediction_rows.extend(result.prediction_rows)
            coefficient_rows.extend(result.coefficient_rows)
            artifact_rows.extend(result.artifact_rows)

    return SmallFeatureResult(
        metric_rows=metric_rows,
        prediction_rows=prediction_rows,
        coefficient_rows=coefficient_rows,
        artifact_rows=artifact_rows,
        feature_rows=feature_rows,
    )


def load_small_feature_predictor_state(
    *,
    seed: int = DEFAULT_SINGLE_SPLIT_SEED,
    model_key: str = "full_moladt",
    noise_floor: float = 0.01,
) -> SmallFeaturePredictorState:
    rows = _load_feature_rows()
    feature_names = MODEL_FEATURES[model_key]
    x = _feature_matrix(rows, feature_names)
    y = rows["expt"].to_numpy(dtype=float)
    train_idx, valid_idx, test_idx = train_valid_test_indices(len(y), seed)
    final_train_idx = np.sort(np.concatenate([train_idx, valid_idx]))
    fit = _fit_rbf_gp(
        model_key=model_key,
        x=x,
        y=y,
        train_idx=final_train_idx,
        feature_names=feature_names,
        noise_floor=noise_floor,
    )
    return SmallFeaturePredictorState(
        fit=fit,
        seed=int(seed),
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
    )


def predict_with_small_feature_state(state: SmallFeaturePredictorState, molecule: Molecule) -> tuple[float, float]:
    vector = _molecule_feature_vector(molecule, state.fit.feature_names)
    x_scaled = (vector - state.fit.feature_mean) / state.fit.feature_scale
    deltas = state.fit.x_train_scaled - x_scaled
    d2 = np.sum(deltas * deltas, axis=1)
    cross = state.fit.signal_variance * np.exp(
        -0.5 * d2 / max(state.fit.lengthscale * state.fit.lengthscale, 1.0e-12)
    )
    mean_scaled = float(cross @ state.fit.alpha)
    solved = cho_solve(state.fit.cholesky, cross, check_finite=False)
    latent_var = state.fit.signal_variance - float(cross @ solved)
    observed_var = max(latent_var + state.fit.noise_variance, 1.0e-10)
    mean = state.fit.y_mean + state.fit.y_scale * mean_scaled
    sd = state.fit.y_scale * math.sqrt(observed_var)
    return float(mean), float(sd)


def write_freesolv_small_feature_outputs(result: SmallFeatureResult, output_dir: Path) -> dict[str, Path]:
    output_dir = ensure_directory(output_dir)
    metrics_path = output_dir / "predictive_metrics.csv"
    predictions_path = output_dir / "predictions.csv"
    coefficients_path = output_dir / "model_coefficients.csv"
    artifacts_path = output_dir / "model_artifacts.csv"
    assignments_path = output_dir / "split_assignments.csv"
    features_path = output_dir / "feature_manifest.csv"
    summary_path = output_dir / "summary.csv"
    paired_path = output_dir / "paired_against_full_moladt.csv"
    svg_path = output_dir / "freesolv_small_feature_ablation.svg"

    metrics = pd.DataFrame(result.metric_rows)
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(result.prediction_rows).to_csv(predictions_path, index=False)
    pd.DataFrame(result.coefficient_rows).to_csv(coefficients_path, index=False)
    artifacts = pd.DataFrame(result.artifact_rows)
    artifacts.to_csv(artifacts_path, index=False)
    pd.DataFrame(_split_assignment_rows(result.artifact_rows)).to_csv(assignments_path, index=False)
    pd.DataFrame(result.feature_rows).to_csv(features_path, index=False)
    summary = _summary_frame(metrics)
    summary.to_csv(summary_path, index=False)
    _paired_frame(metrics).to_csv(paired_path, index=False)
    _write_ablation_svg(summary, svg_path)
    return {
        "predictive_metrics": metrics_path,
        "predictions": predictions_path,
        "model_coefficients": coefficients_path,
        "model_artifacts": artifacts_path,
        "split_assignments": assignments_path,
        "feature_manifest": features_path,
        "summary": summary_path,
        "paired_against_full_moladt": paired_path,
        "svg": svg_path,
    }


def _evaluate_model_split(
    *,
    model_key: str,
    rows: pd.DataFrame,
    mol_ids: tuple[str, ...],
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    noise_floor: float,
    verbose: bool,
) -> SmallFeatureResult:
    start = time.perf_counter()
    train_idx, valid_idx, test_idx = train_valid_test_indices(len(y), seed)
    feature_names = MODEL_FEATURES[model_key]

    if verbose:
        print(
            f"[freesolv/small-feature-gp] model={model_key} seed={seed} "
            f"features={len(feature_names)}",
            flush=True,
        )

    selection_fit = _fit_rbf_gp(
        model_key=model_key,
        x=x,
        y=y,
        train_idx=train_idx,
        feature_names=feature_names,
        noise_floor=noise_floor,
    )
    valid_mean, valid_sd = _predict_rbf_gp(selection_fit, valid_idx)

    final_train_idx = np.sort(np.concatenate([train_idx, valid_idx]))
    final_fit = _fit_rbf_gp(
        model_key=model_key,
        x=x,
        y=y,
        train_idx=final_train_idx,
        feature_names=feature_names,
        noise_floor=noise_floor,
    )
    train_mean, train_sd = _predict_rbf_gp(final_fit, final_train_idx)
    test_mean, test_sd = _predict_rbf_gp(final_fit, test_idx)

    runtime = time.perf_counter() - start
    model_name = MODEL_NAMES[model_key]
    feature_count = len(feature_names)
    parameter_count = 5
    metric_rows = [
        build_metric_row(
            dataset_name="freesolv",
            representation=REPRESENTATION,
            model_name=model_name,
            method=METHOD_NAME,
            split_name="train",
            mol_ids=tuple(mol_ids[int(index)] for index in final_train_idx),
            actual=y[final_train_idx],
            predicted_mean=train_mean,
            predictive_sd=train_sd,
            runtime_seconds=runtime,
            feature_count=feature_count,
            n_train=int(len(final_train_idx)),
            split_scheme=SPLIT_SCHEME,
            source_row_count=len(rows),
            used_row_count=len(rows),
            seed=seed,
            draw_count=1,
            parameter_count=parameter_count,
        ),
        build_metric_row(
            dataset_name="freesolv",
            representation=REPRESENTATION,
            model_name=model_name,
            method=METHOD_NAME,
            split_name="valid",
            mol_ids=tuple(mol_ids[int(index)] for index in valid_idx),
            actual=y[valid_idx],
            predicted_mean=valid_mean,
            predictive_sd=valid_sd,
            runtime_seconds=runtime,
            feature_count=feature_count,
            n_train=int(len(train_idx)),
            split_scheme=SPLIT_SCHEME,
            source_row_count=len(rows),
            used_row_count=len(rows),
            seed=seed,
            draw_count=1,
            parameter_count=parameter_count,
            extra_metrics={"selection_nll": selection_fit.nll},
        ),
        build_metric_row(
            dataset_name="freesolv",
            representation=REPRESENTATION,
            model_name=model_name,
            method=METHOD_NAME,
            split_name="test",
            mol_ids=tuple(mol_ids[int(index)] for index in test_idx),
            actual=y[test_idx],
            predicted_mean=test_mean,
            predictive_sd=test_sd,
            runtime_seconds=runtime,
            feature_count=feature_count,
            n_train=int(len(final_train_idx)),
            split_scheme=SPLIT_SCHEME,
            source_row_count=len(rows),
            used_row_count=len(rows),
            seed=seed,
            draw_count=1,
            parameter_count=parameter_count,
            extra_metrics={"final_nll": final_fit.nll},
        ),
    ]
    prediction_rows = [
        *build_prediction_rows(
            dataset_name="freesolv",
            representation=REPRESENTATION,
            model_name=model_name,
            method=METHOD_NAME,
            split_name="train",
            mol_ids=tuple(mol_ids[int(index)] for index in final_train_idx),
            actual=y[final_train_idx],
            predicted_mean=train_mean,
            predictive_sd=train_sd,
            seed=seed,
        ),
        *build_prediction_rows(
            dataset_name="freesolv",
            representation=REPRESENTATION,
            model_name=model_name,
            method=METHOD_NAME,
            split_name="valid",
            mol_ids=tuple(mol_ids[int(index)] for index in valid_idx),
            actual=y[valid_idx],
            predicted_mean=valid_mean,
            predictive_sd=valid_sd,
            seed=seed,
        ),
        *build_prediction_rows(
            dataset_name="freesolv",
            representation=REPRESENTATION,
            model_name=model_name,
            method=METHOD_NAME,
            split_name="test",
            mol_ids=tuple(mol_ids[int(index)] for index in test_idx),
            actual=y[test_idx],
            predicted_mean=test_mean,
            predictive_sd=test_sd,
            seed=seed,
        ),
    ]
    artifact_rows = [
        {
            "dataset": "freesolv",
            "representation": REPRESENTATION,
            "model": model_name,
            "method": METHOD_NAME,
            "artifact_type": "small_feature_rbf_gp",
            "seed": seed,
            "split_scheme": SPLIT_SCHEME,
            "train_mol_ids": ";".join(mol_ids[int(index)] for index in train_idx),
            "valid_mol_ids": ";".join(mol_ids[int(index)] for index in valid_idx),
            "test_mol_ids": ";".join(mol_ids[int(index)] for index in test_idx),
            "feature_count": feature_count,
            "feature_names": ";".join(feature_names),
            "source_feature_table": str(PROCESSED_DATA_DIR / "freesolv_moladt_featurized_features.csv"),
        }
    ]
    return SmallFeatureResult(
        metric_rows=metric_rows,
        prediction_rows=prediction_rows,
        coefficient_rows=_coefficient_rows(final_fit, runtime_seconds=runtime),
        artifact_rows=artifact_rows,
        feature_rows=[],
    )


def _fit_rbf_gp(
    *,
    model_key: str,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    feature_names: tuple[str, ...],
    noise_floor: float,
) -> _RBFGPFit:
    x_train_raw = x[train_idx]
    feature_mean = x_train_raw.mean(axis=0)
    feature_scale = x_train_raw.std(axis=0, ddof=0)
    feature_scale = np.where(feature_scale < 1.0e-12, 1.0, feature_scale)
    x_scaled = (x - feature_mean) / feature_scale

    y_train_raw = y[train_idx]
    y_mean = float(y_train_raw.mean())
    y_scale = float(y_train_raw.std(ddof=0))
    if y_scale < 1.0e-12:
        y_scale = 1.0
    y_train = (y_train_raw - y_mean) / y_scale

    d2_full = _squared_distances(x_scaled)
    d2_train = d2_full[np.ix_(train_idx, train_idx)]
    n_train = len(train_idx)

    def build_kernel(log_params: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        signal = float(np.exp(log_params[0]))
        lengthscale = float(np.exp(log_params[1]))
        noise = float(np.exp(log_params[2]))
        kernel = signal * np.exp(-0.5 * d2_train / max(lengthscale * lengthscale, 1.0e-12))
        kernel += (noise + 1.0e-7) * np.eye(n_train)
        return kernel, signal, lengthscale, noise

    def objective(log_params: np.ndarray) -> float:
        kernel, _, _, _ = build_kernel(log_params)
        try:
            cholesky = cho_factor(kernel, lower=True, check_finite=False)
            alpha = cho_solve(cholesky, y_train, check_finite=False)
        except (LinAlgError, ValueError):
            return 1.0e25
        logdet = 2.0 * np.sum(np.log(np.diag(cholesky[0])))
        return float(0.5 * y_train @ alpha + 0.5 * logdet + 0.5 * n_train * np.log(2.0 * math.pi))

    initial_lengthscale = _initial_lengthscale(d2_train)
    initial_noise = max(float(noise_floor) * 2.0, 0.05)
    initial = np.log(np.asarray([1.0, initial_lengthscale, initial_noise], dtype=float))
    bounds = [
        (math.log(1.0e-4), math.log(50.0)),
        (math.log(0.15), math.log(50.0)),
        (math.log(float(noise_floor)), math.log(2.0)),
    ]
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": MAX_OPTIMIZER_ITERATIONS, "ftol": 1e-7, "maxls": 25},
    )
    params = result.x if result.success or np.isfinite(result.fun) else initial
    kernel_train, signal, lengthscale, noise = build_kernel(params)
    cholesky = cho_factor(kernel_train, lower=True, check_finite=False)
    alpha = cho_solve(cholesky, y_train, check_finite=False)
    full_kernel = signal * np.exp(-0.5 * d2_full / max(lengthscale * lengthscale, 1.0e-12))
    return _RBFGPFit(
        model_key=model_key,
        feature_names=feature_names,
        signal_variance=signal,
        lengthscale=lengthscale,
        noise_variance=noise,
        y_mean=y_mean,
        y_scale=y_scale,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        x_train_scaled=x_scaled[train_idx].copy(),
        alpha=alpha,
        cholesky=cholesky,
        train_idx=train_idx.copy(),
        full_kernel=full_kernel,
        nll=float(objective(params)),
    )


def _predict_rbf_gp(fit: _RBFGPFit, target_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cross = fit.full_kernel[np.ix_(fit.train_idx, target_idx)]
    mean_scaled = cross.T @ fit.alpha
    solved = cho_solve(fit.cholesky, cross, check_finite=False)
    latent_var = fit.signal_variance - np.sum(cross * solved, axis=0)
    observed_var = np.maximum(latent_var + fit.noise_variance, 1.0e-10)
    mean = fit.y_mean + fit.y_scale * mean_scaled
    sd = fit.y_scale * np.sqrt(observed_var)
    return mean, sd


def _load_feature_rows() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "freesolv_moladt_featurized_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected processed FreeSolv feature table at {path}")
    rows = pd.read_csv(path)
    rows.loc[:, "mol_id"] = rows["mol_id"].astype(str)
    rows = rows.sort_values("mol_id").reset_index(drop=True)
    return rows


def train_valid_test_indices(row_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if row_count != TRAIN_SIZE + VALID_SIZE + TEST_SIZE:
        raise ValueError(f"FreeSolv small-feature GP expects 642 rows; found {row_count}")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(row_count)
    train = np.sort(permutation[:TRAIN_SIZE])
    valid = np.sort(permutation[TRAIN_SIZE : TRAIN_SIZE + VALID_SIZE])
    test = np.sort(permutation[TRAIN_SIZE + VALID_SIZE :])
    return train, valid, test


def _feature_matrix(rows: pd.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    expanded = pd.concat(
        [
            rows.reset_index(drop=True),
            _smiles_graph_feature_frame(rows["smiles"].astype(str).tolist()),
        ],
        axis=1,
    )
    missing = [name for name in feature_names if name not in expanded.columns]
    if missing:
        raise KeyError(f"Missing expected feature columns: {', '.join(missing)}")
    matrix = expanded.loc[:, feature_names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Small-feature matrix contains non-finite values")
    return matrix


def _smiles_graph_feature_frame(smiles_values: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([_smiles_graph_features(smiles) for smiles in smiles_values])


def _smiles_graph_features(smiles: str) -> dict[str, float]:
    molecule = parse_smiles(smiles)
    atoms = sorted(molecule.atoms)
    edges = sorted(molecule_edges(molecule))
    features = {name: 0.0 for name in ADJACENCY_GRAPH_FEATURES}
    for atom in molecule.atoms.values():
        symbol = atom.attributes.symbol.value.lower()
        key = f"smiles_atom_count_{symbol}"
        if key in features:
            features[key] += 1.0

    heavy_atoms = [atom_id for atom_id in atoms if molecule.atoms[atom_id].attributes.symbol.value.upper() != "H"]
    heavy_degrees = {atom_id: 0 for atom_id in heavy_atoms}
    for edge in edges:
        order = _smiles_edge_order(molecule, edge)
        if order < 1.25:
            features["smiles_bond_count_single"] += 1.0
        elif order < 1.75:
            features["smiles_bond_count_aromatic"] += 1.0
        elif order < 2.50:
            features["smiles_bond_count_double"] += 1.0
        else:
            features["smiles_bond_count_triple"] += 1.0
        if edge.a in heavy_degrees:
            heavy_degrees[edge.a] += 1
        if edge.b in heavy_degrees:
            heavy_degrees[edge.b] += 1

    component_count = _component_count(atoms, edges)
    features["smiles_bond_count_total"] = float(len(edges))
    features["smiles_heavy_atom_count"] = float(len(heavy_atoms))
    features["smiles_component_count"] = float(component_count)
    features["smiles_cycle_rank"] = float(max(0, len(edges) - len(atoms) + component_count))
    degree_values = list(heavy_degrees.values())
    features["smiles_heavy_degree_mean"] = float(np.mean(degree_values)) if degree_values else 0.0
    features["smiles_heavy_degree_max"] = float(max(degree_values)) if degree_values else 0.0
    return features


def _molecule_feature_vector(molecule: Molecule, feature_names: tuple[str, ...]) -> np.ndarray:
    smiles_features = _smiles_graph_features(molecule_to_smiles(molecule))
    descriptor_values: dict[str, float] = {}
    if any(name in FULL_MOLADT_FEATURES for name in feature_names):
        from .features import compute_moladt_featurized_descriptors

        descriptor_values = compute_moladt_featurized_descriptors(molecule)
    values = {**smiles_features, **descriptor_values}
    missing = [name for name in feature_names if name not in values]
    if missing:
        raise KeyError(f"Cannot build small-feature vector; missing: {', '.join(missing)}")
    return np.asarray([float(values[name]) for name in feature_names], dtype=float)


def _smiles_edge_order(molecule: Molecule, edge: Any) -> float:
    order = float(effective_order(molecule, edge))
    if order <= 0.0:
        return 1.0
    return order


def _component_count(atoms: Sequence[Any], edges: Sequence[Any]) -> int:
    if not atoms:
        return 0
    adjacency: dict[Any, set[Any]] = {atom_id: set() for atom_id in atoms}
    for edge in edges:
        adjacency.setdefault(edge.a, set()).add(edge.b)
        adjacency.setdefault(edge.b, set()).add(edge.a)
    seen: set[Any] = set()
    count = 0
    for atom_id in atoms:
        if atom_id in seen:
            continue
        count += 1
        stack = [atom_id]
        seen.add(atom_id)
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return count


def _squared_distances(matrix: np.ndarray) -> np.ndarray:
    squared_norm = np.sum(matrix * matrix, axis=1)
    distances = squared_norm[:, None] + squared_norm[None, :] - 2.0 * matrix @ matrix.T
    return np.maximum(distances, 0.0)


def _initial_lengthscale(d2_train: np.ndarray) -> float:
    upper = d2_train[np.triu_indices_from(d2_train, k=1)]
    positive = upper[upper > 1.0e-12]
    if len(positive) == 0:
        return 1.0
    return float(np.clip(np.sqrt(np.median(positive)), 0.5, 20.0))


def _coefficient_rows(fit: _RBFGPFit, *, runtime_seconds: float) -> list[dict[str, Any]]:
    values = {
        "signal_variance": fit.signal_variance,
        "lengthscale": fit.lengthscale,
        "noise_variance": fit.noise_variance,
        "target_mean": fit.y_mean,
        "target_scale": fit.y_scale,
    }
    rows: list[dict[str, Any]] = []
    for rank, (name, value) in enumerate(values.items(), start=1):
        rows.append(
            {
                "dataset": "freesolv",
                "representation": REPRESENTATION,
                "target": "expt",
                "model": MODEL_NAMES[fit.model_key],
                "method": METHOD_NAME,
                "parameter_type": "kernel_hyperparameter",
                "parameter_name": name,
                "feature_group": fit.model_key,
                "equation_term": name,
                "draw_count": 1,
                "runtime_seconds": runtime_seconds,
                "posterior_mean": float(value),
                "posterior_abs_mean": abs(float(value)),
                "posterior_sd": 0.0,
                "posterior_median": float(value),
                "posterior_p05": float(value),
                "posterior_p95": float(value),
                "importance_rank": rank,
            }
        )
    return rows


def _feature_manifest_rows(model_key: str, feature_names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": "freesolv",
            "representation": REPRESENTATION,
            "model": MODEL_NAMES[model_key],
            "feature_rank": index,
            "feature_name": name,
        }
        for index, name in enumerate(feature_names, start=1)
    ]


def _split_assignment_rows(artifact_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifact_rows:
        seed = int(artifact["seed"])
        model = str(artifact["model"])
        for split_name, field_name in (
            ("train", "train_mol_ids"),
            ("valid", "valid_mol_ids"),
            ("test", "test_mol_ids"),
        ):
            mol_ids = [mol_id for mol_id in str(artifact.get(field_name, "")).split(";") if mol_id]
            rows.extend(
                {"model": model, "seed": seed, "split": split_name, "mol_id": mol_id}
                for mol_id in mol_ids
            )
    return rows


def _summary_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    metric_columns = ["rmse", "mae", "r2", "predictive_sd_mean", "coverage_90"]
    rows: list[dict[str, Any]] = []
    for (model, split_name), frame in metrics.groupby(["model", "split"], sort=True):
        row: dict[str, Any] = {
            "model": model,
            "split": split_name,
            "feature_count": int(frame["feature_count"].iloc[0]),
            "n_splits": int(frame["seed"].nunique()),
            "n_eval_mean": float(frame["n_eval"].mean()),
        }
        for column in metric_columns:
            values = frame[column].astype(float)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    test = metrics.loc[metrics["split"] == "test", ["model", "seed", "rmse", "mae"]].copy()
    full = MODEL_NAMES["full_moladt"]
    if full not in set(test["model"]):
        return pd.DataFrame()
    wide = test.pivot(index="seed", columns="model", values=["rmse", "mae"])
    rows: list[dict[str, Any]] = []
    compared_models = [name for key, name in MODEL_NAMES.items() if key != "full_moladt" and name in set(test["model"])]
    for seed in wide.index:
        for model in compared_models:
            rows.append(
                {
                    "seed": int(seed),
                    "model": model,
                    "rmse_model": float(wide.loc[seed, ("rmse", model)]),
                    "rmse_full_moladt": float(wide.loc[seed, ("rmse", full)]),
                    "rmse_delta_model_minus_full_moladt": float(
                        wide.loc[seed, ("rmse", model)] - wide.loc[seed, ("rmse", full)]
                    ),
                    "mae_model": float(wide.loc[seed, ("mae", model)]),
                    "mae_full_moladt": float(wide.loc[seed, ("mae", full)]),
                    "mae_delta_model_minus_full_moladt": float(
                        wide.loc[seed, ("mae", model)] - wide.loc[seed, ("mae", full)]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _write_ablation_svg(summary: pd.DataFrame, destination: Path) -> None:
    test = summary.loc[summary["split"] == "test"].copy()
    if test.empty:
        destination.write_text("", encoding="utf-8")
        return
    order = [MODEL_NAMES[key] for key in MODEL_FEATURES if MODEL_NAMES[key] in set(test["model"])]
    test.loc[:, "model_order"] = test["model"].map({model: index for index, model in enumerate(order)})
    test = test.sort_values("model_order")
    width = 980
    height = 560
    margin_left = 92
    margin_right = 38
    plot_top = 104
    plot_height = 322
    plot_width = width - margin_left - margin_right
    axis_bottom = plot_top + plot_height
    max_value = float((test["rmse_mean"] + test["rmse_std"].fillna(0.0)).max())
    y_max = max(1.0, math.ceil(max_value * 10.0 + 1.0) / 10.0)
    colors = {
        MODEL_NAMES["atom_bag"]: "#2563eb",
        MODEL_NAMES["adjacency_graph"]: "#059669",
        MODEL_NAMES["full_moladt"]: "#dc2626",
    }
    labels = {
        MODEL_NAMES["atom_bag"]: "Atom bag",
        MODEL_NAMES["adjacency_graph"]: "Adjacency graph",
        MODEL_NAMES["full_moladt"]: "Full MolADT",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="FreeSolv small-feature ablation">',
        f'<rect width="{width}" height="{height}" fill="#ffffff" />',
        '<text x="34" y="48" font-size="30" font-family="Helvetica, Arial, sans-serif" fill="#111827" font-weight="700">FreeSolv Small-Feature Ablation</text>',
        '<text x="34" y="78" font-size="15" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">20 deterministic 80:10:10 splits; lower test RMSE is better.</text>',
    ]
    for step in range(6):
        value = y_max * step / 5
        y = axis_bottom - plot_height * step / 5
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#d1d5db" stroke-width="1" />')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="12" font-family="Menlo, Consolas, monospace" fill="#4b5563">{value:.2f}</text>')
    parts.append(f'<line x1="{margin_left}" y1="{axis_bottom}" x2="{margin_left + plot_width}" y2="{axis_bottom}" stroke="#111827" stroke-width="2" />')
    bar_width = 130
    gap = (plot_width - len(test) * bar_width) / max(len(test) + 1, 1)
    for index, row in enumerate(test.itertuples(index=False)):
        model = str(row.model)
        mean = float(row.rmse_mean)
        std = float(row.rmse_std)
        feature_count = int(row.feature_count)
        x = margin_left + gap + index * (bar_width + gap)
        bar_height = plot_height * mean / y_max
        y = axis_bottom - bar_height
        error_top = axis_bottom - plot_height * min(mean + std, y_max) / y_max
        error_bottom = axis_bottom - plot_height * max(mean - std, 0.0) / y_max
        center = x + bar_width / 2
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors.get(model, "#374151")}" opacity="0.94" />')
        parts.append(f'<line x1="{center:.1f}" y1="{error_top:.1f}" x2="{center:.1f}" y2="{error_bottom:.1f}" stroke="#111827" stroke-width="2" />')
        parts.append(f'<line x1="{center - 18:.1f}" y1="{error_top:.1f}" x2="{center + 18:.1f}" y2="{error_top:.1f}" stroke="#111827" stroke-width="2" />')
        parts.append(f'<line x1="{center - 18:.1f}" y1="{error_bottom:.1f}" x2="{center + 18:.1f}" y2="{error_bottom:.1f}" stroke="#111827" stroke-width="2" />')
        parts.append(f'<text x="{center:.1f}" y="{y - 14:.1f}" text-anchor="middle" font-size="14" font-family="Menlo, Consolas, monospace" fill="#111827" font-weight="700">{mean:.3f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{axis_bottom + 34}" text-anchor="middle" font-size="16" font-family="Helvetica, Arial, sans-serif" fill="#111827" font-weight="700">{labels.get(model, model)}</text>')
        parts.append(f'<text x="{center:.1f}" y="{axis_bottom + 58}" text-anchor="middle" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">{feature_count} features</text>')
    parts.append(f'<text x="28" y="{plot_top + plot_height / 2:.1f}" text-anchor="middle" font-size="16" font-family="Helvetica, Arial, sans-serif" fill="#111827" font-weight="700" transform="rotate(-90 28 {plot_top + plot_height / 2:.1f})">Test RMSE (kcal/mol)</text>')
    parts.append("</svg>\n")
    destination.write_text("\n".join(parts), encoding="utf-8")


def _default_output_dir() -> Path:
    return Path("results") / "freesolv_small_feature_gp" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarking.freesolv_small_feature_gp",
        description="Run small, listable MolADT descriptor RBF GPs on repeated FreeSolv splits.",
    )
    parser.add_argument("--split-count", type=int, default=DEFAULT_SPLIT_COUNT)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument(
        "--model",
        choices=("atom_bag", "adjacency_graph", "full_moladt", "all"),
        default="all",
        help="Small feature set to evaluate.",
    )
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=0.01,
        help="Lower bound on the GP observation-noise variance after target standardization.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split_count <= 0:
        raise ValueError("--split-count must be positive")
    if args.noise_floor <= 0.0:
        raise ValueError("--noise-floor must be positive")
    model_keys = tuple(MODEL_FEATURES) if args.model == "all" else (str(args.model),)
    seeds = tuple(range(int(args.seed_start), int(args.seed_start) + int(args.split_count)))
    result = run_freesolv_small_feature_gp_splits(
        model_keys=model_keys,
        seeds=seeds,
        noise_floor=float(args.noise_floor),
        verbose=bool(args.verbose),
    )
    paths = write_freesolv_small_feature_outputs(result, args.output_dir)
    summary = pd.read_csv(paths["summary"])
    print("FreeSolv small-feature MolADT RBF GP repeated splits")
    print(f"  seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} splits)")
    print(f"  output_dir: {args.output_dir}")
    for _, row in summary.loc[summary["split"] == "test"].sort_values("model").iterrows():
        print(
            f"  {row['model']} test RMSE: "
            f"{float(row['rmse_mean']):.6f} +/- {float(row['rmse_std']):.6f} kcal/mol"
        )
        print(
            f"  {row['model']} test MAE: "
            f"{float(row['mae_mean']):.6f} +/- {float(row['mae_std']):.6f} kcal/mol"
        )
    print(f"  summary: {paths['summary']}")
    print(f"  features: {paths['feature_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
