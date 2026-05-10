from __future__ import annotations

import argparse
from collections import Counter
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

from moladt.chem.dietz import Edge
from moladt.chem.molecule import Atom, Molecule, molecule_edges
from moladt.chem.molecule_ops import effective_order
from moladt.io.sdf import read_sdf_record

from .common import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directory
from .predictive_metrics import build_metric_row, build_prediction_rows
from .process_freesolv import FreeSolvArtifacts, process_freesolv_dataset
from .splits import ExportedDataset


MODEL_NAME = "moladt_wl_system_gp"
METHOD_NAME = "empirical_bayes_exact_gp"
REPRESENTATION = "moladt"
DEFAULT_SINGLE_SPLIT_SEED = 18
DEFAULT_SPLIT_COUNT = 20
SPLIT_SCHEME = "moleculenet_random_like:513/64/65;final_refit=train+valid"
TRAIN_SIZE = 513
VALID_SIZE = 64
TEST_SIZE = 65
MAX_OPTIMIZER_ITERATIONS = 120


@dataclass(frozen=True, slots=True)
class FreeSolvWLSystemResult:
    metric_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]
    coefficient_rows: list[dict[str, Any]]
    artifact_rows: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _Bundle:
    rows: pd.DataFrame
    molecules: tuple[Molecule, ...]
    mol_ids: tuple[str, ...]
    y: np.ndarray
    wl_matrix: np.ndarray
    system_matrix: np.ndarray
    wl_names: tuple[str, ...]
    system_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _GPFit:
    component_names: tuple[str, ...]
    weights: np.ndarray
    noise_variance: float
    y_mean: float
    y_scale: float
    alpha: np.ndarray
    cholesky: tuple[np.ndarray, bool]
    train_idx: np.ndarray
    full_kernel: np.ndarray
    nll: float


@dataclass(frozen=True, slots=True)
class WLSystemPredictorState:
    bundle: _Bundle
    fit: _GPFit
    seed: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray


def run_freesolv_wl_system_gp(
    artifacts: FreeSolvArtifacts,
    *,
    seed: int = DEFAULT_SINGLE_SPLIT_SEED,
    verbose: bool = False,
) -> FreeSolvWLSystemResult:
    if artifacts.moladt_featurized_export is None:
        raise RuntimeError("FreeSolv MolADT featurized export is required for the WL + bonding-system GP")
    start = time.perf_counter()
    bundle = _load_bundle(artifacts.moladt_featurized_export)
    components = _build_kernel_components(bundle)
    return _evaluate_wl_system_gp_bundle(
        bundle,
        artifacts.moladt_featurized_export,
        components=components,
        seed=seed,
        verbose=verbose,
        start_time=start,
    )


def run_freesolv_wl_system_gp_splits(
    artifacts: FreeSolvArtifacts,
    *,
    seeds: Sequence[int],
    verbose: bool = False,
) -> FreeSolvWLSystemResult:
    if artifacts.moladt_featurized_export is None:
        raise RuntimeError("FreeSolv MolADT featurized export is required for the WL + bonding-system GP")
    bundle = _load_bundle(artifacts.moladt_featurized_export)
    components = _build_kernel_components(bundle)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for seed in seeds:
        result = _evaluate_wl_system_gp_bundle(
            bundle,
            artifacts.moladt_featurized_export,
            components=components,
            seed=int(seed),
            verbose=verbose,
        )
        metric_rows.extend(result.metric_rows)
        prediction_rows.extend(result.prediction_rows)
        coefficient_rows.extend(result.coefficient_rows)
        artifact_rows.extend(result.artifact_rows)
    return FreeSolvWLSystemResult(
        metric_rows=metric_rows,
        prediction_rows=prediction_rows,
        coefficient_rows=coefficient_rows,
        artifact_rows=artifact_rows,
    )


def _evaluate_wl_system_gp_bundle(
    bundle: _Bundle,
    export: ExportedDataset,
    *,
    components: dict[str, np.ndarray],
    seed: int,
    verbose: bool = False,
    start_time: float | None = None,
) -> FreeSolvWLSystemResult:
    start = time.perf_counter() if start_time is None else start_time
    train_idx, valid_idx, test_idx = train_valid_test_indices(len(bundle.y), seed)

    if verbose:
        print(
            "[freesolv/wl-system-gp] "
            f"molecules={len(bundle.y)} wl_tokens={bundle.wl_matrix.shape[1]} "
            f"system_tokens={bundle.system_matrix.shape[1]} seed={seed}",
            flush=True,
        )

    selection_fit = _fit_gp(components, bundle.y, train_idx)
    valid_mean, valid_sd = _predict_gp(selection_fit, valid_idx)

    final_train_idx = np.sort(np.concatenate([train_idx, valid_idx]))
    final_fit = _fit_gp(components, bundle.y, final_train_idx)
    train_mean, train_sd = _predict_gp(final_fit, final_train_idx)
    test_mean, test_sd = _predict_gp(final_fit, test_idx)

    runtime = time.perf_counter() - start
    source_row_count = export.source_row_count
    used_row_count = int(len(train_idx) + len(valid_idx) + len(test_idx))
    feature_count = int(bundle.wl_matrix.shape[1] + bundle.system_matrix.shape[1])
    parameter_count = int(len(final_fit.weights) + 3)
    metric_rows = [
        build_metric_row(
            dataset_name="freesolv",
            representation=REPRESENTATION,
            model_name=MODEL_NAME,
            method=METHOD_NAME,
            split_name="train",
            mol_ids=tuple(bundle.mol_ids[int(index)] for index in final_train_idx),
            actual=bundle.y[final_train_idx],
            predicted_mean=train_mean,
            predictive_sd=train_sd,
            runtime_seconds=runtime,
            feature_count=feature_count,
            n_train=int(len(final_train_idx)),
            split_scheme=SPLIT_SCHEME,
            source_row_count=source_row_count,
            used_row_count=used_row_count,
            seed=seed,
            draw_count=1,
            parameter_count=parameter_count,
        ),
        build_metric_row(
            dataset_name="freesolv",
            representation=REPRESENTATION,
            model_name=MODEL_NAME,
            method=METHOD_NAME,
            split_name="valid",
            mol_ids=tuple(bundle.mol_ids[int(index)] for index in valid_idx),
            actual=bundle.y[valid_idx],
            predicted_mean=valid_mean,
            predictive_sd=valid_sd,
            runtime_seconds=runtime,
            feature_count=feature_count,
            n_train=int(len(train_idx)),
            split_scheme=SPLIT_SCHEME,
            source_row_count=source_row_count,
            used_row_count=used_row_count,
            seed=seed,
            draw_count=1,
            parameter_count=parameter_count,
            extra_metrics={"selection_nll": selection_fit.nll},
        ),
        build_metric_row(
            dataset_name="freesolv",
            representation=REPRESENTATION,
            model_name=MODEL_NAME,
            method=METHOD_NAME,
            split_name="test",
            mol_ids=tuple(bundle.mol_ids[int(index)] for index in test_idx),
            actual=bundle.y[test_idx],
            predicted_mean=test_mean,
            predictive_sd=test_sd,
            runtime_seconds=runtime,
            feature_count=feature_count,
            n_train=int(len(final_train_idx)),
            split_scheme=SPLIT_SCHEME,
            source_row_count=source_row_count,
            used_row_count=used_row_count,
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
            model_name=MODEL_NAME,
            method=METHOD_NAME,
            split_name="train",
            mol_ids=tuple(bundle.mol_ids[int(index)] for index in final_train_idx),
            actual=bundle.y[final_train_idx],
            predicted_mean=train_mean,
            predictive_sd=train_sd,
            seed=seed,
        ),
        *build_prediction_rows(
            dataset_name="freesolv",
            representation=REPRESENTATION,
            model_name=MODEL_NAME,
            method=METHOD_NAME,
            split_name="valid",
            mol_ids=tuple(bundle.mol_ids[int(index)] for index in valid_idx),
            actual=bundle.y[valid_idx],
            predicted_mean=valid_mean,
            predictive_sd=valid_sd,
            seed=seed,
        ),
        *build_prediction_rows(
            dataset_name="freesolv",
            representation=REPRESENTATION,
            model_name=MODEL_NAME,
            method=METHOD_NAME,
            split_name="test",
            mol_ids=tuple(bundle.mol_ids[int(index)] for index in test_idx),
            actual=bundle.y[test_idx],
            predicted_mean=test_mean,
            predictive_sd=test_sd,
            seed=seed,
        ),
    ]
    coefficient_rows = _coefficient_rows(final_fit, runtime_seconds=runtime)
    artifact_rows = [
        {
            "dataset": "freesolv",
            "representation": REPRESENTATION,
            "model": MODEL_NAME,
            "method": METHOD_NAME,
            "artifact_type": "moladt_token_gp",
            "seed": seed,
            "split_scheme": SPLIT_SCHEME,
            "train_mol_ids": ";".join(bundle.mol_ids[int(index)] for index in train_idx),
            "valid_mol_ids": ";".join(bundle.mol_ids[int(index)] for index in valid_idx),
            "test_mol_ids": ";".join(bundle.mol_ids[int(index)] for index in test_idx),
            "wl_token_count": int(bundle.wl_matrix.shape[1]),
            "system_token_count": int(bundle.system_matrix.shape[1]),
            "source_feature_table": str(export.feature_csv_path),
        }
    ]
    return FreeSolvWLSystemResult(
        metric_rows=metric_rows,
        prediction_rows=prediction_rows,
        coefficient_rows=coefficient_rows,
        artifact_rows=artifact_rows,
    )


def write_freesolv_wl_system_split_outputs(
    result: FreeSolvWLSystemResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = ensure_directory(output_dir)
    metrics_path = output_dir / "predictive_metrics.csv"
    predictions_path = output_dir / "predictions.csv"
    coefficients_path = output_dir / "model_coefficients.csv"
    artifacts_path = output_dir / "model_artifacts.csv"
    assignments_path = output_dir / "split_assignments.csv"
    summary_path = output_dir / "summary.csv"
    pd.DataFrame(result.metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(result.prediction_rows).to_csv(predictions_path, index=False)
    pd.DataFrame(result.coefficient_rows).to_csv(coefficients_path, index=False)
    artifact_frame = pd.DataFrame(result.artifact_rows)
    artifact_frame.to_csv(artifacts_path, index=False)
    assignment_frame = pd.DataFrame(_split_assignment_rows(result.artifact_rows))
    assignment_frame.to_csv(assignments_path, index=False)
    summary_frame = _split_summary_frame(pd.DataFrame(result.metric_rows))
    summary_frame.to_csv(summary_path, index=False)
    return {
        "predictive_metrics": metrics_path,
        "predictions": predictions_path,
        "model_coefficients": coefficients_path,
        "model_artifacts": artifacts_path,
        "split_assignments": assignments_path,
        "summary": summary_path,
    }


def load_wl_system_predictor_state(seed: int = DEFAULT_SINGLE_SPLIT_SEED) -> WLSystemPredictorState:
    rows = pd.read_csv(PROCESSED_DATA_DIR / "freesolv_moladt_featurized_features.csv")
    bundle = _load_bundle_from_rows(rows)
    train_idx, valid_idx, test_idx = train_valid_test_indices(len(bundle.y), seed)
    final_train_idx = np.sort(np.concatenate([train_idx, valid_idx]))
    fit = _fit_gp(_build_kernel_components(bundle), bundle.y, final_train_idx)
    return WLSystemPredictorState(
        bundle=bundle,
        fit=fit,
        seed=seed,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
    )


def predict_with_wl_system_state(state: WLSystemPredictorState, molecule: Molecule) -> tuple[float, float]:
    wl_vector = _counter_to_vector(_graph_token_counts(molecule), state.bundle.wl_names)
    system_vector = _counter_to_vector(_system_token_counts(molecule), state.bundle.system_names)
    train_idx = state.fit.train_idx
    wl_cross = _tanimoto_cross(state.bundle.wl_matrix[train_idx], wl_vector)
    system_cross = _tanimoto_cross(state.bundle.system_matrix[train_idx], system_vector)
    train_combined = np.hstack([state.bundle.wl_matrix[train_idx], state.bundle.system_matrix[train_idx]])
    eval_combined = np.concatenate([wl_vector, system_vector])
    wl_system_cross = _tanimoto_cross(train_combined, eval_combined)
    component_cross = {
        "wl_system_tanimoto": wl_system_cross,
        "system_tanimoto": system_cross,
        "wl_tanimoto": wl_cross,
    }
    cross = np.zeros(len(train_idx), dtype=float)
    self_kernel = 0.0
    for name, weight in zip(state.fit.component_names, state.fit.weights, strict=True):
        cross += weight * component_cross[name]
        self_kernel += weight * _self_tanimoto(
            eval_combined if name == "wl_system_tanimoto" else system_vector if name == "system_tanimoto" else wl_vector
        )
    mean_scaled = float(cross @ state.fit.alpha)
    solved = cho_solve(state.fit.cholesky, cross, check_finite=False)
    latent_var = self_kernel - float(cross @ solved)
    observed_var = max(latent_var + state.fit.noise_variance, 1.0e-10)
    mean = state.fit.y_mean + state.fit.y_scale * mean_scaled
    sd = state.fit.y_scale * math.sqrt(observed_var)
    return float(mean), float(sd)


def train_valid_test_indices(row_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if row_count != TRAIN_SIZE + VALID_SIZE + TEST_SIZE:
        raise ValueError(f"FreeSolv WL + bonding-system GP expects 642 rows; found {row_count}")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(row_count)
    train = np.sort(permutation[:TRAIN_SIZE])
    valid = np.sort(permutation[TRAIN_SIZE : TRAIN_SIZE + VALID_SIZE])
    test = np.sort(permutation[TRAIN_SIZE + VALID_SIZE :])
    return train, valid, test


def _load_bundle(export: ExportedDataset) -> _Bundle:
    return _load_bundle_from_rows(export.rows)


def _load_bundle_from_rows(rows: pd.DataFrame) -> _Bundle:
    rows = rows.copy()
    rows.loc[:, "mol_id"] = rows["mol_id"].astype(str)
    rows = rows.sort_values("mol_id").reset_index(drop=True)
    processed = pd.read_csv(PROCESSED_DATA_DIR / "freesolv_processed.csv")
    processed.loc[:, "mol_id"] = processed["mol_id"].astype(str)
    relpaths = processed.set_index("mol_id")["sdf_relpath"].to_dict()
    molecules: list[Molecule] = []
    for mol_id in rows["mol_id"]:
        relpath = relpaths.get(str(mol_id))
        if relpath is None:
            raise RuntimeError(f"FreeSolv row {mol_id} has no SDF path in freesolv_processed.csv")
        molecules.append(read_sdf_record(RAW_DATA_DIR / "freesolv" / str(relpath)).molecule)
    wl_counters = [_graph_token_counts(molecule) for molecule in molecules]
    system_counters = [_system_token_counts(molecule) for molecule in molecules]
    wl_matrix, wl_names = _vectorize(wl_counters)
    system_matrix, system_names = _vectorize(system_counters)
    return _Bundle(
        rows=rows,
        molecules=tuple(molecules),
        mol_ids=tuple(rows["mol_id"].astype(str).tolist()),
        y=rows["expt"].to_numpy(dtype=float),
        wl_matrix=wl_matrix,
        system_matrix=system_matrix,
        wl_names=tuple(wl_names),
        system_names=tuple(system_names),
    )


def _shell_stats(atom: Atom) -> tuple[int, int, int, str]:
    shell_count = 0
    orbital_count = 0
    electron_count = 0
    parts: list[str] = []
    for shell in atom.shells or ():
        shell_count += 1
        principal = getattr(shell, "principal_quantum_number", 0)
        for name in ("s_subshell", "p_subshell", "d_subshell", "f_subshell"):
            subshell = getattr(shell, name, None)
            if subshell is None:
                continue
            counts: list[str] = []
            for orbital in getattr(subshell, "orbitals", ()):
                count = int(getattr(orbital, "electron_count", 0))
                orbital_count += 1
                electron_count += count
                counts.append(str(count))
            if counts:
                parts.append(f"{principal}{name[0]}{''.join(counts)}")
    return shell_count, orbital_count, electron_count, ".".join(parts)


def _atom_symbol(atom: Atom) -> str:
    return atom.attributes.symbol.value


def _edge_symbol_pair(molecule: Molecule, edge: Edge) -> str:
    left = _atom_symbol(molecule.atoms[edge.a])
    right = _atom_symbol(molecule.atoms[edge.b])
    return "-".join(sorted((left, right)))


def _charge_bucket(charge: int) -> str:
    if charge <= -3:
        return "neg3plus"
    if charge < 0:
        return f"neg{abs(charge)}"
    if charge == 0:
        return "neutral"
    if charge >= 3:
        return "pos3plus"
    return f"pos{charge}"


def _order_bucket(order: float) -> str:
    if order <= 0.25:
        return "ionic_zero"
    if order < 1.25:
        return "single"
    if order < 1.80:
        return "delocalised_1p5"
    if order < 2.50:
        return "double"
    if order < 3.50:
        return "triple"
    return "quadruple_plus"


def _system_kind(system: Any) -> str:
    if system.tag:
        return str(system.tag)
    if len(system.member_edges) == 1 and system.shared_electrons.value in {2, 4, 6, 8}:
        return {
            2: "single_covalent",
            4: "double_covalent",
            6: "triple_covalent",
            8: "quadruple_covalent",
        }[system.shared_electrons.value]
    if system.shared_electrons.value == 0:
        return "zero_electron"
    if len(system.member_edges) > 1:
        return "delocalised_bonding"
    return "other_bonding"


def _graph_token_counts(molecule: Molecule, radius: int = 4) -> Counter[str]:
    atoms = sorted(molecule.atoms)
    systems = [system for _, system in molecule.systems]
    adjacency: dict[Any, list[tuple[Any, str]]] = {}
    edge_labels: dict[Edge, str] = {}
    for edge in sorted(molecule_edges(molecule)):
        containing = [system for system in systems if edge in system.member_edges]
        signature = ".".join(
            sorted(f"{system.shared_electrons.value}e:{len(system.member_edges)}m:{_system_kind(system)}" for system in containing)
        )
        edge_label = (
            f"{_edge_symbol_pair(molecule, edge)}:{_order_bucket(effective_order(molecule, edge))}:"
            f"overlap{len(containing)}:{signature}"
        )
        edge_labels[edge] = edge_label
        adjacency.setdefault(edge.a, []).append((edge.b, edge_label))
        adjacency.setdefault(edge.b, []).append((edge.a, edge_label))
    labels: dict[Any, str] = {}
    for atom_id in atoms:
        atom = molecule.atoms[atom_id]
        shell_count, orbital_count, electron_count, shell_signature = _shell_stats(atom)
        labels[atom_id] = (
            f"{_atom_symbol(atom)}:{_charge_bucket(atom.formal_charge)}:"
            f"sh{shell_count}:orb{orbital_count}:e{electron_count}:{shell_signature}"
        )
    counts: Counter[str] = Counter()
    for label in labels.values():
        counts[f"wl0:{label}"] += 1
    for edge_label in edge_labels.values():
        counts[f"edge_label:{edge_label}"] += 1
    current = labels
    for step in range(1, radius + 1):
        updated: dict[Any, str] = {}
        for atom_id in atoms:
            neighborhood = tuple(sorted((edge_label, current[neighbor]) for neighbor, edge_label in adjacency.get(atom_id, ())))
            updated[atom_id] = f"{current[atom_id]}|{neighborhood}"
        current = updated
        for label in current.values():
            counts[f"wl{step}:{label}"] += 1
    return counts


def _system_token_counts(molecule: Molecule) -> Counter[str]:
    counts: Counter[str] = Counter()
    systems = [system for _, system in molecule.systems]
    for atom in molecule.atoms.values():
        shell_count, orbital_count, electron_count, shell_signature = _shell_stats(atom)
        counts[f"atom:{_atom_symbol(atom)}:{_charge_bucket(atom.formal_charge)}"] += 1
        counts[
            f"atom_shell:{_atom_symbol(atom)}:sh{shell_count}:orb{orbital_count}:"
            f"e{electron_count}:{shell_signature}"
        ] += 1
    for edge in sorted(molecule_edges(molecule)):
        pair = _edge_symbol_pair(molecule, edge)
        order = _order_bucket(effective_order(molecule, edge))
        containing = [system for system in systems if edge in system.member_edges]
        charges = tuple(sorted((molecule.atoms[edge.a].formal_charge, molecule.atoms[edge.b].formal_charge)))
        counts[f"edge:{pair}:{order}"] += 1
        counts[f"edge_overlap:{pair}:{order}:{len(containing)}"] += 1
        counts[f"edge_charge:{pair}:{charges[0]}:{charges[1]}:{order}"] += 1
        for system in containing:
            counts[f"edge_in_system:{pair}:{order}:{system.shared_electrons.value}:{len(system.member_edges)}:{_system_kind(system)}"] += 1
    for system in systems:
        atom_symbols = ".".join(sorted(_atom_symbol(molecule.atoms[atom_id]) for atom_id in system.member_atoms))
        edge_pairs = ".".join(sorted(_edge_symbol_pair(molecule, edge) for edge in system.member_edges))
        counts[f"system:{system.shared_electrons.value}:{len(system.member_atoms)}:{len(system.member_edges)}:{_system_kind(system)}"] += 1
        counts[f"system_atoms:{system.shared_electrons.value}:{len(system.member_edges)}:{_system_kind(system)}:{atom_symbols}"] += 1
        counts[f"system_edges:{system.shared_electrons.value}:{len(system.member_atoms)}:{_system_kind(system)}:{edge_pairs}"] += 1
    return counts


def _vectorize(counters: list[Counter[str]]) -> tuple[np.ndarray, list[str]]:
    names = sorted({key for counter in counters for key, value in counter.items() if value})
    index = {name: column for column, name in enumerate(names)}
    matrix = np.zeros((len(counters), len(names)), dtype=float)
    for row, counter in enumerate(counters):
        for key, value in counter.items():
            column = index.get(key)
            if column is not None:
                matrix[row, column] = float(value)
    return matrix, names


def _counter_to_vector(counter: Counter[str], names: tuple[str, ...]) -> np.ndarray:
    vector = np.zeros(len(names), dtype=float)
    for column, name in enumerate(names):
        vector[column] = float(counter.get(name, 0.0))
    return vector


def _tanimoto_kernel(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.eye(matrix.shape[0], dtype=float)
    dot = matrix @ matrix.T
    norm = np.sum(matrix * matrix, axis=1)
    denom = norm[:, None] + norm[None, :] - dot
    return np.nan_to_num(dot / np.maximum(denom, 1e-12), nan=0.0, posinf=0.0, neginf=0.0)


def _tanimoto_cross(train_matrix: np.ndarray, eval_vector: np.ndarray) -> np.ndarray:
    if train_matrix.shape[1] == 0:
        return np.zeros(train_matrix.shape[0], dtype=float)
    dot = train_matrix @ eval_vector
    train_norm = np.sum(train_matrix * train_matrix, axis=1)
    eval_norm = float(eval_vector @ eval_vector)
    denom = train_norm + eval_norm - dot
    return np.nan_to_num(dot / np.maximum(denom, 1e-12), nan=0.0, posinf=0.0, neginf=0.0)


def _self_tanimoto(vector: np.ndarray) -> float:
    return 1.0 if float(vector @ vector) > 1.0e-12 else 0.0


def _build_kernel_components(bundle: _Bundle) -> dict[str, np.ndarray]:
    wl = _tanimoto_kernel(bundle.wl_matrix)
    system = _tanimoto_kernel(bundle.system_matrix)
    wl_system = _tanimoto_kernel(np.hstack([bundle.wl_matrix, bundle.system_matrix]))
    return {
        "wl_system_tanimoto": wl_system,
        "system_tanimoto": system,
        "wl_tanimoto": wl,
    }


def _fit_gp(components: dict[str, np.ndarray], y: np.ndarray, train_idx: np.ndarray) -> _GPFit:
    component_names = tuple(("wl_system_tanimoto", "system_tanimoto", "wl_tanimoto"))
    matrices = [components[name] for name in component_names]
    y_train_raw = y[train_idx]
    y_mean = float(y_train_raw.mean())
    y_scale = float(y_train_raw.std(ddof=0))
    if y_scale < 1e-12:
        y_scale = 1.0
    y_train = (y_train_raw - y_mean) / y_scale
    train_matrices = [matrix[np.ix_(train_idx, train_idx)] for matrix in matrices]
    n = len(train_idx)

    def build_kernel(log_params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        weights = np.exp(log_params[:-1])
        noise = float(np.exp(log_params[-1]))
        kernel = np.zeros((n, n), dtype=float)
        for weight, matrix in zip(weights, train_matrices, strict=True):
            kernel += weight * matrix
        kernel += (noise + 1.0e-7) * np.eye(n)
        return kernel, weights, noise

    def objective(log_params: np.ndarray) -> float:
        kernel, _, _ = build_kernel(log_params)
        try:
            cholesky = cho_factor(kernel, lower=True, check_finite=False)
            alpha = cho_solve(cholesky, y_train, check_finite=False)
        except (LinAlgError, ValueError):
            return 1.0e25
        logdet = 2.0 * np.sum(np.log(np.diag(cholesky[0])))
        return float(0.5 * y_train @ alpha + 0.5 * logdet + 0.5 * n * np.log(2.0 * math.pi))

    initial = np.log(np.asarray([1.0 / len(component_names)] * len(component_names) + [0.05], dtype=float))
    bounds = [(-8.0, 4.0)] * len(component_names) + [(-9.0, 2.0)]
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": MAX_OPTIMIZER_ITERATIONS, "ftol": 1e-7, "maxls": 25},
    )
    params = result.x if result.success or np.isfinite(result.fun) else initial
    kernel_train, weights, noise = build_kernel(params)
    cholesky = cho_factor(kernel_train, lower=True, check_finite=False)
    alpha = cho_solve(cholesky, y_train, check_finite=False)
    full_kernel = np.zeros_like(matrices[0], dtype=float)
    for weight, matrix in zip(weights, matrices, strict=True):
        full_kernel += weight * matrix
    return _GPFit(
        component_names=component_names,
        weights=weights,
        noise_variance=noise,
        y_mean=y_mean,
        y_scale=y_scale,
        alpha=alpha,
        cholesky=cholesky,
        train_idx=train_idx.copy(),
        full_kernel=full_kernel,
        nll=float(objective(params)),
    )


def _predict_gp(fit: _GPFit, target_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cross = fit.full_kernel[np.ix_(fit.train_idx, target_idx)]
    mean_scaled = cross.T @ fit.alpha
    solved = cho_solve(fit.cholesky, cross, check_finite=False)
    diag = np.maximum(np.diag(fit.full_kernel)[target_idx], 0.0)
    latent_var = diag - np.sum(cross * solved, axis=0)
    observed_var = np.maximum(latent_var + fit.noise_variance, 1.0e-10)
    mean = fit.y_mean + fit.y_scale * mean_scaled
    sd = fit.y_scale * np.sqrt(observed_var)
    return mean, sd


def _coefficient_rows(fit: _GPFit, *, runtime_seconds: float) -> list[dict[str, Any]]:
    values = {
        **{f"kernel_weight[{name}]": float(value) for name, value in zip(fit.component_names, fit.weights, strict=True)},
        "noise_variance": float(fit.noise_variance),
        "target_mean": float(fit.y_mean),
        "target_scale": float(fit.y_scale),
    }
    rows = []
    for rank, (name, value) in enumerate(values.items(), start=1):
        rows.append(
            {
                "dataset": "freesolv",
                "representation": REPRESENTATION,
                "target": "expt",
                "model": MODEL_NAME,
                "method": METHOD_NAME,
                "parameter_type": "kernel_hyperparameter",
                "parameter_name": name,
                "feature_group": "moladt_wl_system",
                "equation_term": name,
                "draw_count": 1,
                "runtime_seconds": runtime_seconds,
                "posterior_mean": value,
                "posterior_abs_mean": abs(value),
                "posterior_sd": 0.0,
                "posterior_median": value,
                "posterior_p05": value,
                "posterior_p95": value,
                "importance_rank": rank,
            }
        )
    return rows


def _split_assignment_rows(artifact_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifact_rows:
        seed = int(artifact["seed"])
        for split_name, field_name in (
            ("train", "train_mol_ids"),
            ("valid", "valid_mol_ids"),
            ("test", "test_mol_ids"),
        ):
            mol_ids = [mol_id for mol_id in str(artifact.get(field_name, "")).split(";") if mol_id]
            rows.extend({"seed": seed, "split": split_name, "mol_id": mol_id} for mol_id in mol_ids)
    return rows


def _split_summary_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    metric_columns = ["rmse", "mae", "r2", "predictive_sd_mean", "coverage_90"]
    rows: list[dict[str, Any]] = []
    for split_name, frame in metrics.groupby("split", sort=True):
        row: dict[str, Any] = {
            "split": split_name,
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


def _default_split_output_dir() -> Path:
    return Path("results") / "freesolv_20split" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.freesolv_wl_system_gp",
        description="Run the MolADT WL + bonding-system FreeSolv GP over repeated random splits.",
    )
    parser.add_argument("--split-count", type=int, default=DEFAULT_SPLIT_COUNT)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=_default_split_output_dir())
    parser.add_argument("--force", action="store_true", help="Regenerate processed FreeSolv exports before fitting")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split_count <= 0:
        raise ValueError("--split-count must be positive")
    seeds = tuple(range(int(args.seed_start), int(args.seed_start) + int(args.split_count)))
    artifacts = process_freesolv_dataset(
        seed=DEFAULT_SINGLE_SPLIT_SEED,
        force=bool(args.force),
        include_moladt=True,
        include_legacy_tabular=False,
        verbose=bool(args.verbose),
    )
    result = run_freesolv_wl_system_gp_splits(artifacts, seeds=seeds, verbose=bool(args.verbose))
    paths = write_freesolv_wl_system_split_outputs(result, args.output_dir)
    summary = pd.read_csv(paths["summary"])
    test_summary = summary.loc[summary["split"] == "test"]
    print("FreeSolv MolADT WL + bonding-system GP repeated splits")
    print(f"  seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} splits)")
    print(f"  output_dir: {args.output_dir}")
    if not test_summary.empty:
        row = test_summary.iloc[0]
        print(
            "  test RMSE: "
            f"{float(row['rmse_mean']):.6f} +/- {float(row['rmse_std']):.6f} kcal/mol"
        )
        print(
            "  test MAE: "
            f"{float(row['mae_mean']):.6f} +/- {float(row['mae_std']):.6f} kcal/mol"
        )
    print(f"  summary: {paths['summary']}")
    print(f"  split assignments: {paths['split_assignments']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
