from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable, Protocol, Sequence, TextIO

import numpy as np
import pandas as pd

from moladt.chem.constants import element_attributes, element_shells
from moladt.chem.coordinate import Coordinate, mk_angstrom
from moladt.chem.dietz import AtomId, BondingSystem, Edge, NonNegative, SystemId, mk_bonding_system, mk_edge
from moladt.chem.molecule import Atom, AtomicSymbol, Molecule
from moladt.chem.molecule_ops import effective_order, neighbors_sigma
from moladt.chem.mutable import MutableMolecule
from moladt.chem.validate import ValidationError, used_electrons_at, validate_molecule
from moladt.examples.sample_molecules import methane, water
from moladt.io import molecule_to_json_bytes
from scripts.common import PROCESSED_DATA_DIR, PROJECT_ROOT, configured_results_dir, ensure_directory
from scripts.features import compute_moladt_featurized_descriptors
from scripts.stan_runner import GP_SCREENED_FEATURE_COUNT


N_STEPS = 2000
N_SEEDS = 5
TOP_K = 10
TOP_DIETZ_K = 5
MAX_HEAVY_ATOMS = 12
TEMPERATURE = 1.0
RANDOM_SEED = 0
MAX_TOTAL_ATOMS = 4 * MAX_HEAVY_ATOMS + 8
HEAVY_ATOM_GROWTH_LIMIT = MAX_HEAVY_ATOMS + 2
DEFAULT_SEED_MOLECULE = "water"
REFERENCE_RESULTS_ENV = "MOLADT_REFERENCE_RESULTS_DIR"

DATASET_PREFIX = "freesolv_moladt_featurized"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "results" / "freesolv" / "run_20260417_162536"
MODEL_NAME = "bayes_gp_rbf_screened"
METHOD_NAME = "laplace"
TARGET_NAME = "expt"

ALLOWED_FREE_SOLV_SYMBOLS = (
    AtomicSymbol.H,
    AtomicSymbol.C,
    AtomicSymbol.N,
    AtomicSymbol.O,
    AtomicSymbol.F,
    AtomicSymbol.Cl,
)
NEW_ATOM_SYMBOLS = (
    AtomicSymbol.C,
    AtomicSymbol.C,
    AtomicSymbol.O,
    AtomicSymbol.N,
    AtomicSymbol.F,
    AtomicSymbol.Cl,
)
MUTATION_SYMBOLS = tuple(symbol for symbol in ALLOWED_FREE_SOLV_SYMBOLS if symbol is not AtomicSymbol.H)

MOVE_WEIGHTS = (
    ("add_terminal_atom", 0.40),
    ("add_sigma_edge", 0.25),
    ("mutate_atom", 0.20),
    ("remove_terminal_atom", 0.10),
    ("add_pi_ring_system", 0.05),
)

GROWTH_MAX_VALENCE = {
    AtomicSymbol.H: 1.0,
    AtomicSymbol.C: 4.0,
    AtomicSymbol.N: 3.0,
    AtomicSymbol.O: 2.0,
    AtomicSymbol.F: 1.0,
    AtomicSymbol.Cl: 1.0,
}


@dataclass(frozen=True, slots=True)
class Prediction:
    mean: float
    sd: float


class FreeSolvPredictor(Protocol):
    def predict(self, molecule: Molecule) -> Prediction:
        ...


@dataclass(frozen=True, slots=True)
class Candidate:
    molecule: Molecule
    predicted_mean: float
    predictive_sd: float
    score: float


@dataclass(slots=True)
class SearchDiagnostics:
    total_proposals: int = 0
    valid_proposals: int = 0
    invalid_proposals: int = 0
    accepted_proposals: int = 0
    unique_valid_molecules_seen: int = 0


@dataclass(frozen=True, slots=True)
class SearchResult:
    target: float
    used_default_target: bool
    seed_molecule: str
    top_candidates: tuple[Candidate, ...]
    diagnostics: SearchDiagnostics
    dietz_candidates: tuple[Candidate, ...] = ()
    molecule_file_paths: tuple[Path, ...] = ()
    dietz_file_paths: tuple[Path, ...] = ()
    model_parameter_source: Path | None = None


@dataclass(frozen=True, slots=True)
class FreeSolvModelParameters:
    alpha: float
    signal_scale: float
    lengthscale: float
    sigma: float
    source_path: Path


@dataclass(frozen=True, slots=True)
class FreeSolvBayesianPredictor:
    feature_names: tuple[str, ...]
    train_mean: np.ndarray
    train_std: np.ndarray
    selected_indices: tuple[int, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    alpha: float
    signal_scale: float
    lengthscale: float
    sigma: float
    chol: np.ndarray
    weight: np.ndarray
    parameter_source_path: Path

    @classmethod
    def load(cls) -> FreeSolvBayesianPredictor:
        metadata_path = PROCESSED_DATA_DIR / f"{DATASET_PREFIX}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing FreeSolv metadata: {metadata_path}")
        metadata = pd.read_json(metadata_path, typ="series")
        _validate_freesolv_metadata(metadata, metadata_path)
        feature_names = tuple(str(name) for name in metadata["feature_names"])
        train_mean = np.asarray(metadata["train_mean"], dtype=float)
        train_std = np.asarray(metadata["train_std"], dtype=float)

        X_train_all = pd.read_csv(PROCESSED_DATA_DIR / f"{DATASET_PREFIX}_X_train.csv").to_numpy(dtype=float)
        y_frame = pd.read_csv(PROCESSED_DATA_DIR / f"{DATASET_PREFIX}_y_train.csv")
        y_train = y_frame.iloc[:, 0].to_numpy(dtype=float)
        selected_indices = _screen_top_correlation_features(X_train_all, y_train, top_k=GP_SCREENED_FEATURE_COUNT)
        X_train = X_train_all[:, selected_indices]
        parameters = _load_gp_parameter_means()

        train_kernel = _rbf_kernel(X_train, X_train, lengthscale=parameters.lengthscale, signal_scale=parameters.signal_scale)
        train_kernel[np.diag_indices_from(train_kernel)] += parameters.sigma**2 + 1e-8
        chol = np.linalg.cholesky(train_kernel)
        centered_y = y_train - parameters.alpha
        weight = np.linalg.solve(chol.T, np.linalg.solve(chol, centered_y))
        return cls(
            feature_names=feature_names,
            train_mean=train_mean,
            train_std=train_std,
            selected_indices=selected_indices,
            X_train=X_train,
            y_train=y_train,
            alpha=parameters.alpha,
            signal_scale=parameters.signal_scale,
            lengthscale=parameters.lengthscale,
            sigma=parameters.sigma,
            chol=chol,
            weight=weight,
            parameter_source_path=parameters.source_path,
        )

    def predict(self, molecule: Molecule) -> Prediction:
        descriptors = compute_moladt_featurized_descriptors(molecule)
        raw = np.asarray([float(descriptors.get(name, 0.0)) for name in self.feature_names], dtype=float)
        standardized = (raw - self.train_mean) / self.train_std
        x_eval = standardized[list(self.selected_indices)].reshape(1, -1)
        cross_kernel = _rbf_kernel(x_eval, self.X_train, lengthscale=self.lengthscale, signal_scale=self.signal_scale)
        mean = float((self.alpha + cross_kernel @ self.weight)[0])
        solve = np.linalg.solve(self.chol, cross_kernel.T)
        marginal_var = self.signal_scale**2 + self.sigma**2 - float(np.sum(np.square(solve), axis=0)[0])
        return Prediction(mean=mean, sd=math.sqrt(max(marginal_var, 1e-9)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.freesolv_inverse_design",
        description="Run the deterministic FreeSolv MolADT inverse-design proof of concept.",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Target hydration free energy. If omitted, use the median FreeSolv experimental value.",
    )
    parser.add_argument(
        "--seed-molecule",
        choices=("water", "methane", "methanol", "carbon-six-ring", "pi-carbon-six-ring"),
        default=DEFAULT_SEED_MOLECULE,
        help="Starting molecule for all deterministic search chains. Defaults to water.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    n_steps: int = N_STEPS,
    n_seeds: int = N_SEEDS,
    top_k: int = TOP_K,
    predictor: FreeSolvPredictor | None = None,
    stream: TextIO | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_inverse_design(
        target=args.target,
        n_steps=n_steps,
        n_seeds=n_seeds,
        top_k=top_k,
        predictor=predictor,
        seed_molecule=args.seed_molecule,
    )
    if predictor is None:
        result = write_result_molecule_files(result, configured_results_dir())
        reference_results_dir = os.environ.get(REFERENCE_RESULTS_ENV)
        if reference_results_dir:
            write_result_molecule_files(result, _resolve_output_dir(reference_results_dir))
    print_report(result, stream=stream or sys.stdout)
    return 0


def run_inverse_design(
    *,
    target: float | None = None,
    n_steps: int = N_STEPS,
    n_seeds: int = N_SEEDS,
    top_k: int = TOP_K,
    rng_seed: int = RANDOM_SEED,
    predictor: FreeSolvPredictor | None = None,
    seed_molecule: str = DEFAULT_SEED_MOLECULE,
) -> SearchResult:
    used_default_target = target is None
    resolved_target = default_target_from_freesolv_dataset() if target is None else float(target)
    active_predictor = predictor or FreeSolvBayesianPredictor.load()
    model_parameter_source = getattr(active_predictor, "parameter_source_path", None)

    candidate_by_key: dict[bytes, Candidate] = {}
    diagnostics = SearchDiagnostics()
    seeds = load_seed_molecules(seed_molecule=seed_molecule, n_seeds=n_seeds)

    for seed_index, starting_molecule in enumerate(seeds):
        rng = random.Random(rng_seed + seed_index)
        current = _validate_candidate(starting_molecule)
        current_candidate = _score_molecule(active_predictor, current, resolved_target)
        candidate_by_key[_molecule_key(current)] = current_candidate

        for _ in range(n_steps):
            diagnostics.total_proposals += 1
            proposal = propose_molecule(current, rng)
            if proposal is None:
                diagnostics.invalid_proposals += 1
                continue
            try:
                proposal = _validate_candidate(proposal)
            except (ValueError, ValidationError):
                diagnostics.invalid_proposals += 1
                continue

            diagnostics.valid_proposals += 1
            proposal_key = _molecule_key(proposal)
            existing_candidate = candidate_by_key.get(proposal_key)
            if existing_candidate is None:
                proposal_candidate = _score_molecule(active_predictor, proposal, resolved_target)
                candidate_by_key[proposal_key] = proposal_candidate
            else:
                proposal_candidate = existing_candidate

            score_delta = proposal_candidate.score - current_candidate.score
            if score_delta >= 0.0 or rng.random() < math.exp(score_delta / TEMPERATURE):
                current = _validate_candidate(proposal)
                current_candidate = proposal_candidate
                diagnostics.accepted_proposals += 1

    diagnostics.unique_valid_molecules_seen = len(candidate_by_key)
    top_candidates = tuple(
        sorted(
            candidate_by_key.values(),
            key=_candidate_sort_key(resolved_target),
            reverse=True,
        )[:top_k]
    )
    dietz_candidates = tuple(
        sorted(
            (
                candidate
                for candidate in candidate_by_key.values()
                if candidate.molecule.systems
            ),
            key=_candidate_sort_key(resolved_target),
            reverse=True,
        )[: min(top_k, TOP_DIETZ_K)]
    )
    return SearchResult(
        target=resolved_target,
        used_default_target=used_default_target,
        seed_molecule=seed_molecule,
        top_candidates=top_candidates,
        dietz_candidates=dietz_candidates,
        diagnostics=diagnostics,
        model_parameter_source=model_parameter_source,
    )


def load_seed_molecules(*, seed_molecule: str = DEFAULT_SEED_MOLECULE, n_seeds: int = N_SEEDS) -> tuple[Molecule, ...]:
    if n_seeds <= 0:
        return ()
    seed_builders: dict[str, Callable[[], Molecule]] = {
        "water": lambda: water,
        "methane": lambda: methane,
        "methanol": _methanol_seed,
        "carbon-six-ring": _carbon_six_ring_seed,
        "pi-carbon-six-ring": _pi_carbon_six_ring_seed,
    }
    try:
        seed_builder = seed_builders[seed_molecule]
    except KeyError as exc:
        available = ", ".join(sorted(seed_builders))
        raise ValueError(f"Unknown seed molecule {seed_molecule!r}; choose one of: {available}") from exc
    seeds = tuple(seed_builder() for _ in range(n_seeds))
    return tuple(_validate_candidate(molecule) for molecule in seeds)


def default_target_from_freesolv_dataset() -> float:
    values: list[np.ndarray] = []
    for split in ("train", "valid", "test"):
        path = PROCESSED_DATA_DIR / f"{DATASET_PREFIX}_y_{split}.csv"
        if path.exists():
            values.append(pd.read_csv(path).iloc[:, 0].to_numpy(dtype=float))
    if not values:
        raise FileNotFoundError("Missing processed FreeSolv target CSVs")
    return float(np.median(np.concatenate(values)))


def propose_molecule(molecule: Molecule, rng: random.Random) -> Molecule | None:
    move_name = _weighted_choice(MOVE_WEIGHTS, rng)
    move = _MOVE_FUNCTIONS[move_name]
    return move(molecule, rng)


def add_terminal_atom(molecule: Molecule, rng: random.Random) -> Molecule | None:
    if len(molecule.atoms) >= MAX_TOTAL_ATOMS:
        return None
    parents = [
        atom_id
        for atom_id, atom in molecule.atoms.items()
        if atom.attributes.symbol is not AtomicSymbol.H
        and (_available_valence(molecule, atom_id) >= 1.0 - 1e-9 or _terminal_hydrogens_attached_to(molecule, atom_id))
    ]
    if not parents:
        return None

    parent_id = rng.choice(sorted(parents))
    allowed_new_symbols = (
        (AtomicSymbol.H,)
        if heavy_atom_count(molecule) >= HEAVY_ATOM_GROWTH_LIMIT
        else NEW_ATOM_SYMBOLS
    )
    new_symbol = rng.choice(allowed_new_symbols)
    new_id = AtomId(max(atom_id.value for atom_id in molecule.atoms) + 1)
    parent_atom = molecule.atoms[parent_id]
    mutable = MutableMolecule.from_molecule(molecule)
    _free_one_valence_slot(mutable, parent_id)
    mutable.atoms[new_id] = _new_atom(new_id, new_symbol, parent_atom)
    mutable.local_bonds.add(mk_edge(parent_id, new_id))
    return _try_valid_candidate(mutable.freeze())


def add_sigma_edge(molecule: Molecule, rng: random.Random) -> Molecule | None:
    candidates: list[tuple[AtomId, AtomId]] = []
    ring_candidates: list[tuple[AtomId, AtomId]] = []
    atom_ids = sorted(
        atom_id
        for atom_id, atom in molecule.atoms.items()
        if atom.attributes.symbol is not AtomicSymbol.H
        and (_available_valence(molecule, atom_id) >= 1.0 - 1e-9 or _terminal_hydrogens_attached_to(molecule, atom_id))
    )
    for left_index, left in enumerate(atom_ids):
        for right in atom_ids[left_index + 1 :]:
            edge = mk_edge(left, right)
            if edge in molecule.local_bonds or _has_localized_singleton_system(molecule, edge):
                continue
            pair = (left, right)
            path_length = _shortest_sigma_path_length(molecule, left, right)
            if path_length is None or path_length < 4 or path_length > 6:
                continue
            candidates.append(pair)
            if path_length == 5:
                ring_candidates.append(pair)

    if not candidates:
        return None
    left, right = rng.choice(ring_candidates if ring_candidates and rng.random() < 0.75 else candidates)
    mutable = MutableMolecule.from_molecule(molecule)
    _free_one_valence_slot(mutable, left)
    _free_one_valence_slot(mutable, right)
    mutable.local_bonds.add(mk_edge(left, right))
    return _try_valid_candidate(mutable.freeze())


def mutate_atom(molecule: Molecule, rng: random.Random) -> Molecule | None:
    candidates = [
        atom_id
        for atom_id, atom in molecule.atoms.items()
        if atom.attributes.symbol is not AtomicSymbol.H
    ]
    if not candidates:
        return None

    atom_id = rng.choice(sorted(candidates))
    atom = molecule.atoms[atom_id]
    symbols = [symbol for symbol in MUTATION_SYMBOLS if symbol is not atom.attributes.symbol]
    if not symbols:
        return None
    new_symbol = rng.choice(symbols)
    mutable = MutableMolecule.from_molecule(molecule)
    mutable.atoms[atom_id] = replace(
        atom,
        attributes=element_attributes(new_symbol),
        shells=element_shells(new_symbol),
    )
    return _try_valid_candidate(mutable.freeze())


def remove_terminal_atom(molecule: Molecule, rng: random.Random) -> Molecule | None:
    candidates = _removable_terminal_atoms(molecule)
    if not candidates:
        return None
    return _remove_atom_if_terminal(molecule, rng.choice(candidates))


def add_pi_ring_system(molecule: Molecule, rng: random.Random) -> Molecule | None:
    rings = [
        ring
        for ring in _detect_carbon_six_rings(molecule)
        if not _has_pi_system(molecule, ring)
    ]
    if not rings:
        return None

    ring = rng.choice(rings)
    next_system_id = SystemId(max((system_id.value for system_id, _ in molecule.systems), default=0) + 1)
    mutable = MutableMolecule.from_molecule(molecule)
    mutable.systems.append((next_system_id, mk_bonding_system(NonNegative(6), ring, "pi_ring")))
    return _try_valid_candidate(mutable.freeze())


def print_report(result: SearchResult, *, stream: TextIO) -> None:
    if result.used_default_target:
        print(f"No --target supplied; using median experimental FreeSolv target: {result.target:.3f}", file=stream)
    print(f"Target FreeSolv hydration free energy: {result.target:.3f}", file=stream)
    print("", file=stream)
    print("Diagnostics", file=stream)
    if result.model_parameter_source is not None:
        print(f"  FreeSolv model parameters: {result.model_parameter_source.relative_to(PROJECT_ROOT)}", file=stream)
    print(f"  seed molecule: {result.seed_molecule}", file=stream)
    print(f"  deterministic seed: {RANDOM_SEED}", file=stream)
    print(f"  total proposals: {result.diagnostics.total_proposals}", file=stream)
    print(f"  valid proposals: {result.diagnostics.valid_proposals}", file=stream)
    print(f"  invalid proposals: {result.diagnostics.invalid_proposals}", file=stream)
    print(f"  accepted proposals: {result.diagnostics.accepted_proposals}", file=stream)
    print(f"  acceptance rate: {_safe_rate(result.diagnostics.accepted_proposals, result.diagnostics.total_proposals):.3f}", file=stream)
    print(f"  invalid proposal rate: {_safe_rate(result.diagnostics.invalid_proposals, result.diagnostics.total_proposals):.3f}", file=stream)
    print(f"  unique valid molecules seen: {result.diagnostics.unique_valid_molecules_seen}", file=stream)
    if result.molecule_file_paths:
        print("  molecule files:", file=stream)
        for path in result.molecule_file_paths:
            print(f"    {path.relative_to(PROJECT_ROOT)}", file=stream)
    if result.dietz_file_paths:
        print("  Dietz molecule files:", file=stream)
        for path in result.dietz_file_paths:
            print(f"    {path.relative_to(PROJECT_ROOT)}", file=stream)
    print("", file=stream)
    print("Top generated molecules", file=stream)
    for rank, candidate in enumerate(result.top_candidates, start=1):
        print_candidate(rank, candidate, result.target, stream=stream)
    if result.dietz_candidates:
        print("Top Dietz-system molecules", file=stream)
        for rank, candidate in enumerate(result.dietz_candidates, start=1):
            print_candidate(rank, candidate, result.target, stream=stream)


def print_candidate(rank: int, candidate: Candidate, target: float, *, stream: TextIO) -> None:
    molecule = candidate.molecule
    print(f"Molecule #{rank}", file=stream)
    print(f"  predicted FreeSolv: {candidate.predicted_mean:.3f}", file=stream)
    print(f"  predictive sd: {candidate.predictive_sd:.3f}", file=stream)
    print(f"  target error: {abs(candidate.predicted_mean - target):.3f}", file=stream)
    print(f"  score: {candidate.score:.3f}", file=stream)
    print(f"  atoms: {len(molecule.atoms)}", file=stream)
    print(f"  heavy atoms: {heavy_atom_count(molecule)}", file=stream)
    print(f"  local bonds: {len(molecule.local_bonds)}", file=stream)
    print(f"  Dietz bonding systems: {len(molecule.systems)}", file=stream)
    print(f"  formula: {molecular_formula(molecule)}", file=stream)
    print(format_dietz_molecule(molecule), file=stream)


def format_dietz_molecule(molecule: Molecule) -> str:
    lines = ["  atoms:"]
    for atom_id in sorted(molecule.atoms):
        atom = molecule.atoms[atom_id]
        lines.append(f"    {atom_id.value} {atom.attributes.symbol.value}")

    lines.append("  local_bonds:")
    if molecule.local_bonds:
        for edge in sorted(molecule.local_bonds):
            lines.append(f"    {{{edge.a.value},{edge.b.value}}}")
    else:
        lines.append("    (none)")

    lines.append("  bonding_systems:")
    if molecule.systems:
        for system_index, (system_id, system) in enumerate(molecule.systems, start=1):
            edges = ",".join(f"{{{edge.a.value},{edge.b.value}}}" for edge in sorted(system.member_edges))
            tag = f", tag={system.tag}" if system.tag else ""
            lines.append(
                "    "
                f"System {system_index} (id={system_id.value}): "
                f"shared_electrons={system.shared_electrons.value}, "
                f"member_edges={{{edges}}}{tag}"
            )
    else:
        lines.append("    (none)")
    return "\n".join(lines)


def molecular_formula(molecule: Molecule) -> str:
    counts = Counter(atom.attributes.symbol.value for atom in molecule.atoms.values())
    ordered: list[str] = []
    if "C" in counts:
        ordered.append("C")
    if "H" in counts:
        ordered.append("H")
    ordered.extend(sorted(symbol for symbol in counts if symbol not in {"C", "H"}))
    return "".join(f"{symbol}{counts[symbol] if counts[symbol] > 1 else ''}" for symbol in ordered)


def heavy_atom_count(molecule: Molecule) -> int:
    return sum(1 for atom in molecule.atoms.values() if atom.attributes.symbol is not AtomicSymbol.H)


def write_result_molecule_files(result: SearchResult, output_dir: Path) -> SearchResult:
    target_dir = ensure_directory(output_dir)
    written_paths = tuple(
        _write_candidate_python_file(target_dir, "top", rank, candidate, result.target, result.seed_molecule)
        for rank, candidate in enumerate(result.top_candidates[:TOP_K], start=1)
    )
    dietz_paths = tuple(
        _write_candidate_python_file(target_dir, "dietz", rank, candidate, result.target, result.seed_molecule)
        for rank, candidate in enumerate(result.dietz_candidates[:TOP_DIETZ_K], start=1)
    )
    return SearchResult(
        target=result.target,
        used_default_target=result.used_default_target,
        seed_molecule=result.seed_molecule,
        top_candidates=result.top_candidates,
        dietz_candidates=result.dietz_candidates,
        diagnostics=result.diagnostics,
        molecule_file_paths=written_paths,
        dietz_file_paths=dietz_paths,
        model_parameter_source=result.model_parameter_source,
    )


def _score_molecule(predictor: FreeSolvPredictor, molecule: Molecule, target: float) -> Candidate:
    molecule = _validate_candidate(molecule)
    prediction = predictor.predict(molecule)
    score = -abs(prediction.mean - target)
    heavy_atoms = heavy_atom_count(molecule)
    if heavy_atoms > MAX_HEAVY_ATOMS:
        score -= 0.1 * float(heavy_atoms - MAX_HEAVY_ATOMS)
    return Candidate(
        molecule=molecule,
        predicted_mean=prediction.mean,
        predictive_sd=prediction.sd,
        score=score,
    )


def _load_gp_parameter_means() -> FreeSolvModelParameters:
    coefficients_path = _find_model_dir() / "details" / "model_coefficients.csv"
    coefficients = pd.read_csv(coefficients_path)
    rows = coefficients.loc[
        (coefficients["dataset"] == "freesolv")
        & (coefficients["representation"] == "moladt_featurized")
        & (coefficients["target"] == TARGET_NAME)
        & (coefficients["model"] == MODEL_NAME)
        & (coefficients["method"] == METHOD_NAME)
    ]
    if len(rows) != 4:
        raise RuntimeError(
            "Expected exactly four FreeSolv GP parameter rows "
            f"for freesolv/moladt_featurized/{TARGET_NAME}/{MODEL_NAME}/{METHOD_NAME}; "
            f"found {len(rows)} in {coefficients_path}"
        )
    parameter_means = {
        str(row["parameter_name"]): float(row["posterior_mean"])
        for _, row in rows.iterrows()
    }
    required = ("alpha", "signal_scale", "lengthscale", "sigma")
    missing = [name for name in required if name not in parameter_means]
    if missing:
        raise RuntimeError(f"Missing FreeSolv GP parameters: {', '.join(missing)}")
    return FreeSolvModelParameters(
        alpha=parameter_means["alpha"],
        signal_scale=parameter_means["signal_scale"],
        lengthscale=parameter_means["lengthscale"],
        sigma=parameter_means["sigma"],
        source_path=coefficients_path,
    )


def _find_model_dir() -> Path:
    if (DEFAULT_MODEL_DIR / "details" / "model_coefficients.csv").exists():
        return DEFAULT_MODEL_DIR
    raise FileNotFoundError(
        "Missing committed FreeSolv Bayesian GP artifact: "
        f"{DEFAULT_MODEL_DIR / 'details' / 'model_coefficients.csv'}"
    )


def _validate_freesolv_metadata(metadata: pd.Series, metadata_path: Path) -> None:
    expected = {
        "dataset": "freesolv",
        "representation": "moladt_featurized",
        "target_name": TARGET_NAME,
    }
    mismatches = [
        f"{key}={metadata.get(key)!r}, expected {expected_value!r}"
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(f"FreeSolv metadata mismatch in {metadata_path}: {'; '.join(mismatches)}")


def _screen_top_correlation_features(X_train: np.ndarray, y_train: np.ndarray, *, top_k: int) -> tuple[int, ...]:
    feature_count = min(int(top_k), X_train.shape[1])
    if feature_count <= 0:
        raise ValueError("GP feature screening requires at least one feature")
    y_centered = y_train - np.mean(y_train)
    scores: list[tuple[float, int]] = []
    for feature_index in range(X_train.shape[1]):
        column = X_train[:, feature_index]
        denominator = float(np.linalg.norm(column) * np.linalg.norm(y_centered))
        score = 0.0 if denominator == 0.0 else abs(float(np.dot(column, y_centered) / denominator))
        scores.append((score, feature_index))
    return tuple(sorted(index for _, index in sorted(scores, reverse=True)[:feature_count]))


def _rbf_kernel(X_left: np.ndarray, X_right: np.ndarray, *, lengthscale: float, signal_scale: float) -> np.ndarray:
    left_sq = np.sum(np.square(X_left), axis=1)[:, np.newaxis]
    right_sq = np.sum(np.square(X_right), axis=1)[np.newaxis, :]
    sqdist = np.maximum(left_sq + right_sq - 2.0 * (X_left @ X_right.T), 0.0)
    return (signal_scale**2) * np.exp(-0.5 * sqdist / max(lengthscale**2, 1e-9))


def _candidate_sort_key(target: float) -> Callable[[Candidate], tuple[float, int, float, int]]:
    def sort_key(candidate: Candidate) -> tuple[float, int, float, int]:
        return (
            candidate.score,
            1 if candidate.molecule.systems else 0,
            -abs(candidate.predicted_mean - target),
            -len(candidate.molecule.atoms),
        )

    return sort_key


def _write_candidate_python_file(
    output_dir: Path,
    prefix: str,
    rank: int,
    candidate: Candidate,
    target: float,
    seed_molecule: str,
) -> Path:
    path = output_dir / f"{prefix}_{rank:02d}_molecule.py"
    path.write_text(_candidate_python_source(rank, candidate, target, seed_molecule), encoding="utf-8")
    return path


def _resolve_output_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _candidate_python_source(rank: int, candidate: Candidate, target: float, seed_molecule: str) -> str:
    molecule = candidate.molecule
    return "\n".join(
        (
            "from __future__ import annotations",
            "",
            "from moladt.chem.constants import element_attributes, element_shells",
            "from moladt.chem.coordinate import Coordinate, mk_angstrom",
            "from moladt.chem.dietz import AtomId, NonNegative, SystemId, mk_bonding_system, mk_edge",
            "from moladt.chem.molecule import Atom, AtomicSymbol, Molecule",
            "from moladt.chem.validate import validate_molecule",
            "",
            f"rank = {rank}",
            f"target_freesolv = {target:.12g}",
            f"seed_molecule = {seed_molecule!r}",
            f"random_seed = {RANDOM_SEED}",
            f"predicted_freesolv = {candidate.predicted_mean:.12g}",
            f"predictive_sd = {candidate.predictive_sd:.12g}",
            f"target_error = {abs(candidate.predicted_mean - target):.12g}",
            f"score = {candidate.score:.12g}",
            f"formula = {molecular_formula(molecule)!r}",
            "",
            "atoms = {",
            *_atom_literal_lines(molecule),
            "}",
            "",
            "local_bonds = frozenset({",
            *_edge_literal_lines(molecule.local_bonds),
            "})",
            "",
            "systems = (",
            *_system_literal_lines(molecule),
            ")",
            "",
            "molecule = validate_molecule(",
            "    Molecule(",
            "        atoms=atoms,",
            "        local_bonds=local_bonds,",
            "        systems=systems,",
            "    )",
            ")",
            "",
            "__all__ = [",
            "    'rank',",
            "    'target_freesolv',",
            "    'seed_molecule',",
            "    'random_seed',",
            "    'predicted_freesolv',",
            "    'predictive_sd',",
            "    'target_error',",
            "    'score',",
            "    'formula',",
            "    'molecule',",
            "]",
            "",
        )
    )


def _atom_literal_lines(molecule: Molecule) -> tuple[str, ...]:
    lines: list[str] = []
    for atom_id in sorted(molecule.atoms):
        atom = molecule.atoms[atom_id]
        symbol = atom.attributes.symbol.value
        lines.extend(
            (
                f"    AtomId({atom_id.value}): Atom(",
                f"        atom_id=AtomId({atom_id.value}),",
                f"        attributes=element_attributes(AtomicSymbol.{symbol}),",
                "        coordinate=Coordinate(",
                f"            mk_angstrom({atom.coordinate.x.value:.12g}),",
                f"            mk_angstrom({atom.coordinate.y.value:.12g}),",
                f"            mk_angstrom({atom.coordinate.z.value:.12g}),",
                "        ),",
                f"        shells=element_shells(AtomicSymbol.{symbol}),",
                f"        formal_charge={atom.formal_charge},",
                "    ),",
            )
        )
    return tuple(lines)


def _edge_literal_lines(edges: frozenset[Edge] | set[Edge]) -> tuple[str, ...]:
    return tuple(f"    mk_edge(AtomId({edge.a.value}), AtomId({edge.b.value}))," for edge in sorted(edges))


def _system_literal_lines(molecule: Molecule) -> tuple[str, ...]:
    lines: list[str] = []
    for system_id, system in molecule.systems:
        lines.append("    (")
        lines.append(f"        SystemId({system_id.value}),")
        lines.append("        mk_bonding_system(")
        lines.append(f"            NonNegative({system.shared_electrons.value}),")
        lines.append("            frozenset({")
        lines.extend(f"                mk_edge(AtomId({edge.a.value}), AtomId({edge.b.value}))," for edge in sorted(system.member_edges))
        lines.append("            }),")
        lines.append(f"            {system.tag!r},")
        lines.append("        ),")
        lines.append("    ),")
    return tuple(lines)


def _new_atom(atom_id: AtomId, symbol: AtomicSymbol, parent_atom: Atom) -> Atom:
    angle = (atom_id.value % 6) * math.pi / 3.0
    radius = 1.2 + 0.05 * float(atom_id.value % 5)
    return Atom(
        atom_id=atom_id,
        attributes=element_attributes(symbol),
        coordinate=Coordinate(
            mk_angstrom(parent_atom.coordinate.x.value + radius * math.cos(angle)),
            mk_angstrom(parent_atom.coordinate.y.value + radius * math.sin(angle)),
            mk_angstrom(parent_atom.coordinate.z.value + 0.1 * float(atom_id.value % 3)),
        ),
        shells=element_shells(symbol),
        formal_charge=0,
    )


def _seed_atom(atom_id: int, symbol: AtomicSymbol, x: float, y: float, z: float) -> Atom:
    resolved_atom_id = AtomId(atom_id)
    return Atom(
        atom_id=resolved_atom_id,
        attributes=element_attributes(symbol),
        coordinate=Coordinate(mk_angstrom(x), mk_angstrom(y), mk_angstrom(z)),
        shells=element_shells(symbol),
        formal_charge=0,
    )


def _seed_molecule(
    atoms: Sequence[Atom],
    bonds: Sequence[tuple[int, int]],
    systems: Sequence[tuple[SystemId, BondingSystem]] = (),
) -> Molecule:
    atom_map = {atom.atom_id: atom for atom in atoms}
    return Molecule(
        atoms=atom_map,
        local_bonds=frozenset(mk_edge(AtomId(left), AtomId(right)) for left, right in bonds),
        systems=tuple(systems),
    )


def _ethane_seed() -> Molecule:
    return _seed_molecule(
        (
            _seed_atom(1, AtomicSymbol.C, -0.77, 0.00, 0.00),
            _seed_atom(2, AtomicSymbol.C, 0.77, 0.00, 0.00),
            _seed_atom(3, AtomicSymbol.H, -1.17, 0.99, 0.00),
            _seed_atom(4, AtomicSymbol.H, -1.17, -0.49, 0.86),
            _seed_atom(5, AtomicSymbol.H, -1.17, -0.49, -0.86),
            _seed_atom(6, AtomicSymbol.H, 1.17, -0.99, 0.00),
            _seed_atom(7, AtomicSymbol.H, 1.17, 0.49, 0.86),
            _seed_atom(8, AtomicSymbol.H, 1.17, 0.49, -0.86),
        ),
        ((1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (2, 8)),
    )


def _methanol_seed() -> Molecule:
    return _seed_molecule(
        (
            _seed_atom(1, AtomicSymbol.C, 0.00, 0.00, 0.00),
            _seed_atom(2, AtomicSymbol.O, 1.43, 0.00, 0.00),
            _seed_atom(3, AtomicSymbol.H, -0.36, 1.02, 0.00),
            _seed_atom(4, AtomicSymbol.H, -0.36, -0.51, 0.88),
            _seed_atom(5, AtomicSymbol.H, -0.36, -0.51, -0.88),
            _seed_atom(6, AtomicSymbol.H, 1.86, 0.86, 0.00),
        ),
        ((1, 2), (1, 3), (1, 4), (1, 5), (2, 6)),
    )


def _carbon_six_ring_seed() -> Molecule:
    carbon_atoms = tuple(
        _seed_atom(
            index + 1,
            AtomicSymbol.C,
            1.40 * math.cos(index * math.pi / 3.0),
            1.40 * math.sin(index * math.pi / 3.0),
            0.00,
        )
        for index in range(6)
    )
    hydrogen_atoms = tuple(
        _seed_atom(
            index + 7,
            AtomicSymbol.H,
            2.40 * math.cos(index * math.pi / 3.0),
            2.40 * math.sin(index * math.pi / 3.0),
            0.00,
        )
        for index in range(6)
    )
    ring_bonds = tuple((index + 1, ((index + 1) % 6) + 1) for index in range(6))
    hydrogen_bonds = tuple((index + 1, index + 7) for index in range(6))
    return _seed_molecule((*carbon_atoms, *hydrogen_atoms), (*ring_bonds, *hydrogen_bonds))


def _pi_carbon_six_ring_seed() -> Molecule:
    molecule = add_pi_ring_system(_carbon_six_ring_seed(), random.Random(0))
    if molecule is None:
        raise RuntimeError("Internal carbon six-ring seed could not accept a pi_ring bonding system")
    return molecule


def _available_valence(molecule: Molecule, atom_id: AtomId) -> float:
    atom = molecule.atoms[atom_id]
    return _growth_max_valence(atom.attributes.symbol) - used_electrons_at(molecule, atom_id)


def _growth_max_valence(symbol: AtomicSymbol) -> float:
    return GROWTH_MAX_VALENCE[symbol]


def _terminal_hydrogens_attached_to(molecule: Molecule, atom_id: AtomId) -> tuple[AtomId, ...]:
    hydrogens: list[AtomId] = []
    system_atoms = {member for _, system in molecule.systems for member in system.member_atoms}
    for neighbor in neighbors_sigma(molecule, atom_id):
        atom = molecule.atoms[neighbor]
        if atom.attributes.symbol is not AtomicSymbol.H:
            continue
        if neighbor in system_atoms:
            continue
        if len(neighbors_sigma(molecule, neighbor)) == 1:
            hydrogens.append(neighbor)
    return tuple(sorted(hydrogens))


def _free_one_valence_slot(mutable: MutableMolecule, atom_id: AtomId) -> None:
    molecule = mutable.freeze()
    if _available_valence(molecule, atom_id) >= 1.0 - 1e-9:
        return
    hydrogens = _terminal_hydrogens_attached_to(molecule, atom_id)
    if not hydrogens:
        return
    _remove_atom_from_mutable(mutable, hydrogens[0])


def _has_localized_singleton_system(molecule: Molecule, edge: Edge) -> bool:
    return any(
        len(system.member_edges) == 1
        and system.shared_electrons.value == 2
        and edge in system.member_edges
        for _, system in molecule.systems
    )


def _removable_terminal_atoms(molecule: Molecule) -> tuple[AtomId, ...]:
    if len(molecule.atoms) <= 1:
        return ()
    protected_atoms = {
        atom_id
        for _, system in molecule.systems
        if len(system.member_edges) > 1
        for atom_id in system.member_atoms
    }
    candidates = [
        atom_id
        for atom_id in molecule.atoms
        if atom_id not in protected_atoms
        and molecule.atoms[atom_id].attributes.symbol is not AtomicSymbol.H
        and len(neighbors_sigma(molecule, atom_id)) == 1
    ]
    return tuple(sorted(candidates))


def _remove_atom_if_terminal(molecule: Molecule, atom_id: AtomId) -> Molecule | None:
    if atom_id not in _removable_terminal_atoms(molecule):
        return None
    mutable = MutableMolecule.from_molecule(molecule)
    _remove_atom_from_mutable(mutable, atom_id)
    return _try_valid_candidate(mutable.freeze())


def _remove_atom_from_mutable(mutable: MutableMolecule, atom_id: AtomId) -> None:
    incident_edges = {
        edge
        for edge in mutable.local_bonds
        if edge.a == atom_id or edge.b == atom_id
    }
    mutable.atoms.pop(atom_id, None)
    mutable.local_bonds.difference_update(incident_edges)
    mutable.systems = [
        (system_id, system)
        for system_id, system in mutable.systems
        if atom_id not in system.member_atoms
        and not any(edge in incident_edges for edge in system.member_edges)
    ]


def _detect_carbon_six_rings(molecule: Molecule) -> tuple[frozenset[Edge], ...]:
    adjacency: dict[AtomId, list[AtomId]] = {}
    for edge in molecule.local_bonds:
        adjacency.setdefault(edge.a, []).append(edge.b)
        adjacency.setdefault(edge.b, []).append(edge.a)
    for atom_id in adjacency:
        adjacency[atom_id].sort()

    discovered: set[frozenset[Edge]] = set()

    def search(path: list[AtomId], current: AtomId) -> None:
        if len(path) == 6:
            if path[0] in adjacency.get(current, []):
                ring_atoms = frozenset(path)
                if all(molecule.atoms[atom_id].attributes.symbol is AtomicSymbol.C for atom_id in ring_atoms):
                    ring = frozenset(
                        mk_edge(path[index], path[index + 1] if index < 5 else path[0])
                        for index in range(6)
                    )
                    if path[0] == min(path):
                        discovered.add(ring)
            return
        for neighbor in adjacency.get(current, []):
            if neighbor in path:
                continue
            search([*path, neighbor], neighbor)

    for start in sorted(adjacency):
        search([start], start)
    return tuple(sorted(discovered, key=_ring_sort_key))


def _has_pi_system(molecule: Molecule, ring: frozenset[Edge]) -> bool:
    return any(
        system.member_edges == ring
        and system.shared_electrons.value == 6
        and len(system.member_edges) == 6
        for _, system in molecule.systems
    )


def _is_valid_pi_ring(molecule: Molecule, system: BondingSystem) -> bool:
    if system.shared_electrons.value != 6 or len(system.member_edges) != 6:
        return False
    ring_atoms = {atom_id for edge in system.member_edges for atom_id in (edge.a, edge.b)}
    if len(ring_atoms) != 6:
        return False
    if any(molecule.atoms[atom_id].attributes.symbol is not AtomicSymbol.C for atom_id in ring_atoms):
        return False
    if any(edge not in molecule.local_bonds for edge in system.member_edges):
        return False
    ring_degree = {atom_id: 0 for atom_id in ring_atoms}
    for edge in system.member_edges:
        ring_degree[edge.a] += 1
        ring_degree[edge.b] += 1
    return all(degree == 2 for degree in ring_degree.values())


def _shortest_sigma_path_length(molecule: Molecule, start: AtomId, goal: AtomId) -> int | None:
    frontier: list[tuple[AtomId, int]] = [(start, 0)]
    seen = {start}
    while frontier:
        current, distance = frontier.pop(0)
        if current == goal:
            return distance
        for neighbor in neighbors_sigma(molecule, current):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            frontier.append((neighbor, distance + 1))
    return None


def _validate_candidate(molecule: Molecule) -> Molecule:
    validate_molecule(molecule)
    if not _is_connected(molecule):
        raise ValidationError("Molecule is disconnected")
    _ensure_supported_symbols(molecule)
    _ensure_no_hydrogen_hydrogen_local_bonds(molecule)
    _ensure_conservative_generator_valence(molecule)
    _ensure_sound_bonding_systems(molecule)
    for edge in molecule.local_bonds:
        effective_order(molecule, edge)
    return molecule


def _try_valid_candidate(molecule: Molecule) -> Molecule | None:
    try:
        return _validate_candidate(_canonicalize_atom_ids(_complete_terminal_hydrogens(molecule)))
    except (ValueError, ValidationError):
        return None


def _ensure_supported_symbols(molecule: Molecule) -> None:
    unsupported = [
        atom.attributes.symbol.value
        for atom in molecule.atoms.values()
        if atom.attributes.symbol not in GROWTH_MAX_VALENCE
    ]
    if unsupported:
        raise ValidationError(f"Unsupported generated elements: {', '.join(sorted(set(unsupported)))}")


def _ensure_no_hydrogen_hydrogen_local_bonds(molecule: Molecule) -> None:
    for edge in molecule.local_bonds:
        if (
            molecule.atoms[edge.a].attributes.symbol is AtomicSymbol.H
            and molecule.atoms[edge.b].attributes.symbol is AtomicSymbol.H
        ):
            raise ValidationError("Generated molecules may not contain H-H local bonds")


def _ensure_conservative_generator_valence(molecule: Molecule) -> None:
    for atom_id, atom in molecule.atoms.items():
        used = used_electrons_at(molecule, atom_id)
        if used > _growth_max_valence(atom.attributes.symbol) + 1e-9:
            raise ValidationError(f"Atom {atom_id.value} exceeds generator valence cap")


def _ensure_sound_bonding_systems(molecule: Molecule) -> None:
    seen: set[tuple[int, tuple[tuple[int, int], ...], str | None]] = set()
    for _, system in molecule.systems:
        signature = (
            system.shared_electrons.value,
            tuple(sorted((edge.a.value, edge.b.value) for edge in system.member_edges)),
            system.tag,
        )
        if signature in seen:
            raise ValidationError("Duplicate Dietz bonding system")
        seen.add(signature)
        if system.tag == "pi_ring" and not _is_valid_pi_ring(molecule, system):
            raise ValidationError("pi_ring bonding system is not a simple carbon six-ring")


def _complete_terminal_hydrogens(molecule: Molecule) -> Molecule:
    mutable = MutableMolecule.from_molecule(molecule)
    system_atoms = {atom_id for _, system in molecule.systems for atom_id in system.member_atoms}
    removable_hydrogens = [
        atom_id
        for atom_id, atom in molecule.atoms.items()
        if atom.attributes.symbol is AtomicSymbol.H
        and atom.formal_charge == 0
        and atom_id not in system_atoms
        and len(neighbors_sigma(molecule, atom_id)) == 1
    ]
    for atom_id in removable_hydrogens:
        _remove_atom_from_mutable(mutable, atom_id)

    next_atom_id = max((atom_id.value for atom_id in mutable.atoms), default=0) + 1
    for atom_id in sorted(tuple(mutable.atoms)):
        atom = mutable.atoms[atom_id]
        if atom.attributes.symbol is AtomicSymbol.H:
            continue
        current = mutable.freeze()
        available = _growth_max_valence(atom.attributes.symbol) - used_electrons_at(current, atom_id)
        hydrogen_count = max(0, int(math.floor(available + 1e-9)))
        for _ in range(hydrogen_count):
            hydrogen_id = AtomId(next_atom_id)
            next_atom_id += 1
            mutable.atoms[hydrogen_id] = _new_atom(hydrogen_id, AtomicSymbol.H, atom)
            mutable.local_bonds.add(mk_edge(atom_id, hydrogen_id))

    return mutable.freeze()


def _canonicalize_atom_ids(molecule: Molecule) -> Molecule:
    ordered_ids = tuple(sorted(molecule.atoms))
    canonical_ids = tuple(AtomId(index) for index in range(1, len(ordered_ids) + 1))
    if ordered_ids == canonical_ids:
        return molecule

    atom_id_map = {old_id: new_id for old_id, new_id in zip(ordered_ids, canonical_ids, strict=True)}
    atoms = {
        atom_id_map[old_id]: replace(atom, atom_id=atom_id_map[old_id])
        for old_id, atom in molecule.atoms.items()
    }
    local_bonds = frozenset(
        mk_edge(atom_id_map[edge.a], atom_id_map[edge.b])
        for edge in molecule.local_bonds
    )
    systems = tuple(
        (
            SystemId(index),
            mk_bonding_system(
                system.shared_electrons,
                frozenset(mk_edge(atom_id_map[edge.a], atom_id_map[edge.b]) for edge in system.member_edges),
                system.tag,
            ),
        )
        for index, (_, system) in enumerate(molecule.systems, start=1)
    )
    return Molecule(atoms=atoms, local_bonds=local_bonds, systems=systems)


def _is_connected(molecule: Molecule) -> bool:
    atom_ids = set(molecule.atoms)
    if len(atom_ids) <= 1:
        return True
    adjacency = {atom_id: set() for atom_id in atom_ids}
    for edge in _all_edges(molecule):
        if edge.a in adjacency and edge.b in adjacency:
            adjacency[edge.a].add(edge.b)
            adjacency[edge.b].add(edge.a)
    start = next(iter(atom_ids))
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == atom_ids


def _all_edges(molecule: Molecule) -> set[Edge]:
    edges = set(molecule.local_bonds)
    for _, system in molecule.systems:
        edges.update(system.member_edges)
    return edges


def _weighted_choice(weighted_names: tuple[tuple[str, float], ...], rng: random.Random) -> str:
    total_weight = sum(weight for _, weight in weighted_names)
    threshold = rng.random() * total_weight
    cumulative = 0.0
    for name, weight in weighted_names:
        cumulative += weight
        if threshold <= cumulative:
            return name
    return weighted_names[-1][0]


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _molecule_key(molecule: Molecule) -> bytes:
    return molecule_to_json_bytes(molecule)


def _ring_sort_key(ring: frozenset[Edge]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((edge.a.value, edge.b.value) for edge in ring))


_MOVE_FUNCTIONS: dict[str, Callable[[Molecule, random.Random], Molecule | None]] = {
    "add_terminal_atom": add_terminal_atom,
    "add_sigma_edge": add_sigma_edge,
    "mutate_atom": mutate_atom,
    "remove_terminal_atom": remove_terminal_atom,
    "add_pi_ring_system": add_pi_ring_system,
}


if __name__ == "__main__":
    raise SystemExit(main())
