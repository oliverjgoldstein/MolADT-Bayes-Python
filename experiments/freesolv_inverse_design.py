from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, replace
import json
import math
import os
import random
import sys
import time
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
from moladt.io import molecule_from_dict, molecule_to_json_bytes
from moladt.viewer import open_molecule_viewer, write_molecule_viewer_collection_html
from scripts.common import PROCESSED_DATA_DIR, PROJECT_ROOT, configured_results_dir, ensure_directory
from scripts.features import compute_moladt_featurized_descriptors
from scripts.stan_runner import GP_SCREENED_FEATURE_COUNT


N_STEPS = 2000
N_SEEDS = 5
MIN_UNIQUE_VALID_MOLECULES = 1_000
MAX_PROPOSALS_PER_REQUIRED_MOLECULE = 100
TOP_K = 10
TOP_DIETZ_K = 5
MAX_HEAVY_ATOMS = 12
TEMPERATURE = 1.0
RANDOM_SEED = 0
MAX_TOTAL_ATOMS = 4 * MAX_HEAVY_ATOMS + 8
HEAVY_ATOM_GROWTH_LIMIT = MAX_HEAVY_ATOMS + 2
DEFAULT_SEED_MOLECULE = "water"
REFERENCE_RESULTS_ENV = "MOLADT_REFERENCE_RESULTS_DIR"
CREDIBLE_SCORE_NOISE_FLOOR = 1.0

DATASET_PREFIX = "freesolv_moladt_featurized"
FREESOLV_RESULTS_DIR = PROJECT_ROOT / "results" / "freesolv"
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
VALENCE_TOLERANCE = 1e-9
TERMINAL_FREE_SOLV_SYMBOLS = (AtomicSymbol.H, AtomicSymbol.F, AtomicSymbol.Cl)


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
    bayesian_credible_score_percent: float
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
    generated_candidates: tuple[Candidate, ...] = ()
    minimum_unique_valid_molecules: int = 0
    dietz_candidates: tuple[Candidate, ...] = ()
    molecule_file_paths: tuple[Path, ...] = ()
    dietz_file_paths: tuple[Path, ...] = ()
    generated_candidate_file_paths: tuple[Path, ...] = ()
    viewer_file_paths: tuple[Path, ...] = ()
    model_parameter_source: Path | None = None
    model_draw_source: Path | None = None


@dataclass(frozen=True, slots=True)
class FreeSolvModelParameters:
    alpha: float
    signal_scale: float
    lengthscale: float
    sigma: float
    source_path: Path


@dataclass(frozen=True, slots=True)
class FreeSolvPosteriorDraws:
    alpha: np.ndarray
    signal_scale: np.ndarray
    lengthscale: np.ndarray
    sigma: np.ndarray
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
    alpha_draws: np.ndarray
    signal_scale_draws: np.ndarray
    lengthscale_draws: np.ndarray
    sigma_draws: np.ndarray
    draw_weights: np.ndarray
    chol: np.ndarray
    parameter_source_path: Path
    draw_source_path: Path

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
        model_dir = _find_model_dir()
        parameters = _load_gp_parameter_means(model_dir)
        draws = _load_gp_posterior_draws(model_dir)
        draw_weights = _precompute_gp_draw_weights(
            X_train=X_train,
            y_train=y_train,
            alpha=draws.alpha,
            signal_scale=draws.signal_scale,
            lengthscale=draws.lengthscale,
            sigma=draws.sigma,
        )

        train_kernel = _rbf_kernel(X_train, X_train, lengthscale=parameters.lengthscale, signal_scale=parameters.signal_scale)
        train_kernel[np.diag_indices_from(train_kernel)] += parameters.sigma**2 + 1e-8
        chol = np.linalg.cholesky(train_kernel)
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
            alpha_draws=draws.alpha,
            signal_scale_draws=draws.signal_scale,
            lengthscale_draws=draws.lengthscale,
            sigma_draws=draws.sigma,
            draw_weights=draw_weights,
            chol=chol,
            parameter_source_path=parameters.source_path,
            draw_source_path=draws.source_path,
        )

    def predict(self, molecule: Molecule) -> Prediction:
        descriptors = compute_moladt_featurized_descriptors(molecule)
        raw = np.asarray([float(descriptors.get(name, 0.0)) for name in self.feature_names], dtype=float)
        standardized = (raw - self.train_mean) / self.train_std
        x_eval = standardized[list(self.selected_indices)].reshape(1, -1)
        mean_by_draw = _gp_mean_by_draw(
            x_eval=x_eval,
            X_train=self.X_train,
            alpha=self.alpha_draws,
            signal_scale=self.signal_scale_draws,
            lengthscale=self.lengthscale_draws,
            draw_weights=self.draw_weights,
        )
        predictive_mean = float(np.mean(mean_by_draw))

        cross_kernel = _rbf_kernel(x_eval, self.X_train, lengthscale=self.lengthscale, signal_scale=self.signal_scale)
        solve = np.linalg.solve(self.chol, cross_kernel.T)
        conditional_var = self.signal_scale**2 + self.sigma**2 - float(np.sum(np.square(solve), axis=0)[0])
        predictive_var = max(conditional_var, 1e-9) + float(np.var(mean_by_draw))
        return Prediction(mean=predictive_mean, sd=math.sqrt(max(predictive_var, 1e-9)))


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
    parser.add_argument(
        "--open-viewer",
        action="store_true",
        help="Write viewer HTML for top generated molecules and open it in the default browser.",
    )
    parser.add_argument(
        "--viewer-count",
        type=int,
        default=TOP_K,
        help=f"Number of top generated molecule viewers to write/open when --open-viewer is set. Defaults to {TOP_K}.",
    )
    parser.add_argument(
        "--view-results",
        type=Path,
        default=None,
        help="Write a viewer for an existing inverse-design result directory instead of running a search.",
    )
    parser.add_argument(
        "--viewer-output",
        type=Path,
        default=None,
        help="Optional output path for --view-results. Defaults to <result-dir>/top_molecules.viewer.html.",
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
    output_stream = stream or sys.stdout
    if args.view_results is not None:
        viewer_path = write_saved_inverse_design_viewer_file(
            args.view_results,
            output_path=args.viewer_output,
            count=args.viewer_count,
        )
        print(f"Viewer: {viewer_path}", file=output_stream)
        if args.open_viewer:
            open_result_viewers((viewer_path,), stream=output_stream)
        return 0

    result = run_inverse_design(
        target=args.target,
        n_steps=n_steps,
        n_seeds=n_seeds,
        top_k=top_k,
        predictor=predictor,
        seed_molecule=args.seed_molecule,
        min_unique_valid_molecules=0 if predictor is not None else MIN_UNIQUE_VALID_MOLECULES,
        progress_stream=output_stream,
    )
    if predictor is None:
        results_dir = configured_results_dir()
        result = write_result_molecule_files(result, results_dir)
        if args.open_viewer:
            result = write_result_viewer_files(result, results_dir, count=args.viewer_count)
            open_result_viewers(result.viewer_file_paths, stream=output_stream)
        reference_results_dir = os.environ.get(REFERENCE_RESULTS_ENV)
        if reference_results_dir:
            write_result_molecule_files(result, _resolve_output_dir(reference_results_dir))
    print_report(result, stream=output_stream)
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
    min_unique_valid_molecules: int = 0,
    max_total_proposals: int | None = None,
    progress_stream: TextIO | None = None,
) -> SearchResult:
    used_default_target = target is None
    resolved_target = default_target_from_freesolv_dataset() if target is None else float(target)
    active_predictor = predictor or FreeSolvBayesianPredictor.load()
    model_parameter_source = getattr(active_predictor, "parameter_source_path", None)
    model_draw_source = getattr(active_predictor, "draw_source_path", None)

    candidate_by_key: dict[bytes, Candidate] = {}
    diagnostics = SearchDiagnostics()
    seeds = load_seed_molecules(seed_molecule=seed_molecule, n_seeds=n_seeds)

    chains: list[tuple[random.Random, Molecule, Candidate]] = []
    for seed_index, starting_molecule in enumerate(seeds):
        rng = random.Random(rng_seed + seed_index)
        current = _validate_candidate(starting_molecule)
        current_candidate = _score_molecule(active_predictor, current, resolved_target)
        candidate_by_key[_molecule_key(current)] = current_candidate
        chains.append((rng, current, current_candidate))

    seed_candidate_keys = frozenset(candidate_by_key)
    generated_candidate_keys: set[bytes] = set()
    planned_proposals = max(0, n_steps) * len(chains)
    minimum_unique = max(0, int(min_unique_valid_molecules))
    if minimum_unique and not chains:
        raise ValueError("At least one seed chain is required when requesting generated molecules")
    proposal_limit = _proposal_limit(
        planned_proposals=planned_proposals,
        minimum_unique_valid_molecules=minimum_unique,
        max_total_proposals=max_total_proposals,
    )
    chain_index = 0
    progress_start = time.perf_counter()

    while chains and (
        len(generated_candidate_keys) < minimum_unique
        if minimum_unique
        else diagnostics.total_proposals < planned_proposals
    ):
        if diagnostics.total_proposals >= proposal_limit:
            raise RuntimeError(
                "FreeSolv inverse design could not generate "
                f"{minimum_unique} unique valid molecules within "
                f"{proposal_limit} proposal attempts; generated {len(generated_candidate_keys)}"
            )

        rng, current, current_candidate = chains[chain_index]
        diagnostics.total_proposals += 1
        proposal = propose_molecule(current, rng)
        if proposal is None:
            diagnostics.invalid_proposals += 1
        else:
            try:
                proposal = _validate_candidate(proposal)
            except (ValueError, ValidationError):
                diagnostics.invalid_proposals += 1
            else:
                diagnostics.valid_proposals += 1
                proposal_key = _molecule_key(proposal)
                existing_candidate = candidate_by_key.get(proposal_key)
                if existing_candidate is None:
                    proposal_candidate = _score_molecule(active_predictor, proposal, resolved_target)
                    candidate_by_key[proposal_key] = proposal_candidate
                    if proposal_key not in seed_candidate_keys:
                        generated_candidate_keys.add(proposal_key)
                    _print_generation_progress(
                        progress_stream,
                        generated_count=len(generated_candidate_keys),
                        target_count=minimum_unique,
                        elapsed_seconds=time.perf_counter() - progress_start,
                    )
                else:
                    proposal_candidate = existing_candidate

                score_delta = proposal_candidate.score - current_candidate.score
                if score_delta >= 0.0 or rng.random() < math.exp(score_delta / TEMPERATURE):
                    current = _validate_candidate(proposal)
                    current_candidate = proposal_candidate
                    diagnostics.accepted_proposals += 1

        chains[chain_index] = (rng, current, current_candidate)
        chain_index = (chain_index + 1) % len(chains)

    diagnostics.unique_valid_molecules_seen = len(candidate_by_key)
    all_candidates = tuple(
        sorted(
            candidate_by_key.values(),
            key=_candidate_sort_key(resolved_target),
            reverse=True,
        )
    )
    generated_candidates = tuple(
        candidate
        for candidate in all_candidates
        if _molecule_key(candidate.molecule) in generated_candidate_keys
    )
    ranked_candidates = generated_candidates or all_candidates
    top_candidates = ranked_candidates[:top_k]
    dietz_candidates = tuple(
        candidate
        for candidate in ranked_candidates
        if candidate.molecule.systems
    )[: min(top_k, TOP_DIETZ_K)]
    return SearchResult(
        target=resolved_target,
        used_default_target=used_default_target,
        seed_molecule=seed_molecule,
        top_candidates=top_candidates,
        generated_candidates=generated_candidates,
        dietz_candidates=dietz_candidates,
        diagnostics=diagnostics,
        minimum_unique_valid_molecules=minimum_unique,
        model_parameter_source=model_parameter_source,
        model_draw_source=model_draw_source,
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


def _proposal_limit(
    *,
    planned_proposals: int,
    minimum_unique_valid_molecules: int,
    max_total_proposals: int | None,
) -> int:
    if max_total_proposals is not None:
        if max_total_proposals < 0:
            raise ValueError("max_total_proposals must be non-negative")
        return max_total_proposals
    if minimum_unique_valid_molecules <= 0:
        return planned_proposals
    minimum_budget = minimum_unique_valid_molecules * MAX_PROPOSALS_PER_REQUIRED_MOLECULE
    return max(planned_proposals, minimum_budget)


def _print_generation_progress(
    stream: TextIO | None,
    *,
    generated_count: int,
    target_count: int,
    elapsed_seconds: float,
) -> None:
    if stream is None:
        return
    average_seconds = elapsed_seconds / max(1, generated_count)
    if target_count > 0:
        print(
            "Generated unique valid candidates: "
            f"{generated_count}/{target_count} "
            f"(elapsed {elapsed_seconds:.2f}s, avg {average_seconds:.3f}s/candidate)",
            file=stream,
            flush=True,
        )
    else:
        print(
            "Generated unique valid candidates: "
            f"{generated_count} "
            f"(elapsed {elapsed_seconds:.2f}s, avg {average_seconds:.3f}s/candidate)",
            file=stream,
            flush=True,
        )


def propose_molecule(molecule: Molecule, rng: random.Random) -> Molecule | None:
    move_names = [name for name, _ in MOVE_WEIGHTS]
    move_weights = dict(MOVE_WEIGHTS)
    while move_names:
        available_weights = tuple((name, move_weights[name]) for name in move_names)
        move_name = _weighted_choice(available_weights, rng)
        proposal = _MOVE_FUNCTIONS[move_name](molecule, rng)
        if proposal is not None:
            return proposal
        move_names.remove(move_name)
    return None


def add_terminal_atom(molecule: Molecule, rng: random.Random) -> Molecule | None:
    if len(molecule.atoms) >= MAX_TOTAL_ATOMS:
        return None
    parents = [
        atom_id
        for atom_id, atom in molecule.atoms.items()
        if atom.attributes.symbol is not AtomicSymbol.H
        and _can_accept_extra_sigma(molecule, atom_id)
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
        and _can_accept_extra_sigma(molecule, atom_id)
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
        (atom_id, symbols)
        for atom_id, atom in molecule.atoms.items()
        if atom.attributes.symbol is not AtomicSymbol.H
        for symbols in (_legal_mutation_symbols(molecule, atom_id),)
        if symbols
    ]
    if not candidates:
        return None

    atom_id, symbols = rng.choice(sorted(candidates, key=lambda item: item[0].value))
    atom = molecule.atoms[atom_id]
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
        and _can_add_pi_ring_system(molecule, ring)
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
    if result.model_draw_source is not None:
        print(f"  FreeSolv posterior draws: {result.model_draw_source.relative_to(PROJECT_ROOT)}", file=stream)
    print(f"  seed molecule: {result.seed_molecule}", file=stream)
    print(f"  deterministic seed: {RANDOM_SEED}", file=stream)
    if result.minimum_unique_valid_molecules:
        print(f"  minimum unique valid molecules: {result.minimum_unique_valid_molecules}", file=stream)
    print(f"  total proposals: {result.diagnostics.total_proposals}", file=stream)
    print(f"  valid proposals: {result.diagnostics.valid_proposals}", file=stream)
    print(f"  invalid proposals: {result.diagnostics.invalid_proposals}", file=stream)
    print(f"  accepted proposals: {result.diagnostics.accepted_proposals}", file=stream)
    print(f"  acceptance rate: {_safe_rate(result.diagnostics.accepted_proposals, result.diagnostics.total_proposals):.3f}", file=stream)
    print(f"  invalid proposal rate: {_safe_rate(result.diagnostics.invalid_proposals, result.diagnostics.total_proposals):.3f}", file=stream)
    print(f"  unique valid molecules seen: {result.diagnostics.unique_valid_molecules_seen}", file=stream)
    print(f"  unique generated molecules: {len(result.generated_candidates)}", file=stream)
    if result.molecule_file_paths:
        print("  molecule files:", file=stream)
        for path in result.molecule_file_paths:
            print(f"    {path.relative_to(PROJECT_ROOT)}", file=stream)
    if result.dietz_file_paths:
        print("  Dietz molecule files:", file=stream)
        for path in result.dietz_file_paths:
            print(f"    {path.relative_to(PROJECT_ROOT)}", file=stream)
    if result.generated_candidate_file_paths:
        print("  generated molecule bundle files:", file=stream)
        for path in result.generated_candidate_file_paths:
            print(f"    {path.relative_to(PROJECT_ROOT)}", file=stream)
    if result.viewer_file_paths:
        print("  viewer files:", file=stream)
        for path in result.viewer_file_paths:
            print(f"    {path.relative_to(PROJECT_ROOT)}", file=stream)
    print("", file=stream)
    print("Top generated molecules by Bayesian credible score", file=stream)
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
    print(f"  Bayesian credible score: {candidate.bayesian_credible_score_percent:.2f}%", file=stream)
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
    _remove_stale_candidate_files(target_dir)
    written_paths = tuple(
        _write_candidate_python_file(target_dir, "top", rank, candidate, result.target, result.seed_molecule)
        for rank, candidate in enumerate(result.top_candidates[:TOP_K], start=1)
    )
    dietz_paths = tuple(
        _write_candidate_python_file(target_dir, "dietz", rank, candidate, result.target, result.seed_molecule)
        for rank, candidate in enumerate(result.dietz_candidates[:TOP_DIETZ_K], start=1)
    )
    generated_paths = _write_generated_candidate_bundle(
        target_dir,
        result.generated_candidates or result.top_candidates,
        result.target,
        result.seed_molecule,
    )
    return SearchResult(
        target=result.target,
        used_default_target=result.used_default_target,
        seed_molecule=result.seed_molecule,
        top_candidates=result.top_candidates,
        generated_candidates=result.generated_candidates,
        dietz_candidates=result.dietz_candidates,
        diagnostics=result.diagnostics,
        minimum_unique_valid_molecules=result.minimum_unique_valid_molecules,
        molecule_file_paths=written_paths,
        dietz_file_paths=dietz_paths,
        generated_candidate_file_paths=generated_paths,
        viewer_file_paths=result.viewer_file_paths,
        model_parameter_source=result.model_parameter_source,
        model_draw_source=result.model_draw_source,
    )


def write_result_viewer_files(result: SearchResult, output_dir: Path, *, count: int = TOP_K) -> SearchResult:
    target_dir = ensure_directory(output_dir)
    viewer_count = max(0, min(int(count), len(result.top_candidates)))
    if viewer_count == 0:
        return replace(result, viewer_file_paths=())
    path = target_dir / "top_molecules.viewer.html"
    entries = tuple(
        (
            f"FreeSolv top #{rank}: {molecular_formula(candidate.molecule)}",
            candidate.molecule,
        )
        for rank, candidate in enumerate(result.top_candidates[:viewer_count], start=1)
    )
    written = write_molecule_viewer_collection_html(
        entries,
        path,
        title=f"Top {viewer_count} FreeSolv inverse-design molecules",
    )
    return replace(result, viewer_file_paths=(written,))


def write_saved_inverse_design_viewer_file(
    result_dir: Path,
    *,
    output_path: Path | None = None,
    count: int = TOP_K,
) -> Path:
    """Write a viewer for molecules already saved by an inverse-design run."""

    source_dir = Path(result_dir)
    jsonl_path = source_dir / "generated_molecules.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing inverse-design molecule bundle: {jsonl_path}")

    viewer_count = max(0, int(count))
    entries: list[tuple[str, Molecule]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(entries) >= viewer_count:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            molecule = validate_molecule(molecule_from_dict(record["molecule"]))
            rank = int(record.get("rank", len(entries) + 1))
            formula = str(record.get("formula", molecular_formula(molecule)))
            score = float(record.get("bayesian_credible_score_percent", 0.0))
            entries.append((f"FreeSolv top #{rank}: {formula} ({score:.2f}% credible score)", molecule))

    if not entries:
        raise ValueError(f"No generated molecules found in {jsonl_path}")

    target_path = Path(output_path) if output_path is not None else source_dir / "top_molecules.viewer.html"
    target_path.unlink(missing_ok=True)
    return write_molecule_viewer_collection_html(
        tuple(entries),
        target_path,
        title=f"Top {len(entries)} FreeSolv inverse-design molecules",
    )


def open_result_viewers(paths: Sequence[Path], *, stream: TextIO) -> None:
    for path in paths:
        open_molecule_viewer(path)
        print(f"Opened viewer: {path.resolve().as_uri()}", file=stream)


def _remove_stale_candidate_files(output_dir: Path) -> None:
    for prefix in ("top", "dietz"):
        for path in output_dir.glob(f"{prefix}_*_molecule.py"):
            path.unlink()
    for path in output_dir.glob("top_*_molecule.viewer.html"):
        path.unlink()
    for path in output_dir.glob("top_molecules.viewer.html"):
        path.unlink()


def _write_generated_candidate_bundle(
    output_dir: Path,
    candidates: Sequence[Candidate],
    target: float,
    seed_molecule: str,
) -> tuple[Path, ...]:
    csv_path = output_dir / "generated_molecules.csv"
    jsonl_path = output_dir / "generated_molecules.jsonl"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "rank",
            "formula",
            "predicted_freesolv",
            "predictive_sd",
            "target_error",
            "bayesian_credible_score_percent",
            "score",
            "atoms",
            "heavy_atoms",
            "local_bonds",
            "dietz_bonding_systems",
            "seed_molecule",
            "random_seed",
            "target_freesolv",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(candidates, start=1):
            writer.writerow(_generated_candidate_metadata(rank, candidate, target, seed_molecule))

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for rank, candidate in enumerate(candidates, start=1):
            metadata = _generated_candidate_metadata(rank, candidate, target, seed_molecule)
            record = {
                **metadata,
                "molecule": json.loads(molecule_to_json_bytes(candidate.molecule)),
            }
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return (csv_path, jsonl_path)


def _generated_candidate_metadata(
    rank: int,
    candidate: Candidate,
    target: float,
    seed_molecule: str,
) -> dict[str, int | float | str]:
    molecule = candidate.molecule
    return {
        "rank": rank,
        "formula": molecular_formula(molecule),
        "predicted_freesolv": candidate.predicted_mean,
        "predictive_sd": candidate.predictive_sd,
        "target_error": abs(candidate.predicted_mean - target),
        "bayesian_credible_score_percent": candidate.bayesian_credible_score_percent,
        "score": candidate.score,
        "atoms": len(molecule.atoms),
        "heavy_atoms": heavy_atom_count(molecule),
        "local_bonds": len(molecule.local_bonds),
        "dietz_bonding_systems": len(molecule.systems),
        "seed_molecule": seed_molecule,
        "random_seed": RANDOM_SEED,
        "target_freesolv": target,
    }


def _score_molecule(predictor: FreeSolvPredictor, molecule: Molecule, target: float) -> Candidate:
    molecule = _validate_candidate(molecule)
    prediction = predictor.predict(molecule)
    score = _target_log_credible_score(prediction, target)
    bayesian_credible_score_percent = _bayesian_credible_score_percent(score)
    heavy_atoms = heavy_atom_count(molecule)
    if heavy_atoms > MAX_HEAVY_ATOMS:
        score -= 0.1 * float(heavy_atoms - MAX_HEAVY_ATOMS)
    return Candidate(
        molecule=molecule,
        predicted_mean=prediction.mean,
        predictive_sd=prediction.sd,
        bayesian_credible_score_percent=bayesian_credible_score_percent,
        score=score,
    )


def _target_log_credible_score(prediction: Prediction, target: float) -> float:
    variance = prediction.sd**2 + CREDIBLE_SCORE_NOISE_FLOOR**2
    target_error = prediction.mean - target
    return -0.5 * (target_error**2 / variance) - 0.5 * math.log(variance)


def _bayesian_credible_score_percent(score: float) -> float:
    return max(0.0, min(100.0, 100.0 * math.exp(min(0.0, score))))


def _load_gp_parameter_means(model_dir: Path) -> FreeSolvModelParameters:
    coefficients_path = model_dir / "details" / "model_coefficients.csv"
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


def _load_gp_posterior_draws(model_dir: Path) -> FreeSolvPosteriorDraws:
    draws_path = _find_gp_draws_path(model_dir)
    draw_frame = pd.read_csv(draws_path, comment="#")
    required = ("alpha", "signal_scale", "lengthscale", "sigma")
    missing = [name for name in required if name not in draw_frame.columns]
    if missing:
        raise RuntimeError(f"Missing FreeSolv GP posterior draw columns: {', '.join(missing)}")
    alpha = draw_frame["alpha"].to_numpy(dtype=float)
    signal_scale = draw_frame["signal_scale"].to_numpy(dtype=float)
    lengthscale = draw_frame["lengthscale"].to_numpy(dtype=float)
    sigma = draw_frame["sigma"].to_numpy(dtype=float)
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
        raise RuntimeError(f"No finite FreeSolv GP posterior draws found in {draws_path}")
    return FreeSolvPosteriorDraws(
        alpha=alpha[finite_mask],
        signal_scale=signal_scale[finite_mask],
        lengthscale=lengthscale[finite_mask],
        sigma=sigma[finite_mask],
        source_path=draws_path,
    )


def _find_gp_draws_path(model_dir: Path) -> Path:
    draws_dir = (
        model_dir
        / "details"
        / "stan_output"
        / "freesolv"
        / "moladt_featurized"
        / MODEL_NAME
        / METHOD_NAME
    )
    draw_files = tuple(sorted(draws_dir.glob("*.csv")))
    if len(draw_files) != 1:
        raise FileNotFoundError(f"Expected one latest FreeSolv GP posterior draw CSV in {draws_dir}; found {len(draw_files)}")
    return draw_files[0]


def _find_model_dir() -> Path:
    run_dirs = tuple(
        sorted(
            (path for path in FREESOLV_RESULTS_DIR.glob("run_*") if path.is_dir()),
            key=lambda path: path.name,
        )
    )
    if not run_dirs:
        raise FileNotFoundError(f"Missing FreeSolv result runs under {FREESOLV_RESULTS_DIR}")

    model_dir = run_dirs[-1]
    coefficients_path = model_dir / "details" / "model_coefficients.csv"
    if not coefficients_path.exists():
        raise FileNotFoundError(
            "Latest FreeSolv run is missing the Bayesian GP coefficient artifact: "
            f"{coefficients_path}"
        )
    _find_gp_draws_path(model_dir)
    return model_dir


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


def _precompute_gp_draw_weights(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: np.ndarray,
    signal_scale: np.ndarray,
    lengthscale: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    weights = np.empty((alpha.shape[0], X_train.shape[0]), dtype=float)
    for draw_index, (draw_alpha, draw_signal, draw_length, draw_sigma) in enumerate(
        zip(alpha, signal_scale, lengthscale, sigma, strict=True)
    ):
        train_kernel = _rbf_kernel(X_train, X_train, lengthscale=draw_length, signal_scale=draw_signal)
        train_kernel[np.diag_indices_from(train_kernel)] += draw_sigma**2 + 1e-8
        try:
            chol = np.linalg.cholesky(train_kernel)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"FreeSolv GP covariance was not positive definite for posterior draw {draw_index}") from exc
        centered_y = y_train - draw_alpha
        weights[draw_index] = np.linalg.solve(chol.T, np.linalg.solve(chol, centered_y))
    return weights


def _gp_mean_by_draw(
    *,
    x_eval: np.ndarray,
    X_train: np.ndarray,
    alpha: np.ndarray,
    signal_scale: np.ndarray,
    lengthscale: np.ndarray,
    draw_weights: np.ndarray,
) -> np.ndarray:
    sqdist = np.sum(np.square(X_train - x_eval.reshape(1, -1)), axis=1)
    cross_kernel_by_draw = np.square(signal_scale)[:, np.newaxis] * np.exp(
        -0.5 * sqdist[np.newaxis, :] / np.maximum(np.square(lengthscale), 1e-9)[:, np.newaxis]
    )
    return alpha + np.einsum("dn,dn->d", cross_kernel_by_draw, draw_weights)


def _rbf_kernel(X_left: np.ndarray, X_right: np.ndarray, *, lengthscale: float, signal_scale: float) -> np.ndarray:
    left_sq = np.sum(np.square(X_left), axis=1)[:, np.newaxis]
    right_sq = np.sum(np.square(X_right), axis=1)[np.newaxis, :]
    sqdist = np.maximum(left_sq + right_sq - 2.0 * (X_left @ X_right.T), 0.0)
    return (signal_scale**2) * np.exp(-0.5 * sqdist / max(lengthscale**2, 1e-9))


def _candidate_sort_key(target: float) -> Callable[[Candidate], tuple[float, float, float, int, int]]:
    def sort_key(candidate: Candidate) -> tuple[float, float, float, int, int]:
        return (
            candidate.bayesian_credible_score_percent,
            -candidate.predictive_sd,
            -abs(candidate.predicted_mean - target),
            1 if candidate.molecule.systems else 0,
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
            "from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system",
            "from moladt.chem.molecule import AtomicSymbol, Molecule",
            "from moladt.chem.validate import validate_molecule",
            "from moladt.examples._literal import atom",
            "",
            f"rank = {rank}",
            f"target_freesolv = {target:.12g}",
            f"seed_molecule = {seed_molecule!r}",
            f"random_seed = {RANDOM_SEED}",
            f"predicted_freesolv = {candidate.predicted_mean:.12g}",
            f"predictive_sd = {candidate.predictive_sd:.12g}",
            f"target_error = {abs(candidate.predicted_mean - target):.12g}",
            f"bayesian_credible_score_percent = {candidate.bayesian_credible_score_percent:.12g}",
            f"score = {candidate.score:.12g}",
            f"formula = {molecular_formula(molecule)!r}",
            "",
            "molecule = validate_molecule(",
            "    Molecule(",
            "        atoms={",
            *_atom_literal_lines(molecule),
            "        },",
            "        local_bonds=frozenset(",
            "            {",
            *_edge_literal_lines(molecule.local_bonds),
            "            }",
            "        ),",
            "        systems=(",
            *_system_literal_lines(molecule),
            "        ),",
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
            "    'bayesian_credible_score_percent',",
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
        charge_arg = "" if atom.formal_charge == 0 else f", formal_charge={atom.formal_charge}"
        lines.append(
            f"            AtomId({atom_id.value}): atom("
            f"{atom_id.value}, AtomicSymbol.{symbol}, "
            f"{atom.coordinate.x.value:.3f}, "
            f"{atom.coordinate.y.value:.3f}, "
            f"{atom.coordinate.z.value:.3f}{charge_arg}),"
        )
    return tuple(lines)


def _edge_literal_lines(edges: frozenset[Edge] | set[Edge]) -> tuple[str, ...]:
    return tuple(f"                Edge(AtomId({edge.a.value}), AtomId({edge.b.value}))," for edge in sorted(edges))


def _system_literal_lines(molecule: Molecule) -> tuple[str, ...]:
    lines: list[str] = []
    for system_id, system in molecule.systems:
        lines.append("            (")
        lines.append(f"                SystemId({system_id.value}),")
        lines.append("                mk_bonding_system(")
        lines.append(f"                    NonNegative({system.shared_electrons.value}),")
        lines.append("                    frozenset(")
        lines.append("                        {")
        lines.extend(
            f"                            Edge(AtomId({edge.a.value}), AtomId({edge.b.value})),"
            for edge in sorted(system.member_edges)
        )
        lines.append("                        }")
        lines.append("                    ),")
        lines.append(f"                    {system.tag!r},")
        lines.append("                ),")
        lines.append("            ),")
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


def _can_accept_extra_sigma(molecule: Molecule, atom_id: AtomId) -> bool:
    atom = molecule.atoms[atom_id]
    if atom.attributes.symbol in TERMINAL_FREE_SOLV_SYMBOLS:
        return False
    return _hydrogen_slots_after(
        molecule,
        atom_id,
        atom.attributes.symbol,
        extra_used=1.0,
    ) is not None


def _legal_mutation_symbols(molecule: Molecule, atom_id: AtomId) -> tuple[AtomicSymbol, ...]:
    atom = molecule.atoms[atom_id]
    if atom.attributes.symbol is AtomicSymbol.H or _atom_participates_in_system(molecule, atom_id):
        return ()
    symbols: list[AtomicSymbol] = []
    for symbol in MUTATION_SYMBOLS:
        if symbol is atom.attributes.symbol:
            continue
        hydrogen_slots = _hydrogen_slots_after(molecule, atom_id, symbol)
        if hydrogen_slots is None:
            continue
        if symbol in TERMINAL_FREE_SOLV_SYMBOLS and not _can_be_terminal_after_completion(
            molecule,
            atom_id,
            hydrogen_slots=hydrogen_slots,
        ):
            continue
        symbols.append(symbol)
    return tuple(symbols)


def _can_add_pi_ring_system(molecule: Molecule, ring: frozenset[Edge]) -> bool:
    system = mk_bonding_system(NonNegative(6), ring, "pi_ring")
    if not _is_valid_pi_ring(molecule, system):
        return False
    if any(_has_localized_singleton_system(molecule, edge) for edge in ring):
        return False
    ring_atoms = {atom_id for edge in ring for atom_id in (edge.a, edge.b)}
    return all(
        _hydrogen_slots_after(molecule, atom_id, AtomicSymbol.C, extra_used=1.0) is not None
        for atom_id in ring_atoms
    )


def _hydrogen_slots_after(
    molecule: Molecule,
    atom_id: AtomId,
    symbol: AtomicSymbol,
    *,
    extra_used: float = 0.0,
) -> int | None:
    base_used = _used_electrons_without_terminal_hydrogens(molecule, atom_id) + extra_used
    available = _growth_max_valence(symbol) - base_used
    if available < -VALENCE_TOLERANCE:
        return None
    rounded = round(available)
    if abs(available - rounded) > VALENCE_TOLERANCE:
        return None
    return max(0, int(rounded))


def _used_electrons_without_terminal_hydrogens(molecule: Molecule, atom_id: AtomId) -> float:
    return used_electrons_at(molecule, atom_id) - float(len(_terminal_hydrogens_attached_to(molecule, atom_id)))


def _can_be_terminal_after_completion(molecule: Molecule, atom_id: AtomId, *, hydrogen_slots: int) -> bool:
    return (
        not _atom_participates_in_system(molecule, atom_id)
        and _sigma_degree_without_terminal_hydrogens(molecule, atom_id) + hydrogen_slots == 1
    )


def _sigma_degree_without_terminal_hydrogens(molecule: Molecule, atom_id: AtomId) -> int:
    removable_hydrogens = set(_terminal_hydrogens_attached_to(molecule, atom_id))
    return sum(1 for neighbor in neighbors_sigma(molecule, atom_id) if neighbor not in removable_hydrogens)


def _atom_participates_in_system(molecule: Molecule, atom_id: AtomId) -> bool:
    return any(atom_id in system.member_atoms for _, system in molecule.systems)


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
    _ensure_neutral_formal_charges(molecule)
    _ensure_no_hydrogen_hydrogen_local_bonds(molecule)
    _ensure_terminal_atom_rules(molecule)
    _ensure_conservative_generator_valence(molecule)
    _ensure_closed_valence_shells(molecule)
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


def _ensure_neutral_formal_charges(molecule: Molecule) -> None:
    charged_atoms = [
        atom_id.value
        for atom_id, atom in molecule.atoms.items()
        if atom.formal_charge != 0
    ]
    if charged_atoms:
        formatted = ", ".join(str(atom_id) for atom_id in charged_atoms)
        raise ValidationError(f"FreeSolv generated candidates must be neutral; charged atoms: {formatted}")


def _ensure_no_hydrogen_hydrogen_local_bonds(molecule: Molecule) -> None:
    for edge in molecule.local_bonds:
        if (
            molecule.atoms[edge.a].attributes.symbol is AtomicSymbol.H
            and molecule.atoms[edge.b].attributes.symbol is AtomicSymbol.H
        ):
            raise ValidationError("Generated molecules may not contain H-H local bonds")


def _ensure_terminal_atom_rules(molecule: Molecule) -> None:
    system_atoms = {atom_id for _, system in molecule.systems for atom_id in system.member_atoms}
    for atom_id, atom in molecule.atoms.items():
        symbol = atom.attributes.symbol
        if symbol not in TERMINAL_FREE_SOLV_SYMBOLS:
            continue
        sigma_neighbors = tuple(neighbors_sigma(molecule, atom_id))
        if atom_id in system_atoms:
            raise ValidationError(f"Terminal element {symbol.value} may not participate in Dietz systems")
        if len(sigma_neighbors) != 1:
            raise ValidationError(f"Terminal element {symbol.value}#{atom_id.value} must have exactly one sigma bond")
        if symbol is AtomicSymbol.H:
            neighbor = molecule.atoms[sigma_neighbors[0]]
            if neighbor.attributes.symbol is AtomicSymbol.H:
                raise ValidationError("Generated molecules may not contain H-H local bonds")


def _ensure_conservative_generator_valence(molecule: Molecule) -> None:
    for atom_id, atom in molecule.atoms.items():
        used = used_electrons_at(molecule, atom_id)
        if used > _growth_max_valence(atom.attributes.symbol) + VALENCE_TOLERANCE:
            raise ValidationError(f"Atom {atom_id.value} exceeds generator valence cap")


def _ensure_closed_valence_shells(molecule: Molecule) -> None:
    for atom_id, atom in molecule.atoms.items():
        expected = _growth_max_valence(atom.attributes.symbol)
        used = used_electrons_at(molecule, atom_id)
        if not math.isclose(used, expected, abs_tol=VALENCE_TOLERANCE):
            raise ValidationError(
                f"Atom {atom_id.value} has incomplete generated valence shell "
                f"({used:.3g} used, {expected:.3g} expected)"
            )


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
        hydrogen_count = max(0, int(math.floor(available + VALENCE_TOLERANCE)))
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
    chemistry_key = {
        "atoms": [
            (atom_id.value, atom.attributes.symbol.value, atom.formal_charge)
            for atom_id, atom in sorted(molecule.atoms.items(), key=lambda item: item[0].value)
        ],
        "local_bonds": [(edge.a.value, edge.b.value) for edge in sorted(molecule.local_bonds)],
        "systems": [
            (
                system_id.value,
                system.shared_electrons.value,
                [(edge.a.value, edge.b.value) for edge in sorted(system.member_edges)],
                system.tag,
            )
            for system_id, system in molecule.systems
        ],
        "smiles_atom_stereo": [
            (
                stereo.center.value,
                stereo.stereo_class.value,
                stereo.configuration,
                stereo.token,
            )
            for stereo in molecule.smiles_stereochemistry.atom_stereo
        ],
        "smiles_bond_stereo": [
            (
                stereo.start_atom.value,
                stereo.end_atom.value,
                stereo.direction.value,
            )
            for stereo in molecule.smiles_stereochemistry.bond_stereo
        ],
    }
    return json.dumps(chemistry_key, sort_keys=True, separators=(",", ":")).encode("utf-8")


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
