from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from moladt.chem.dietz import mk_edge
from moladt.chem.molecule import AtomicSymbol, Molecule, effective_order
from moladt.inference import compute_descriptors as compute_moladt_descriptors
from moladt.inference.descriptors import coordinate_descriptors
from moladt.io.sdf import parse_sdf_record
from moladt.io.smiles import molecule_to_smiles, parse_smiles

from .common import FailureRecord

if TYPE_CHECKING:
    from rdkit import Chem

_SMILES_HASH_BINS = 64
_SMILES_SUMMARY_FEATURE_GROUPS: dict[str, str] = {
    "smiles_char_length": "smiles_summary",
    "smiles_token_count": "smiles_summary",
    "smiles_branch_open_count": "smiles_summary",
    "smiles_branch_close_count": "smiles_summary",
    "smiles_ring_marker_count": "smiles_summary",
    "smiles_aromatic_token_count": "smiles_summary",
    "smiles_bracket_token_count": "smiles_summary",
    "smiles_bond_token_count": "smiles_summary",
    "smiles_directional_bond_count": "smiles_summary",
    "smiles_chirality_marker_count": "smiles_summary",
    "smiles_charge_marker_count": "smiles_summary",
    "smiles_component_count": "smiles_summary",
}
_SMILES_TOKEN_HASH_FEATURE_GROUPS: dict[str, str] = {
    f"smiles_token_hash_{index:02d}": "smiles_token_hash"
    for index in range(_SMILES_HASH_BINS)
}
_SMILES_BIGRAM_HASH_FEATURE_GROUPS: dict[str, str] = {
    f"smiles_bigram_hash_{index:02d}": "smiles_bigram_hash"
    for index in range(_SMILES_HASH_BINS)
}

BASE_FEATURE_GROUPS: dict[str, str] = {
    **_SMILES_SUMMARY_FEATURE_GROUPS,
    **_SMILES_TOKEN_HASH_FEATURE_GROUPS,
    **_SMILES_BIGRAM_HASH_FEATURE_GROUPS,
}

SDF_GEOMETRY_FEATURE_GROUPS: dict[str, str] = {
    "z_mean": "geometry_atoms",
    "z_max": "geometry_atoms",
    "atom_count": "geometry_atoms",
}

THREED_FEATURE_GROUPS: dict[str, str] = {
    "radius_of_gyration": "shape_3d",
    "asphericity": "shape_3d",
    "eccentricity": "shape_3d",
    "inertial_shape_factor": "shape_3d",
    "npr1": "shape_3d",
    "npr2": "shape_3d",
    "pmi1": "shape_3d",
    "pmi2": "shape_3d",
    "pmi3": "shape_3d",
    "distance_mean": "distance_3d",
    "distance_std": "distance_3d",
    "distance_max": "distance_3d",
}

MOLADT_FEATURE_GROUPS: dict[str, str] = {
    "weight": "adt_composition",
    "polar": "adt_composition",
    "surface": "adt_size_proxy",
    "bond_order": "adt_topology",
    "donor_count": "adt_polarity",
    "acceptor_count": "adt_polarity",
    "heavy_atoms": "adt_composition",
    "halogens": "adt_composition",
    "atom_count_c": "adt_elements",
    "atom_count_n": "adt_elements",
    "atom_count_o": "adt_elements",
    "atom_count_f": "adt_elements",
    "atom_count_p": "adt_elements",
    "atom_count_s": "adt_elements",
    "atom_count_cl": "adt_elements",
    "atom_count_br": "adt_elements",
    "atom_count_i": "adt_elements",
    "atom_count_h": "adt_elements",
    "formal_charge_sum": "adt_charge",
    "abs_formal_charge_sum": "adt_charge",
    "positive_charge_count": "adt_charge",
    "negative_charge_count": "adt_charge",
    "bonding_system_count": "adt_bonding",
    "multicentre_system_count": "adt_bonding",
    "pi_ring_system_count": "adt_bonding",
    "zero_electron_system_count": "adt_bonding",
    "sigma_edge_count": "adt_bonding",
    "effective_bond_order_sum": "adt_bonding",
    "effective_bond_order_mean": "adt_bonding",
    "effective_bond_order_max": "adt_bonding",
    "aromatic_rings": "adt_topology",
    "aromatic_atom_count": "adt_topology",
    "aromatic_atom_fraction": "adt_topology",
    "ring_edge_fraction": "adt_topology",
    "rotatable_bonds": "adt_topology",
    "heavy_atom_degree_mean": "adt_topology",
    "heavy_atom_degree_max": "adt_topology",
}

MOLADT_GEOMETRY_FEATURE_GROUPS: dict[str, str] = {
    **MOLADT_FEATURE_GROUPS,
    "radius_of_gyration": "adt_geometry",
    "distance_mean": "adt_geometry",
    "distance_std": "adt_geometry",
    "distance_max": "adt_geometry",
    "inertia_eigenvalue_min": "adt_geometry",
    "inertia_eigenvalue_mid": "adt_geometry",
    "inertia_eigenvalue_max": "adt_geometry",
}

_PAIR_SYMBOLS: tuple[str, ...] = tuple(symbol.value for symbol in AtomicSymbol)
_PAIR_SYMBOL_ORDER = {symbol: index for index, symbol in enumerate(_PAIR_SYMBOLS)}
_PAIR_RADIAL_CENTERS: tuple[float, ...] = (1.5, 2.5, 3.5, 4.5)
_PAIR_RADIAL_SIGMA = 0.75
_ANGLE_CENTERS_DEGREES: tuple[float, ...] = (60.0, 90.0, 120.0, 180.0)
_ANGLE_SIGMA_DEGREES = 18.0
_DIHEDRAL_CENTERS_DEGREES: tuple[float, ...] = (0.0, 60.0, 120.0, 180.0)
_DIHEDRAL_SIGMA_DEGREES = 20.0


def _pair_feature_token(symbol: str) -> str:
    return symbol.lower()


def _pair_feature_name(prefix: str, left_symbol: str, right_symbol: str) -> str:
    left_token = _pair_feature_token(left_symbol)
    right_token = _pair_feature_token(right_symbol)
    return f"{prefix}_{left_token}_{right_token}"


def _radial_feature_name(prefix: str, center: float) -> str:
    return f"{prefix}_{center:.1f}a".replace(".", "p")


def _angular_feature_name(prefix: str, center: float) -> str:
    return f"{prefix}_{int(center)}d"


_TYPED_PAIR_COUNT_FEATURES: dict[str, str] = {
    _pair_feature_name("pair_count", left_symbol, right_symbol): "adt_typed_pairs"
    for index, left_symbol in enumerate(_PAIR_SYMBOLS)
    for right_symbol in _PAIR_SYMBOLS[index:]
}
_TYPED_PAIR_INTERACTION_FEATURES: dict[str, str] = {
    _pair_feature_name("pair_interaction", left_symbol, right_symbol): "adt_typed_pairs"
    for index, left_symbol in enumerate(_PAIR_SYMBOLS)
    for right_symbol in _PAIR_SYMBOLS[index:]
}
_TYPED_SYSTEM_FEATURES: dict[str, str] = {
    "system_member_atoms_mean": "adt_typed_systems",
    "system_member_atoms_max": "adt_typed_systems",
    "system_member_edges_mean": "adt_typed_systems",
    "system_member_edges_max": "adt_typed_systems",
    "system_shared_electrons_sum": "adt_typed_systems",
    "system_shared_electrons_mean": "adt_typed_systems",
    "system_shared_electrons_max": "adt_typed_systems",
}
_TYPED_EDGE_BUCKET_FEATURES: dict[str, str] = {
    "edge_order_sigma_like_count": "adt_typed_edge_order",
    "edge_order_delocalized_count": "adt_typed_edge_order",
    "edge_order_double_like_count": "adt_typed_edge_order",
    "edge_order_triple_plus_count": "adt_typed_edge_order",
}
_TYPED_RADIAL_FEATURES: dict[str, str] = {
    **{_radial_feature_name("aprdf_all", center): "adt_typed_radial" for center in _PAIR_RADIAL_CENTERS},
    **{_radial_feature_name("aprdf_edge_order", center): "adt_typed_radial" for center in _PAIR_RADIAL_CENTERS},
    **{_radial_feature_name("aprdf_system_edge", center): "adt_typed_radial" for center in _PAIR_RADIAL_CENTERS},
}
_TYPED_ANGLE_FEATURES: dict[str, str] = {
    **{_angular_feature_name("bond_angle_all", center): "adt_typed_angles" for center in _ANGLE_CENTERS_DEGREES},
    **{_angular_feature_name("bond_angle_distance_weighted", center): "adt_typed_angles" for center in _ANGLE_CENTERS_DEGREES},
    **{_angular_feature_name("bond_angle_order_weighted", center): "adt_typed_angles" for center in _ANGLE_CENTERS_DEGREES},
}
_TYPED_DIHEDRAL_FEATURES: dict[str, str] = {
    **{_angular_feature_name("torsion_all", center): "adt_typed_torsions" for center in _DIHEDRAL_CENTERS_DEGREES},
    **{_angular_feature_name("torsion_distance_weighted", center): "adt_typed_torsions" for center in _DIHEDRAL_CENTERS_DEGREES},
    **{_angular_feature_name("torsion_order_weighted", center): "adt_typed_torsions" for center in _DIHEDRAL_CENTERS_DEGREES},
}

MOLADT_FEATURIZED_FEATURE_GROUPS: dict[str, str] = {
    **MOLADT_FEATURE_GROUPS,
    **_TYPED_PAIR_COUNT_FEATURES,
    **_TYPED_PAIR_INTERACTION_FEATURES,
    **_TYPED_SYSTEM_FEATURES,
    **_TYPED_EDGE_BUCKET_FEATURES,
    **_TYPED_RADIAL_FEATURES,
    **_TYPED_ANGLE_FEATURES,
    **_TYPED_DIHEDRAL_FEATURES,
}


_SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\[\]]+\]|Br|Cl|Si|Na|Fe|se|as|B|C|N|O|S|P|F|I|b|c|n|o|p|s|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@@?|%[0-9]{2}|[0-9]|\*)"
)


def tokenize_smiles_lexically(smiles: str) -> tuple[str, ...]:
    tokens = _SMILES_TOKEN_PATTERN.findall(smiles)
    if "".join(tokens) != smiles:
        raise ValueError(f"Unsupported SMILES token sequence: {smiles}")
    if not tokens:
        raise ValueError("SMILES string is empty")
    return tuple(tokens)


def _stable_hash_bin(prefix: str, token: str, *, bins: int) -> int:
    digest = hashlib.blake2b(f"{prefix}:{token}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % bins


def compute_smiles_string_features(smiles: str) -> dict[str, float]:
    tokens = tokenize_smiles_lexically(smiles)
    features = {name: 0.0 for name in BASE_FEATURE_GROUPS}
    features["smiles_char_length"] = float(len(smiles))
    features["smiles_token_count"] = float(len(tokens))
    features["smiles_branch_open_count"] = float(sum(token == "(" for token in tokens))
    features["smiles_branch_close_count"] = float(sum(token == ")" for token in tokens))
    features["smiles_ring_marker_count"] = float(sum(token.isdigit() or token.startswith("%") for token in tokens))
    features["smiles_aromatic_token_count"] = float(sum(token in {"b", "c", "n", "o", "p", "s", "se", "as"} for token in tokens))
    features["smiles_bracket_token_count"] = float(sum(token.startswith("[") for token in tokens))
    features["smiles_bond_token_count"] = float(sum(token in {"-", "=", "#", ":", "/", "\\"} for token in tokens))
    features["smiles_directional_bond_count"] = float(sum(token in {"/", "\\"} for token in tokens))
    features["smiles_chirality_marker_count"] = float(sum("@" in token for token in tokens))
    features["smiles_charge_marker_count"] = float(sum(("+" in token) or ("-" in token and token != "-") for token in tokens))
    features["smiles_component_count"] = float(smiles.count(".") + 1)
    for token in tokens:
        token_bin = _stable_hash_bin("token", token, bins=_SMILES_HASH_BINS)
        features[f"smiles_token_hash_{token_bin:02d}"] += 1.0
    for left, right in zip(tokens, tokens[1:], strict=False):
        bigram_bin = _stable_hash_bin("bigram", f"{left}>{right}", bins=_SMILES_HASH_BINS)
        features[f"smiles_bigram_hash_{bigram_bin:02d}"] += 1.0
    return features


@dataclass(frozen=True, slots=True)
class FeatureTable:
    rows: pd.DataFrame
    feature_names: tuple[str, ...]
    feature_groups: dict[str, str]
    failures: tuple[FailureRecord, ...]


@dataclass(frozen=True, slots=True)
class GeometricFeatureTable:
    rows: pd.DataFrame
    atomic_numbers: tuple[np.ndarray, ...]
    coordinates: tuple[np.ndarray, ...]
    global_feature_names: tuple[str, ...]
    global_feature_groups: dict[str, str]
    global_features: np.ndarray | None
    failures: tuple[FailureRecord, ...]


def canonicalize_smiles(smiles: str) -> str:
    chem = _import_rdkit_chem()
    if chem is None:
        molecule = parse_smiles(smiles)
        return molecule_to_smiles(molecule)
    molecule = chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def canonical_smiles_from_mol(molecule: "Chem.Mol") -> str:
    chem = _require_rdkit_chem()
    sanitized = _sanitize_rdkit_mol(molecule)
    return chem.MolToSmiles(sanitized, canonical=True, isomericSmiles=True)


def canonical_smiles_from_molecule(molecule: Molecule) -> str:
    return molecule_to_smiles(molecule)


def _coerce_moladt_molecule(raw_molecule: Any) -> Molecule:
    if isinstance(raw_molecule, Molecule):
        return raw_molecule
    if hasattr(raw_molecule, "GetAtoms"):
        return rdkit_mol_to_moladt_record(raw_molecule).molecule
    raise TypeError(f"Unsupported molecule object: {type(raw_molecule).__name__}")


def featurize_smiles_dataframe(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    mol_id_column: str,
    smiles_column: str,
    target_column: str,
) -> FeatureTable:
    rows: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    for record in dataframe.itertuples(index=False):
        mol_id = str(getattr(record, mol_id_column))
        smiles = str(getattr(record, smiles_column))
        target = float(getattr(record, target_column))
        try:
            base = compute_smiles_string_features(smiles)
        except Exception as exc:
            failures.append(FailureRecord(dataset=dataset_name, mol_id=mol_id, stage="smiles_featurize", error=str(exc)))
            continue
        row = {"mol_id": mol_id, "smiles": smiles, target_column: target}
        row.update(base)
        rows.append(row)
    feature_names = tuple(BASE_FEATURE_GROUPS)
    return FeatureTable(rows=pd.DataFrame(rows), feature_names=feature_names, feature_groups=dict(BASE_FEATURE_GROUPS), failures=tuple(failures))


def featurize_sdf_records(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    mol_id_column: str,
    mol_column: str,
    target_column: str,
    record_index_column: str | None = None,
) -> FeatureTable:
    rows: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    feature_groups = {**BASE_FEATURE_GROUPS, **THREED_FEATURE_GROUPS}
    for record in dataframe.itertuples(index=False):
        mol_id = str(getattr(record, mol_id_column))
        target = float(getattr(record, target_column))
        raw_molecule = getattr(record, mol_column)
        try:
            molecule = _sanitize_rdkit_mol(raw_molecule)
            canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            features = compute_base_descriptors(molecule)
            features.update(compute_3d_descriptors(molecule))
        except Exception as exc:
            failures.append(FailureRecord(dataset=dataset_name, mol_id=mol_id, stage="sdf_featurize", error=str(exc)))
            continue
        row: dict[str, Any] = {"mol_id": mol_id, "smiles": canonical, target_column: target}
        if record_index_column is not None:
            row[record_index_column] = int(getattr(record, record_index_column))
        row.update(features)
        rows.append(row)
    feature_names = tuple(feature_groups)
    return FeatureTable(rows=pd.DataFrame(rows), feature_names=feature_names, feature_groups=feature_groups, failures=tuple(failures))


def featurize_moladt_records(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    mol_id_column: str,
    mol_column: str,
    target_column: str,
    record_index_column: str | None = None,
) -> FeatureTable:
    rows: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    for record in dataframe.itertuples(index=False):
        mol_id = str(getattr(record, mol_id_column))
        target = float(getattr(record, target_column))
        molecule = getattr(record, mol_column)
        try:
            molecule = _coerce_moladt_molecule(molecule)
            canonical = canonical_smiles_from_molecule(molecule)
            features = compute_moladt_descriptors(molecule).to_dict()
        except Exception as exc:
            failures.append(FailureRecord(dataset=dataset_name, mol_id=mol_id, stage="moladt_featurize", error=str(exc)))
            continue
        row: dict[str, Any] = {"mol_id": mol_id, "smiles": canonical, target_column: target}
        if record_index_column is not None:
            row[record_index_column] = int(getattr(record, record_index_column))
        row.update(features)
        rows.append(row)
    feature_names = tuple(MOLADT_FEATURE_GROUPS)
    return FeatureTable(rows=pd.DataFrame(rows), feature_names=feature_names, feature_groups=dict(MOLADT_FEATURE_GROUPS), failures=tuple(failures))


def featurize_moladt_featurized_records(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    mol_id_column: str,
    mol_column: str,
    target_column: str,
    record_index_column: str | None = None,
) -> FeatureTable:
    # Legacy richer descriptor helper kept for side experiments. The main
    # benchmark contract is `smiles` vs `moladt`; this path changes the
    # information budget and is intentionally not part of the default
    # representation comparison.
    rows: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    for record in dataframe.itertuples(index=False):
        mol_id = str(getattr(record, mol_id_column))
        target = float(getattr(record, target_column))
        molecule = getattr(record, mol_column)
        try:
            molecule = _coerce_moladt_molecule(molecule)
            canonical = canonical_smiles_from_molecule(molecule)
            features = compute_moladt_featurized_descriptors(molecule)
        except Exception as exc:
            failures.append(FailureRecord(dataset=dataset_name, mol_id=mol_id, stage="moladt_featurized_featurize", error=str(exc)))
            continue
        row: dict[str, Any] = {"mol_id": mol_id, "smiles": canonical, target_column: target}
        if record_index_column is not None:
            row[record_index_column] = int(getattr(record, record_index_column))
        row.update(features)
        rows.append(row)
    feature_names = tuple(MOLADT_FEATURIZED_FEATURE_GROUPS)
    return FeatureTable(
        rows=pd.DataFrame(rows),
        feature_names=feature_names,
        feature_groups=dict(MOLADT_FEATURIZED_FEATURE_GROUPS),
        failures=tuple(failures),
    )


def featurize_moladt_smiles_dataframe(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    mol_id_column: str,
    smiles_column: str,
    target_column: str,
) -> FeatureTable:
    rows: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    for record in dataframe.itertuples(index=False):
        mol_id = str(getattr(record, mol_id_column))
        smiles = str(getattr(record, smiles_column))
        target = float(getattr(record, target_column))
        try:
            moladt_molecule = parse_smiles(smiles)
            features = compute_moladt_descriptors(moladt_molecule).to_dict()
        except Exception as exc:
            failures.append(FailureRecord(dataset=dataset_name, mol_id=mol_id, stage="moladt_smiles_featurize", error=str(exc)))
            continue
        row: dict[str, Any] = {"mol_id": mol_id, "smiles": smiles, target_column: target}
        row.update(features)
        rows.append(row)
    feature_names = tuple(MOLADT_FEATURE_GROUPS)
    return FeatureTable(rows=pd.DataFrame(rows), feature_names=feature_names, feature_groups=dict(MOLADT_FEATURE_GROUPS), failures=tuple(failures))


def featurize_sdf_geometry_records(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    mol_id_column: str,
    mol_column: str,
    target_column: str,
    record_index_column: str | None = None,
) -> GeometricFeatureTable:
    rows: list[dict[str, Any]] = []
    atomic_numbers: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    global_rows: list[dict[str, float]] = []
    failures: list[FailureRecord] = []
    feature_names = tuple(SDF_GEOMETRY_FEATURE_GROUPS)
    for record in dataframe.itertuples(index=False):
        mol_id = str(getattr(record, mol_id_column))
        target = float(getattr(record, target_column))
        molecule = getattr(record, mol_column)
        try:
            molecule = _coerce_moladt_molecule(molecule)
            ordered_atoms = [molecule.atoms[atom_id] for atom_id in sorted(molecule.atoms)]
            z = np.asarray([atom.attributes.atomic_number for atom in ordered_atoms], dtype=np.int64)
            pos = np.asarray(
                [[atom.coordinate.x.value, atom.coordinate.y.value, atom.coordinate.z.value] for atom in ordered_atoms],
                dtype=float,
            )
            canonical = canonical_smiles_from_molecule(molecule)
            if pos.shape != (len(z), 3):
                raise ValueError(f"Expected coordinates with shape ({len(z)}, 3) but found {pos.shape}")
        except Exception as exc:
            failures.append(FailureRecord(dataset=dataset_name, mol_id=mol_id, stage="sdf_geometry_featurize", error=str(exc)))
            continue
        row: dict[str, Any] = {"mol_id": mol_id, "smiles": canonical, target_column: target}
        if record_index_column is not None:
            row[record_index_column] = int(getattr(record, record_index_column))
        rows.append(row)
        atomic_numbers.append(z)
        coordinates.append(pos)
        global_rows.append(
            {
                "z_mean": float(np.mean(z)) if len(z) else 0.0,
                "z_max": float(np.max(z)) if len(z) else 0.0,
                "atom_count": float(len(z)),
            }
        )
    return GeometricFeatureTable(
        rows=pd.DataFrame(rows),
        atomic_numbers=tuple(atomic_numbers),
        coordinates=tuple(coordinates),
        global_feature_names=feature_names,
        global_feature_groups=dict(SDF_GEOMETRY_FEATURE_GROUPS),
        global_features=np.asarray([[row[name] for name in feature_names] for row in global_rows], dtype=float) if global_rows else None,
        failures=tuple(failures),
    )


def featurize_moladt_geometry_records(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    mol_id_column: str,
    mol_column: str,
    target_column: str,
    record_index_column: str | None = None,
) -> GeometricFeatureTable:
    rows: list[dict[str, Any]] = []
    atomic_numbers: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    global_rows: list[dict[str, float]] = []
    failures: list[FailureRecord] = []
    feature_names = tuple(MOLADT_FEATURE_GROUPS)
    for record in dataframe.itertuples(index=False):
        mol_id = str(getattr(record, mol_id_column))
        target = float(getattr(record, target_column))
        molecule = getattr(record, mol_column)
        try:
            molecule = _coerce_moladt_molecule(molecule)
            canonical = canonical_smiles_from_molecule(molecule)
            descriptor_dict = compute_moladt_descriptors(molecule).to_dict()
            descriptor_dict.update(coordinate_descriptors(molecule))
            ordered_atoms = [molecule.atoms[atom_id] for atom_id in sorted(molecule.atoms)]
            z = np.asarray([atom.attributes.atomic_number for atom in ordered_atoms], dtype=np.int64)
            pos = np.asarray(
                [
                    [atom.coordinate.x.value, atom.coordinate.y.value, atom.coordinate.z.value]
                    for atom in ordered_atoms
                ],
                dtype=float,
            )
            if pos.shape != (len(z), 3):
                raise ValueError(f"Expected MolADT coordinates with shape ({len(z)}, 3) but found {pos.shape}")
        except Exception as exc:
            failures.append(FailureRecord(dataset=dataset_name, mol_id=mol_id, stage="moladt_geometry_featurize", error=str(exc)))
            continue
        row: dict[str, Any] = {"mol_id": mol_id, "smiles": canonical, target_column: target}
        if record_index_column is not None:
            row[record_index_column] = int(getattr(record, record_index_column))
        rows.append(row)
        atomic_numbers.append(z)
        coordinates.append(pos)
        global_rows.append(descriptor_dict)
    return GeometricFeatureTable(
        rows=pd.DataFrame(rows),
        atomic_numbers=tuple(atomic_numbers),
        coordinates=tuple(coordinates),
        global_feature_names=feature_names,
        global_feature_groups=dict(MOLADT_GEOMETRY_FEATURE_GROUPS),
        global_features=np.asarray([[row[name] for name in feature_names] for row in global_rows], dtype=float) if global_rows else None,
        failures=tuple(failures),
    )


def featurize_moladt_featurized_geometry_records(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    mol_id_column: str,
    mol_column: str,
    target_column: str,
    record_index_column: str | None = None,
) -> GeometricFeatureTable:
    # Legacy richer geometry helper kept for side experiments. The current
    # public benchmark reports `sdf_geom` and `moladt_geom` instead.
    rows: list[dict[str, Any]] = []
    atomic_numbers: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    global_rows: list[dict[str, float]] = []
    failures: list[FailureRecord] = []
    feature_names = tuple(MOLADT_FEATURIZED_FEATURE_GROUPS)
    for record in dataframe.itertuples(index=False):
        mol_id = str(getattr(record, mol_id_column))
        target = float(getattr(record, target_column))
        raw_molecule = getattr(record, mol_column)
        try:
            canonical = canonical_smiles_from_mol(raw_molecule)
            moladt_record = rdkit_mol_to_moladt_record(raw_molecule)
            descriptor_dict = compute_moladt_featurized_descriptors(moladt_record.molecule)
            ordered_atoms = [moladt_record.molecule.atoms[atom_id] for atom_id in sorted(moladt_record.molecule.atoms)]
            z = np.asarray([atom.attributes.atomic_number for atom in ordered_atoms], dtype=np.int64)
            pos = np.asarray(
                [
                    [atom.coordinate.x.value, atom.coordinate.y.value, atom.coordinate.z.value]
                    for atom in ordered_atoms
                ],
                dtype=float,
            )
            if pos.shape != (len(z), 3):
                raise ValueError(f"Expected MolADT coordinates with shape ({len(z)}, 3) but found {pos.shape}")
        except Exception as exc:
            failures.append(FailureRecord(dataset=dataset_name, mol_id=mol_id, stage="moladt_featurized_geometry_featurize", error=str(exc)))
            continue
        row: dict[str, Any] = {"mol_id": mol_id, "smiles": canonical, target_column: target}
        if record_index_column is not None:
            row[record_index_column] = int(getattr(record, record_index_column))
        rows.append(row)
        atomic_numbers.append(z)
        coordinates.append(pos)
        global_rows.append(descriptor_dict)
    return GeometricFeatureTable(
        rows=pd.DataFrame(rows),
        atomic_numbers=tuple(atomic_numbers),
        coordinates=tuple(coordinates),
        global_feature_names=feature_names,
        global_feature_groups=dict(MOLADT_FEATURIZED_FEATURE_GROUPS),
        global_features=np.asarray([[row[name] for name in feature_names] for row in global_rows], dtype=float) if global_rows else None,
        failures=tuple(failures),
    )


def load_rdkit_sdf_records(sdf_path: Path) -> list["Chem.Mol | None"]:
    chem = _require_rdkit_chem()
    supplier = chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    return [molecule for molecule in supplier]


def compute_base_descriptors(molecule: "Chem.Mol") -> dict[str, float]:
    descriptors, lipinski, rd_mol_descriptors, chem = _require_rdkit_descriptors()
    reference = _sanitize_rdkit_mol(molecule)
    return {
        "molecular_weight": float(descriptors.MolWt(reference)),
        "hba": float(lipinski.NumHAcceptors(reference)),
        "hbd": float(lipinski.NumHDonors(reference)),
        "tpsa": float(rd_mol_descriptors.CalcTPSA(reference)),
        "rotatable_bond_count": float(lipinski.NumRotatableBonds(reference)),
        "ring_count": float(rd_mol_descriptors.CalcNumRings(reference)),
        "aromatic_ring_count": float(rd_mol_descriptors.CalcNumAromaticRings(reference)),
        "fraction_csp3": float(rd_mol_descriptors.CalcFractionCSP3(reference)),
        "formal_charge": float(chem.GetFormalCharge(reference)),
        "heavy_atom_count": float(reference.GetNumHeavyAtoms()),
    }


def compute_3d_descriptors(molecule: "Chem.Mol") -> dict[str, float]:
    _, _, rd_mol_descriptors, _ = _require_rdkit_descriptors()
    reference = _sanitize_rdkit_mol(molecule)
    if reference.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer coordinates")
    return {
        "radius_of_gyration": float(rd_mol_descriptors.CalcRadiusOfGyration(reference)),
        "asphericity": float(rd_mol_descriptors.CalcAsphericity(reference)),
        "eccentricity": float(rd_mol_descriptors.CalcEccentricity(reference)),
        "inertial_shape_factor": float(rd_mol_descriptors.CalcInertialShapeFactor(reference)),
        "npr1": float(rd_mol_descriptors.CalcNPR1(reference)),
        "npr2": float(rd_mol_descriptors.CalcNPR2(reference)),
        "pmi1": float(rd_mol_descriptors.CalcPMI1(reference)),
        "pmi2": float(rd_mol_descriptors.CalcPMI2(reference)),
        "pmi3": float(rd_mol_descriptors.CalcPMI3(reference)),
        **pairwise_distance_summaries(reference),
    }


def pairwise_distance_summaries(molecule: "Chem.Mol") -> dict[str, float]:
    conformer = molecule.GetConformer()
    coordinates = np.asarray(conformer.GetPositions(), dtype=float)
    if coordinates.shape[0] < 2:
        return {"distance_mean": 0.0, "distance_std": 0.0, "distance_max": 0.0}
    deltas = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    return {
        "distance_mean": float(np.mean(upper)),
        "distance_std": float(np.std(upper)),
        "distance_max": float(np.max(upper)),
    }


@contextmanager
def _suppress_rdkit_logs(*channels: str):
    rd_logger = _import_rdkit_logger()
    if rd_logger is None:
        yield
        return
    for channel in channels:
        rd_logger.DisableLog(channel)
    try:
        yield
    finally:
        for channel in reversed(channels):
            rd_logger.EnableLog(channel)


def _sanitize_rdkit_mol(molecule: "Chem.Mol") -> "Chem.Mol":
    chem = _require_rdkit_chem()
    working = chem.Mol(molecule)
    remove_hs = chem.RemoveHsParameters()
    remove_hs.removeDegreeZero = True
    remove_hs.showWarnings = False
    with _suppress_rdkit_logs("rdApp.error", "rdApp.warning"):
        chem.SanitizeMol(working)
        without_hydrogens = chem.RemoveHs(working, remove_hs)
        chem.SanitizeMol(without_hydrogens)
    return without_hydrogens


def rdkit_mol_to_moladt_record(molecule: "Chem.Mol"):
    chem = _require_rdkit_chem()
    working = chem.Mol(molecule)
    with _suppress_rdkit_logs("rdApp.error", "rdApp.warning"):
        chem.SanitizeMol(working)
    mol_block = chem.MolToMolBlock(working)
    return parse_sdf_record(f"{mol_block}\n$$$$\n")


def _import_rdkit_chem():
    try:
        from rdkit import Chem
    except ImportError:
        return None
    return Chem


def _require_rdkit_chem():
    chem = _import_rdkit_chem()
    if chem is None:
        raise RuntimeError("RDKit is required for this interop helper but is not installed")
    return chem


def _import_rdkit_logger():
    try:
        from rdkit import RDLogger
    except ImportError:
        return None
    return RDLogger


def _require_rdkit_descriptors():
    chem = _require_rdkit_chem()
    try:
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    except ImportError as exc:
        raise RuntimeError("RDKit descriptor helpers are not installed") from exc
    return Descriptors, Lipinski, rdMolDescriptors, chem


def compute_moladt_featurized_descriptors(molecule: Molecule) -> dict[str, float]:
    # Legacy richer ADT helper kept for side experiments. It adds typed
    # pair, radial, angle, torsion, and bonding-system channels on top of
    # the compact MolADT descriptor set, which means it is not the fair
    # `smiles` vs `moladt` benchmark comparison.
    descriptors = compute_moladt_descriptors(molecule).to_dict()
    descriptors.update(_typed_pair_features(molecule))
    descriptors.update(_typed_system_features(molecule))
    descriptors.update(_typed_edge_order_bucket_features(molecule))
    descriptors.update(_typed_radial_features(molecule))
    descriptors.update(_typed_angle_features(molecule))
    descriptors.update(_typed_torsion_features(molecule))
    return descriptors


def _ordered_moladt_atoms(molecule: Molecule):
    return [molecule.atoms[atom_id] for atom_id in sorted(molecule.atoms)]


def _unique_moladt_edges(molecule: Molecule):
    edges = set(molecule.local_bonds)
    for _, system in molecule.systems:
        edges.update(system.member_edges)
    return tuple(sorted(edges))


def _coordinate_vector(atom: Any) -> np.ndarray:
    return np.asarray(
        [atom.coordinate.x.value, atom.coordinate.y.value, atom.coordinate.z.value],
        dtype=float,
    )


def _moladt_coordinate_map(molecule: Molecule) -> dict[Any, np.ndarray]:
    return {
        atom_id: _coordinate_vector(atom)
        for atom_id, atom in molecule.atoms.items()
    }


def _moladt_atomic_numbers(molecule: Molecule) -> dict[Any, float]:
    return {
        atom_id: float(atom.attributes.atomic_number)
        for atom_id, atom in molecule.atoms.items()
    }


def _moladt_edge_order_map(molecule: Molecule) -> dict[Any, float]:
    return {
        edge: float(effective_order(molecule, edge))
        for edge in _unique_moladt_edges(molecule)
    }


def _moladt_edge_order(edge_orders: dict[Any, float], left: Any, right: Any) -> float:
    return float(edge_orders[mk_edge(left, right)])


def _moladt_adjacency(molecule: Molecule) -> dict[Any, tuple[Any, ...]]:
    adjacency = {atom_id: set() for atom_id in molecule.atoms}
    for edge in _unique_moladt_edges(molecule):
        adjacency[edge.a].add(edge.b)
        adjacency[edge.b].add(edge.a)
    return {
        atom_id: tuple(sorted(neighbors))
        for atom_id, neighbors in adjacency.items()
    }


def _typed_pair_features(molecule: Molecule) -> dict[str, float]:
    features = {name: 0.0 for name in (*_TYPED_PAIR_COUNT_FEATURES, *_TYPED_PAIR_INTERACTION_FEATURES)}
    ordered_atoms = _ordered_moladt_atoms(molecule)
    if len(ordered_atoms) < 2:
        return features
    coordinates = np.asarray(
        [[atom.coordinate.x.value, atom.coordinate.y.value, atom.coordinate.z.value] for atom in ordered_atoms],
        dtype=float,
    )
    atomic_numbers = np.asarray([atom.attributes.atomic_number for atom in ordered_atoms], dtype=float)
    symbols = [atom.attributes.symbol.value for atom in ordered_atoms]
    upper_i, upper_j = np.triu_indices(len(ordered_atoms), k=1)
    deltas = coordinates[upper_i] - coordinates[upper_j]
    distances = np.linalg.norm(deltas, axis=1)
    safe_distances = np.maximum(distances, 1e-6)
    interaction_values = (atomic_numbers[upper_i] * atomic_numbers[upper_j]) / safe_distances
    for left_index, right_index, interaction in zip(upper_i, upper_j, interaction_values, strict=True):
        left_symbol = symbols[int(left_index)]
        right_symbol = symbols[int(right_index)]
        if _PAIR_SYMBOL_ORDER[left_symbol] > _PAIR_SYMBOL_ORDER[right_symbol]:
            left_symbol, right_symbol = right_symbol, left_symbol
        count_name = _pair_feature_name("pair_count", left_symbol, right_symbol)
        interaction_name = _pair_feature_name("pair_interaction", left_symbol, right_symbol)
        features[count_name] += 1.0
        features[interaction_name] += float(interaction)
    return features


def _typed_system_features(molecule: Molecule) -> dict[str, float]:
    atom_sizes = [float(len(system.member_atoms)) for _, system in molecule.systems]
    edge_sizes = [float(len(system.member_edges)) for _, system in molecule.systems]
    shared_electrons = [float(system.shared_electrons.value) for _, system in molecule.systems]
    return {
        "system_member_atoms_mean": float(np.mean(atom_sizes)) if atom_sizes else 0.0,
        "system_member_atoms_max": float(np.max(atom_sizes)) if atom_sizes else 0.0,
        "system_member_edges_mean": float(np.mean(edge_sizes)) if edge_sizes else 0.0,
        "system_member_edges_max": float(np.max(edge_sizes)) if edge_sizes else 0.0,
        "system_shared_electrons_sum": float(np.sum(shared_electrons)) if shared_electrons else 0.0,
        "system_shared_electrons_mean": float(np.mean(shared_electrons)) if shared_electrons else 0.0,
        "system_shared_electrons_max": float(np.max(shared_electrons)) if shared_electrons else 0.0,
    }


def _typed_edge_order_bucket_features(molecule: Molecule) -> dict[str, float]:
    features = {name: 0.0 for name in _TYPED_EDGE_BUCKET_FEATURES}
    for edge in _unique_moladt_edges(molecule):
        order = float(effective_order(molecule, edge))
        if order <= 1.10:
            features["edge_order_sigma_like_count"] += 1.0
        elif order < 1.80:
            features["edge_order_delocalized_count"] += 1.0
        elif order < 2.40:
            features["edge_order_double_like_count"] += 1.0
        else:
            features["edge_order_triple_plus_count"] += 1.0
    return features


def _typed_radial_features(molecule: Molecule) -> dict[str, float]:
    features = {name: 0.0 for name in _TYPED_RADIAL_FEATURES}
    ordered_atoms = _ordered_moladt_atoms(molecule)
    if len(ordered_atoms) < 2:
        return features
    coordinates = np.asarray(
        [[atom.coordinate.x.value, atom.coordinate.y.value, atom.coordinate.z.value] for atom in ordered_atoms],
        dtype=float,
    )
    atomic_numbers = np.asarray([atom.attributes.atomic_number for atom in ordered_atoms], dtype=float)
    upper_i, upper_j = np.triu_indices(len(ordered_atoms), k=1)
    distances = np.linalg.norm(coordinates[upper_i] - coordinates[upper_j], axis=1)
    pair_weights = atomic_numbers[upper_i] * atomic_numbers[upper_j]
    for center in _PAIR_RADIAL_CENTERS:
        channel = np.exp(-((distances - center) ** 2) / (2.0 * (_PAIR_RADIAL_SIGMA ** 2)))
        features[_radial_feature_name("aprdf_all", center)] = float(np.sum(pair_weights * channel))
    system_edges = {edge for _, system in molecule.systems for edge in system.member_edges}
    edge_distances: list[float] = []
    edge_weights: list[float] = []
    system_edge_distances: list[float] = []
    system_edge_weights: list[float] = []
    for edge in _unique_moladt_edges(molecule):
        atom_a = molecule.atoms[edge.a]
        atom_b = molecule.atoms[edge.b]
        distance = float(
            np.linalg.norm(
                np.asarray(
                    [
                        atom_a.coordinate.x.value - atom_b.coordinate.x.value,
                        atom_a.coordinate.y.value - atom_b.coordinate.y.value,
                        atom_a.coordinate.z.value - atom_b.coordinate.z.value,
                    ],
                    dtype=float,
                )
            )
        )
        weight = float(effective_order(molecule, edge))
        edge_distances.append(distance)
        edge_weights.append(weight)
        if edge in system_edges:
            system_edge_distances.append(distance)
            system_edge_weights.append(weight)
    edge_distances_arr = np.asarray(edge_distances, dtype=float)
    edge_weights_arr = np.asarray(edge_weights, dtype=float)
    system_edge_distances_arr = np.asarray(system_edge_distances, dtype=float)
    system_edge_weights_arr = np.asarray(system_edge_weights, dtype=float)
    for center in _PAIR_RADIAL_CENTERS:
        if edge_distances_arr.size:
            edge_channel = np.exp(-((edge_distances_arr - center) ** 2) / (2.0 * (_PAIR_RADIAL_SIGMA ** 2)))
            features[_radial_feature_name("aprdf_edge_order", center)] = float(np.sum(edge_weights_arr * edge_channel))
        if system_edge_distances_arr.size:
            system_edge_channel = np.exp(-((system_edge_distances_arr - center) ** 2) / (2.0 * (_PAIR_RADIAL_SIGMA ** 2)))
            features[_radial_feature_name("aprdf_system_edge", center)] = float(np.sum(system_edge_weights_arr * system_edge_channel))
    return features


def _typed_angle_features(molecule: Molecule) -> dict[str, float]:
    features = {name: 0.0 for name in _TYPED_ANGLE_FEATURES}
    adjacency = _moladt_adjacency(molecule)
    coordinates = _moladt_coordinate_map(molecule)
    atomic_numbers = _moladt_atomic_numbers(molecule)
    edge_orders = _moladt_edge_order_map(molecule)
    for center_id, neighbors in adjacency.items():
        if len(neighbors) < 2:
            continue
        center = coordinates[center_id]
        for left_index, left_id in enumerate(neighbors):
            for right_id in neighbors[left_index + 1 :]:
                left_vector = coordinates[left_id] - center
                right_vector = coordinates[right_id] - center
                left_distance = float(np.linalg.norm(left_vector))
                right_distance = float(np.linalg.norm(right_vector))
                if left_distance <= 1e-6 or right_distance <= 1e-6:
                    continue
                cosine = float(np.dot(left_vector, right_vector) / (left_distance * right_distance))
                angle_degrees = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
                distance_weight = float(
                    (atomic_numbers[left_id] * atomic_numbers[right_id]) / max(left_distance * right_distance, 1e-6)
                )
                order_weight = float(
                    0.5
                    * (
                        _moladt_edge_order(edge_orders, center_id, left_id)
                        + _moladt_edge_order(edge_orders, center_id, right_id)
                    )
                )
                for radial_center in _ANGLE_CENTERS_DEGREES:
                    channel = math.exp(-((angle_degrees - radial_center) ** 2) / (2.0 * (_ANGLE_SIGMA_DEGREES ** 2)))
                    features[_angular_feature_name("bond_angle_all", radial_center)] += float(channel)
                    features[_angular_feature_name("bond_angle_distance_weighted", radial_center)] += float(
                        distance_weight * channel
                    )
                    features[_angular_feature_name("bond_angle_order_weighted", radial_center)] += float(
                        order_weight * channel
                    )
    return features


def _typed_torsion_features(molecule: Molecule) -> dict[str, float]:
    features = {name: 0.0 for name in _TYPED_DIHEDRAL_FEATURES}
    adjacency = _moladt_adjacency(molecule)
    coordinates = _moladt_coordinate_map(molecule)
    atomic_numbers = _moladt_atomic_numbers(molecule)
    edge_orders = _moladt_edge_order_map(molecule)
    for edge in _unique_moladt_edges(molecule):
        left_neighbors = [atom_id for atom_id in adjacency[edge.a] if atom_id != edge.b]
        right_neighbors = [atom_id for atom_id in adjacency[edge.b] if atom_id != edge.a]
        if not left_neighbors or not right_neighbors:
            continue
        central_order = _moladt_edge_order(edge_orders, edge.a, edge.b)
        for terminal_left in left_neighbors:
            for terminal_right in right_neighbors:
                if terminal_left == terminal_right:
                    continue
                dihedral_degrees = _absolute_dihedral_degrees(
                    coordinates[terminal_left],
                    coordinates[edge.a],
                    coordinates[edge.b],
                    coordinates[terminal_right],
                )
                left_distance = float(np.linalg.norm(coordinates[terminal_left] - coordinates[edge.a]))
                right_distance = float(np.linalg.norm(coordinates[terminal_right] - coordinates[edge.b]))
                if left_distance <= 1e-6 or right_distance <= 1e-6:
                    continue
                distance_weight = float(
                    (atomic_numbers[terminal_left] * atomic_numbers[terminal_right]) / max(left_distance * right_distance, 1e-6)
                )
                for radial_center in _DIHEDRAL_CENTERS_DEGREES:
                    channel = math.exp(
                        -((dihedral_degrees - radial_center) ** 2) / (2.0 * (_DIHEDRAL_SIGMA_DEGREES ** 2))
                    )
                    features[_angular_feature_name("torsion_all", radial_center)] += float(channel)
                    features[_angular_feature_name("torsion_distance_weighted", radial_center)] += float(
                        distance_weight * channel
                    )
                    features[_angular_feature_name("torsion_order_weighted", radial_center)] += float(
                        central_order * channel
                    )
    return features


def _absolute_dihedral_degrees(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray, point_d: np.ndarray) -> float:
    bond_ab = point_a - point_b
    bond_bc = point_c - point_b
    bond_cd = point_d - point_c
    norm_bc = float(np.linalg.norm(bond_bc))
    if norm_bc <= 1e-6:
        return 0.0
    bc_unit = bond_bc / norm_bc
    normal_left = bond_ab - np.dot(bond_ab, bc_unit) * bc_unit
    normal_right = bond_cd - np.dot(bond_cd, bc_unit) * bc_unit
    left_norm = float(np.linalg.norm(normal_left))
    right_norm = float(np.linalg.norm(normal_right))
    if left_norm <= 1e-6 or right_norm <= 1e-6:
        return 0.0
    normal_left /= left_norm
    normal_right /= right_norm
    x = float(np.dot(normal_left, normal_right))
    y = float(np.dot(np.cross(bc_unit, normal_left), normal_right))
    return abs(float(np.degrees(np.arctan2(y, x))))
