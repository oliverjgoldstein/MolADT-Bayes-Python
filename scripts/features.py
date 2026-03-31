from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from moladt.inference import compute_descriptors as compute_moladt_descriptors
from moladt.io.sdf import parse_sdf_record

from .common import FailureRecord

BASE_FEATURE_GROUPS: dict[str, str] = {
    "molecular_weight": "size_2d",
    "hba": "polarity_2d",
    "hbd": "polarity_2d",
    "tpsa": "polarity_2d",
    "rotatable_bond_count": "topology_2d",
    "ring_count": "topology_2d",
    "aromatic_ring_count": "topology_2d",
    "fraction_csp3": "topology_2d",
    "formal_charge": "charge_2d",
    "heavy_atom_count": "size_2d",
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
    "surface": "adt_geometry_proxy",
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
    "radius_of_gyration": "adt_geometry",
    "distance_mean": "adt_geometry",
    "distance_std": "adt_geometry",
    "distance_max": "adt_geometry",
    "inertia_eigenvalue_min": "adt_geometry",
    "inertia_eigenvalue_mid": "adt_geometry",
    "inertia_eigenvalue_max": "adt_geometry",
}


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
    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def canonical_smiles_from_mol(molecule: Chem.Mol) -> str:
    sanitized = _sanitize_rdkit_mol(molecule)
    return Chem.MolToSmiles(sanitized, canonical=True, isomericSmiles=True)


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
            molecule = Chem.MolFromSmiles(smiles, sanitize=True)
            if molecule is None:
                raise ValueError("RDKit returned None from MolFromSmiles")
            canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            base = compute_base_descriptors(molecule)
        except Exception as exc:
            failures.append(FailureRecord(dataset=dataset_name, mol_id=mol_id, stage="smiles_featurize", error=str(exc)))
            continue
        row = {"mol_id": mol_id, "smiles": canonical, target_column: target}
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
        raw_molecule = getattr(record, mol_column)
        try:
            canonical = canonical_smiles_from_mol(raw_molecule)
            moladt_record = rdkit_mol_to_moladt_record(raw_molecule)
            features = compute_moladt_descriptors(moladt_record.molecule).to_dict()
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


def featurize_moladt_smiles_dataframe(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    mol_id_column: str,
    smiles_column: str,
    target_column: str,
) -> FeatureTable:
    from moladt.io.smiles import parse_smiles

    rows: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    for record in dataframe.itertuples(index=False):
        mol_id = str(getattr(record, mol_id_column))
        smiles = str(getattr(record, smiles_column))
        target = float(getattr(record, target_column))
        try:
            canonical = canonicalize_smiles(smiles)
            moladt_molecule = parse_smiles(canonical)
            features = compute_moladt_descriptors(moladt_molecule).to_dict()
        except Exception as exc:
            failures.append(FailureRecord(dataset=dataset_name, mol_id=mol_id, stage="moladt_smiles_featurize", error=str(exc)))
            continue
        row: dict[str, Any] = {"mol_id": mol_id, "smiles": canonical, target_column: target}
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
        raw_molecule = getattr(record, mol_column)
        try:
            molecule = Chem.Mol(raw_molecule)
            with _suppress_rdkit_logs("rdApp.error", "rdApp.warning"):
                Chem.SanitizeMol(molecule)
            if molecule.GetNumConformers() == 0:
                raise ValueError("Molecule has no conformer coordinates")
            conformer = molecule.GetConformer()
            z = np.asarray([atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.int64)
            pos = np.asarray(conformer.GetPositions(), dtype=float)
            canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
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
        raw_molecule = getattr(record, mol_column)
        try:
            canonical = canonical_smiles_from_mol(raw_molecule)
            moladt_record = rdkit_mol_to_moladt_record(raw_molecule)
            descriptor_dict = compute_moladt_descriptors(moladt_record.molecule).to_dict()
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
        global_feature_groups=dict(MOLADT_FEATURE_GROUPS),
        global_features=np.asarray([[row[name] for name in feature_names] for row in global_rows], dtype=float) if global_rows else None,
        failures=tuple(failures),
    )


def load_rdkit_sdf_records(sdf_path: Path) -> list[Chem.Mol | None]:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    return [molecule for molecule in supplier]


def compute_base_descriptors(molecule: Chem.Mol) -> dict[str, float]:
    reference = _sanitize_rdkit_mol(molecule)
    return {
        "molecular_weight": float(Descriptors.MolWt(reference)),
        "hba": float(Lipinski.NumHAcceptors(reference)),
        "hbd": float(Lipinski.NumHDonors(reference)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(reference)),
        "rotatable_bond_count": float(Lipinski.NumRotatableBonds(reference)),
        "ring_count": float(rdMolDescriptors.CalcNumRings(reference)),
        "aromatic_ring_count": float(rdMolDescriptors.CalcNumAromaticRings(reference)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(reference)),
        "formal_charge": float(Chem.GetFormalCharge(reference)),
        "heavy_atom_count": float(reference.GetNumHeavyAtoms()),
    }


def compute_3d_descriptors(molecule: Chem.Mol) -> dict[str, float]:
    reference = _sanitize_rdkit_mol(molecule)
    if reference.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer coordinates")
    return {
        "radius_of_gyration": float(rdMolDescriptors.CalcRadiusOfGyration(reference)),
        "asphericity": float(rdMolDescriptors.CalcAsphericity(reference)),
        "eccentricity": float(rdMolDescriptors.CalcEccentricity(reference)),
        "inertial_shape_factor": float(rdMolDescriptors.CalcInertialShapeFactor(reference)),
        "npr1": float(rdMolDescriptors.CalcNPR1(reference)),
        "npr2": float(rdMolDescriptors.CalcNPR2(reference)),
        "pmi1": float(rdMolDescriptors.CalcPMI1(reference)),
        "pmi2": float(rdMolDescriptors.CalcPMI2(reference)),
        "pmi3": float(rdMolDescriptors.CalcPMI3(reference)),
        **pairwise_distance_summaries(reference),
    }


def pairwise_distance_summaries(molecule: Chem.Mol) -> dict[str, float]:
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
    for channel in channels:
        RDLogger.DisableLog(channel)
    try:
        yield
    finally:
        for channel in reversed(channels):
            RDLogger.EnableLog(channel)


def _sanitize_rdkit_mol(molecule: Chem.Mol) -> Chem.Mol:
    working = Chem.Mol(molecule)
    remove_hs = Chem.RemoveHsParameters()
    remove_hs.removeDegreeZero = True
    remove_hs.showWarnings = False
    with _suppress_rdkit_logs("rdApp.error", "rdApp.warning"):
        Chem.SanitizeMol(working)
        without_hydrogens = Chem.RemoveHs(working, remove_hs)
        Chem.SanitizeMol(without_hydrogens)
    return without_hydrogens


def rdkit_mol_to_moladt_record(molecule: Chem.Mol):
    working = Chem.Mol(molecule)
    with _suppress_rdkit_logs("rdApp.error", "rdApp.warning"):
        Chem.SanitizeMol(working)
    mol_block = Chem.MolToMolBlock(working)
    return parse_sdf_record(f"{mol_block}\n$$$$\n")
