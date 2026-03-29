from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

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


@dataclass(frozen=True, slots=True)
class FeatureTable:
    rows: pd.DataFrame
    feature_names: tuple[str, ...]
    feature_groups: dict[str, str]
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


def _sanitize_rdkit_mol(molecule: Chem.Mol) -> Chem.Mol:
    working = Chem.Mol(molecule)
    Chem.SanitizeMol(working)
    without_hydrogens = Chem.RemoveHs(working)
    Chem.SanitizeMol(without_hydrogens)
    return without_hydrogens
