from __future__ import annotations

import json

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from scripts.download_data import freesolv_raw_dir, qm9_raw_dir, zinc_archive_filename, zinc_normalized_source_name, zinc_raw_dir
from scripts.features import canonicalize_smiles, compute_3d_descriptors, compute_base_descriptors, FeatureTable
from scripts.splits import deterministic_split_indices, export_standardized_splits
from scripts.stan_runner import build_stan_data, write_stan_data_json


def test_download_path_resolution_is_deterministic() -> None:
    assert freesolv_raw_dir().name == "freesolv"
    assert qm9_raw_dir().name == "qm9"
    assert zinc_raw_dir().name == "zinc"
    assert zinc_archive_filename("250K", "2D") == "zinc15_250K_2D.tar.gz"
    assert zinc_normalized_source_name("1M", "2D", ".csv") == "zinc15_1M_2D.csv"


def test_canonical_smiles_generation_is_stable() -> None:
    assert canonicalize_smiles("OCC") == "CCO"


def test_descriptor_generation_produces_expected_keys() -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(molecule, randomSeed=1)
    base = compute_base_descriptors(molecule)
    shape = compute_3d_descriptors(molecule)
    assert base["molecular_weight"] > 0.0
    assert base["heavy_atom_count"] == 3.0
    assert shape["radius_of_gyration"] > 0.0
    assert shape["distance_max"] > 0.0


def test_split_indices_are_reproducible() -> None:
    first = deterministic_split_indices(20, seed=7)
    second = deterministic_split_indices(20, seed=7)
    assert all((left == right).all() for left, right in zip(first, second, strict=True))


def test_stan_data_serialization_round_trip(tmp_path, monkeypatch) -> None:
    import scripts.splits as splits

    monkeypatch.setattr(splits, "PROCESSED_DATA_DIR", tmp_path)
    rows = pd.DataFrame(
        [
            {"mol_id": f"mol_{index}", "smiles": "CCO", "target": float(index), "x1": float(index), "x2": float(index + 1)}
            for index in range(12)
        ]
    )
    table = FeatureTable(
        rows=rows,
        feature_names=("x1", "x2"),
        feature_groups={"x1": "family_a", "x2": "family_b"},
        failures=(),
    )
    exported = export_standardized_splits(table, dataset_name="demo", representation="smiles", target_name="target", seed=3)
    json_path = write_stan_data_json(exported, student_df=4.0)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["N"] == len(exported.y_train)
    assert payload["K"] == len(exported.feature_names)
    assert len(payload["X"]) == len(exported.y_train)
    assert payload["group_id"] == [1, 2]
