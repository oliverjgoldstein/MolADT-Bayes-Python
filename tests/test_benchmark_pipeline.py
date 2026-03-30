from __future__ import annotations

import json

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from scripts.download_data import freesolv_raw_dir, qm9_raw_dir, zinc_archive_filename, zinc_normalized_source_name, zinc_raw_dir
from scripts.features import canonicalize_smiles, compute_3d_descriptors, compute_base_descriptors, FeatureTable
from scripts.process_freesolv import process_freesolv_dataset
from scripts.process_qm9 import process_qm9_dataset
from scripts.run_all import build_parser
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


def test_run_all_parser_accepts_verbose_for_benchmark() -> None:
    args = build_parser().parse_args(["benchmark", "--verbose"])
    assert args.command == "benchmark"
    assert args.verbose is True


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
    assert isinstance(payload["y_mean"], float)
    assert payload["y_scale"] > 0.0


def test_process_freesolv_creates_processed_directory_before_writing(tmp_path, monkeypatch) -> None:
    import scripts.process_freesolv as process_freesolv
    import scripts.splits as splits

    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(process_freesolv, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(splits, "PROCESSED_DATA_DIR", processed_dir)

    class FakeDownloads:
        csv_path = tmp_path / "SAMPL.csv"
        repo_extract_dir = tmp_path / "FreeSolv-master"

    monkeypatch.setattr(process_freesolv, "download_freesolv", lambda force=False: FakeDownloads())
    monkeypatch.setattr(
        process_freesolv,
        "_canonicalize_freesolv_csv",
        lambda downloads: (
            pd.DataFrame([{"mol_id": "freesolv_0001", "smiles": "O", "expt": -1.0}]),
            [],
        ),
    )
    monkeypatch.setattr(
        process_freesolv,
        "featurize_smiles_dataframe",
        lambda *args, **kwargs: FeatureTable(
            rows=pd.DataFrame(
                [
                    {
                        "mol_id": f"freesolv_{index + 1:04d}",
                        "smiles": "O",
                        "expt": float(index),
                        "x1": float(index),
                        "x2": float(index + 1),
                    }
                    for index in range(12)
                ]
            ),
            feature_names=("x1", "x2"),
            feature_groups={"x1": "group_a", "x2": "group_b"},
            failures=(),
        ),
    )

    def fake_export_standardized_splits(*args, **kwargs):
        assert processed_dir.is_dir()
        return export_standardized_splits(args[0], dataset_name="freesolv", representation="smiles", target_name="expt", seed=1)

    monkeypatch.setattr(process_freesolv, "export_standardized_splits", fake_export_standardized_splits)

    artifacts = process_freesolv_dataset(include_sdf=False)

    assert processed_dir.is_dir()
    assert artifacts.processed_csv_path.exists()


def test_process_qm9_creates_processed_directory_before_writing(tmp_path, monkeypatch) -> None:
    import scripts.process_qm9 as process_qm9
    import scripts.splits as splits

    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(process_qm9, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(splits, "PROCESSED_DATA_DIR", processed_dir)

    class FakeDownloads:
        sdf_path = tmp_path / "qm9.sdf"
        csv_path = tmp_path / "qm9.csv"

    monkeypatch.setattr(process_qm9, "download_qm9", lambda force=False: FakeDownloads())
    aligned_frame = pd.DataFrame(
        [
            {"mol_id": f"qm9_{index:06d}", "smiles": "O", "mu": float(index), "sdf_record_index": index, "rdkit_mol": object()}
            for index in range(12)
        ]
    )
    monkeypatch.setattr(process_qm9, "_build_qm9_aligned_frame", lambda *args, **kwargs: (aligned_frame, []))
    monkeypatch.setattr(
        process_qm9,
        "featurize_smiles_dataframe",
        lambda *args, **kwargs: FeatureTable(
            rows=pd.DataFrame(
                [
                    {"mol_id": f"qm9_{index:06d}", "smiles": "O", "mu": float(index), "x1": float(index), "x2": float(index + 1)}
                    for index in range(12)
                ]
            ),
            feature_names=("x1", "x2"),
            feature_groups={"x1": "group_a", "x2": "group_b"},
            failures=(),
        ),
    )
    monkeypatch.setattr(
        process_qm9,
        "featurize_sdf_records",
        lambda *args, **kwargs: FeatureTable(
            rows=pd.DataFrame(
                [
                    {"mol_id": f"qm9_{index:06d}", "mu": float(index), "sdf_record_index": index, "x1": float(index), "x2": float(index + 1)}
                    for index in range(12)
                ]
            ),
            feature_names=("x1", "x2"),
            feature_groups={"x1": "group_a", "x2": "group_b"},
            failures=(),
        ),
    )

    def fake_export_standardized_splits(*args, **kwargs):
        assert processed_dir.is_dir()
        return export_standardized_splits(args[0], dataset_name=kwargs["dataset_name"], representation=kwargs["representation"], target_name=kwargs["target_name"], seed=kwargs["seed"])

    monkeypatch.setattr(process_qm9, "export_standardized_splits", fake_export_standardized_splits)

    artifacts = process_qm9_dataset(limit=12)

    assert processed_dir.is_dir()
    assert artifacts.processed_csv_path.exists()
