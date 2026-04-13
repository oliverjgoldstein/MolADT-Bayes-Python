from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from scripts.benchmark_zinc import (
    _measure_smiles_csv_string_parse,
    _measure_moladt_library_parse,
    _measure_smiles_library_parse,
    _moladt_parse_render_from_rdkit_mol,
    _prepare_timing_library,
    _read_timing_library_manifest,
)
from scripts.download_data import (
    download_freesolv,
    download_qm9,
    download_zinc,
    freesolv_raw_dir,
    qm9_raw_dir,
    zinc_archive_filename,
    zinc_normalized_source_name,
    zinc_raw_dir,
)
from scripts.features import (
    FeatureTable,
    MOLADT_FEATURE_GROUPS,
    canonicalize_smiles,
    compute_3d_descriptors,
    compute_base_descriptors,
    featurize_moladt_records,
    featurize_moladt_featurized_records,
    featurize_moladt_smiles_dataframe,
)
from scripts.process_freesolv import FreeSolvArtifacts, process_freesolv_dataset
from scripts.process_qm9 import process_qm9_dataset
from scripts.run_all import _stan_methods_for_artifacts, _stan_models_for_artifacts, build_parser
from scripts.splits import ExportedDataset, deterministic_split_indices, deterministic_split_partition, export_standardized_splits
from scripts.stan_runner import build_stan_data, write_stan_data_json


def test_download_path_resolution_is_deterministic() -> None:
    assert freesolv_raw_dir().name == "freesolv"
    assert qm9_raw_dir().name == "qm9"
    assert zinc_raw_dir().name == "zinc"
    assert zinc_archive_filename("250K", "2D") == "zinc15_250K_2D.tar.gz"
    assert zinc_normalized_source_name("1M", "2D", ".csv") == "zinc15_1M_2D.csv"


def test_download_freesolv_prefers_vendored_snapshot(tmp_path, monkeypatch) -> None:
    import scripts.download_data as download_data

    raw_dir = tmp_path / "raw"
    freesolv_dir = raw_dir / "freesolv"
    sdf_dir = freesolv_dir / "sdffiles"
    sdf_dir.mkdir(parents=True)
    (freesolv_dir / "SAMPL.csv").write_text("smiles,expt\nO,-1.0\n", encoding="utf-8")
    (sdf_dir / "demo.sdf").write_text("demo\n", encoding="utf-8")
    monkeypatch.setattr(download_data, "RAW_DATA_DIR", raw_dir)

    def fail_download(*args, **kwargs):
        raise AssertionError("download_freesolv should not hit the network when the vendored snapshot exists")

    monkeypatch.setattr(download_data, "download_file", fail_download)

    downloads = download_freesolv()

    assert downloads.csv_path == freesolv_dir / "SAMPL.csv"
    assert downloads.repo_archive_path is None
    assert downloads.repo_extract_dir == freesolv_dir


def test_download_qm9_prefers_vendored_snapshot(tmp_path, monkeypatch) -> None:
    import scripts.download_data as download_data

    raw_dir = tmp_path / "raw"
    qm9_dir = raw_dir / "qm9"
    qm9_dir.mkdir(parents=True)
    (qm9_dir / "qm9.sdf").write_text("demo\n", encoding="utf-8")
    (qm9_dir / "qm9.sdf.csv").write_text("mol_id,mu\nmol_1,0.1\n", encoding="utf-8")
    monkeypatch.setattr(download_data, "RAW_DATA_DIR", raw_dir)

    def fail_download(*args, **kwargs):
        raise AssertionError("download_qm9 should not hit the network when the vendored snapshot exists")

    monkeypatch.setattr(download_data, "download_first", fail_download)
    monkeypatch.setattr(download_data, "download_file", fail_download)

    downloads = download_qm9()

    assert downloads.sdf_path == qm9_dir / "qm9.sdf"
    assert downloads.csv_path == qm9_dir / "qm9.sdf.csv"
    assert downloads.archive_path is None
    assert downloads.extract_dir == qm9_dir


def test_download_qm9_prefers_v3000_file_when_archive_contains_one(tmp_path, monkeypatch) -> None:
    import scripts.download_data as download_data

    raw_dir = tmp_path / "raw"
    qm9_dir = raw_dir / "qm9"
    extract_dir = qm9_dir / "extracted"
    extract_dir.mkdir(parents=True)
    archive_path = qm9_dir / "qm9.tar.gz"
    archive_path.write_text("archive", encoding="utf-8")
    (extract_dir / "gdb9.sdf").write_text("v2000\n", encoding="utf-8")
    (extract_dir / "gdb9_v3000.sdf").write_text("v3000\n", encoding="utf-8")
    (extract_dir / "qm9.csv").write_text("mol_id,mu\nmol_1,0.1\n", encoding="utf-8")
    monkeypatch.setattr(download_data, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(download_data, "download_first", lambda *args, **kwargs: archive_path)
    monkeypatch.setattr(download_data, "extract_archive", lambda *args, **kwargs: extract_dir)
    monkeypatch.setattr(download_data, "copy_if_needed", lambda source, destination, force=False: source)

    downloads = download_qm9()

    assert downloads.sdf_path == extract_dir / "gdb9_v3000.sdf"


def test_download_qm9_refreshes_truncated_cached_copy_from_extracted_source(tmp_path, monkeypatch) -> None:
    import scripts.download_data as download_data

    raw_dir = tmp_path / "raw"
    qm9_dir = raw_dir / "qm9"
    extract_dir = qm9_dir / "extracted"
    extract_dir.mkdir(parents=True)
    cached_sdf = qm9_dir / "qm9.sdf"
    cached_csv = qm9_dir / "qm9.sdf.csv"
    source_sdf = extract_dir / "qm9.sdf"
    source_csv = extract_dir / "qm9.csv"
    cached_sdf.write_text("partial\n", encoding="utf-8")
    cached_csv.write_text("mol_id,mu\nmol_1,0.1\n", encoding="utf-8")
    source_sdf.write_text("complete sdf payload\n", encoding="utf-8")
    source_csv.write_text("mol_id,mu\nmol_1,0.1\nmol_2,0.2\n", encoding="utf-8")
    monkeypatch.setattr(download_data, "RAW_DATA_DIR", raw_dir)

    def fail_download(*args, **kwargs):
        raise AssertionError("download_qm9 should repair from the extracted source without hitting the network")

    monkeypatch.setattr(download_data, "download_first", fail_download)
    monkeypatch.setattr(download_data, "download_file", fail_download)

    downloads = download_qm9()

    assert downloads.sdf_path == cached_sdf
    assert downloads.csv_path == cached_csv
    assert cached_sdf.read_text(encoding="utf-8") == source_sdf.read_text(encoding="utf-8")
    assert cached_csv.read_text(encoding="utf-8") == source_csv.read_text(encoding="utf-8")
    assert downloads.extract_dir == extract_dir


def test_download_zinc_prefers_vendored_snapshot(tmp_path, monkeypatch) -> None:
    import scripts.download_data as download_data

    raw_dir = tmp_path / "raw"
    zinc_dir = raw_dir / "zinc"
    zinc_dir.mkdir(parents=True)
    zinc_csv = zinc_dir / "zinc15_250K_2D.csv"
    zinc_csv.write_text("smiles\nCCO\n", encoding="utf-8")
    monkeypatch.setattr(download_data, "RAW_DATA_DIR", raw_dir)

    def fail_download(*args, **kwargs):
        raise AssertionError("download_zinc should not hit the network when the vendored snapshot exists")

    monkeypatch.setattr(download_data, "download_file", fail_download)

    downloads = download_zinc()

    assert downloads.source_path == zinc_csv
    assert downloads.archive_path is None
    assert downloads.extract_dir == zinc_dir


def test_canonical_smiles_generation_is_stable() -> None:
    assert canonicalize_smiles("OCC") == canonicalize_smiles("CCO")


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


def test_exact_split_partition_preserves_unused_rows() -> None:
    partition = deterministic_split_partition(25, seed=3, train_size=10, valid_size=5, test_size=5, scheme="paper:10/5/5")

    assert len(partition.train_indices) == 10
    assert len(partition.valid_indices) == 5
    assert len(partition.test_indices) == 5
    assert len(partition.unused_indices) == 5
    assert partition.scheme == "paper:10/5/5"


def test_run_all_parser_accepts_verbose_for_benchmark() -> None:
    args = build_parser().parse_args(["benchmark", "--verbose"])
    assert args.command == "benchmark"
    assert args.verbose is True


def test_run_all_defaults_freesolv_to_single_best_model() -> None:
    rows = pd.DataFrame([{"mol_id": "mol_1", "smiles": "CCO", "target": 1.0}])
    bundle = ExportedDataset(
        dataset_name="freesolv",
        representation="moladt",
        target_name="target",
        split_scheme="fractional:0.800/0.100/0.100",
        source_row_count=3,
        used_row_count=3,
        feature_names=("x1",),
        feature_groups={"x1": "group_a"},
        group_names=("group_a",),
        group_ids=(1,),
        rows=rows,
        X_train=np.asarray([[0.0]]),
        X_valid=np.asarray([[0.0]]),
        X_test=np.asarray([[0.0]]),
        y_train=np.asarray([1.0]),
        y_valid=np.asarray([1.0]),
        y_test=np.asarray([1.0]),
        mol_ids_train=("mol_1",),
        mol_ids_valid=("mol_1",),
        mol_ids_test=("mol_1",),
        metadata_path=Path("demo_metadata.json"),
        feature_csv_path=Path("demo_features.csv"),
    )
    artifacts = FreeSolvArtifacts(
        processed_csv_path=Path("freesolv_processed.csv"),
        moladt_index_path=None,
        tabular_exports={"moladt": bundle},
        geometric_exports={},
        smiles_export=bundle,
        moladt_export=bundle,
        moladt_featurized_export=None,
        failure_csv_paths=(),
    )

    args = build_parser().parse_args(["freesolv"])

    assert _stan_models_for_artifacts(artifacts, args) == ("bayes_gp_rbf_screened",)


def test_run_all_defaults_qm9_to_single_best_model() -> None:
    rows = pd.DataFrame([{"mol_id": "mol_1", "smiles": "CCO", "target": 1.0}])
    bundle = ExportedDataset(
        dataset_name="qm9",
        representation="moladt_featurized",
        target_name="target",
        split_scheme="paper:110462/10000/10000",
        source_row_count=130462,
        used_row_count=130462,
        feature_names=("x1",),
        feature_groups={"x1": "group_a"},
        group_names=("group_a",),
        group_ids=(1,),
        rows=rows,
        X_train=np.asarray([[0.0]]),
        X_valid=np.asarray([[0.0]]),
        X_test=np.asarray([[0.0]]),
        y_train=np.asarray([1.0]),
        y_valid=np.asarray([1.0]),
        y_test=np.asarray([1.0]),
        mol_ids_train=("mol_1",),
        mol_ids_valid=("mol_1",),
        mol_ids_test=("mol_1",),
        metadata_path=Path("demo_metadata.json"),
        feature_csv_path=Path("demo_features.csv"),
    )
    from scripts.process_qm9 import QM9Artifacts

    qm9_artifacts = QM9Artifacts(
        processed_csv_path=Path("qm9_processed.csv"),
        moladt_index_path=Path("qm9_index.csv"),
        tabular_exports={"moladt_featurized": bundle},
        geometric_exports={},
        smiles_export=bundle,
        moladt_export=bundle,
        moladt_featurized_export=bundle,
        failure_csv_paths=(),
    )

    args = build_parser().parse_args(["qm9"])

    assert _stan_models_for_artifacts(qm9_artifacts, args) == ("bayes_linear_student_t",)


def test_run_all_defaults_freesolv_to_single_best_algorithm() -> None:
    rows = pd.DataFrame([{"mol_id": "mol_1", "smiles": "CCO", "target": 1.0}])
    bundle = ExportedDataset(
        dataset_name="freesolv",
        representation="moladt",
        target_name="target",
        split_scheme="fractional:0.800/0.100/0.100",
        source_row_count=3,
        used_row_count=3,
        feature_names=("x1",),
        feature_groups={"x1": "group_a"},
        group_names=("group_a",),
        group_ids=(1,),
        rows=rows,
        X_train=np.asarray([[0.0]]),
        X_valid=np.asarray([[0.0]]),
        X_test=np.asarray([[0.0]]),
        y_train=np.asarray([1.0]),
        y_valid=np.asarray([1.0]),
        y_test=np.asarray([1.0]),
        mol_ids_train=("mol_1",),
        mol_ids_valid=("mol_1",),
        mol_ids_test=("mol_1",),
        metadata_path=Path("demo_metadata.json"),
        feature_csv_path=Path("demo_features.csv"),
    )
    artifacts = FreeSolvArtifacts(
        processed_csv_path=Path("freesolv_processed.csv"),
        moladt_index_path=None,
        tabular_exports={"moladt": bundle},
        geometric_exports={},
        smiles_export=bundle,
        moladt_export=bundle,
        moladt_featurized_export=None,
        failure_csv_paths=(),
    )

    args = build_parser().parse_args(["freesolv"])

    assert _stan_methods_for_artifacts(artifacts, args) == ("laplace",)


def test_run_all_defaults_qm9_to_single_best_algorithm() -> None:
    rows = pd.DataFrame([{"mol_id": "mol_1", "smiles": "CCO", "target": 1.0}])
    bundle = ExportedDataset(
        dataset_name="qm9",
        representation="moladt_featurized",
        target_name="target",
        split_scheme="paper:110462/10000/10000",
        source_row_count=130462,
        used_row_count=130462,
        feature_names=("x1",),
        feature_groups={"x1": "group_a"},
        group_names=("group_a",),
        group_ids=(1,),
        rows=rows,
        X_train=np.asarray([[0.0]]),
        X_valid=np.asarray([[0.0]]),
        X_test=np.asarray([[0.0]]),
        y_train=np.asarray([1.0]),
        y_valid=np.asarray([1.0]),
        y_test=np.asarray([1.0]),
        mol_ids_train=("mol_1",),
        mol_ids_valid=("mol_1",),
        mol_ids_test=("mol_1",),
        metadata_path=Path("demo_metadata.json"),
        feature_csv_path=Path("demo_features.csv"),
    )
    from scripts.process_qm9 import QM9Artifacts

    qm9_artifacts = QM9Artifacts(
        processed_csv_path=Path("qm9_processed.csv"),
        moladt_index_path=Path("qm9_index.csv"),
        tabular_exports={"moladt_featurized": bundle},
        geometric_exports={},
        smiles_export=bundle,
        moladt_export=bundle,
        moladt_featurized_export=bundle,
        failure_csv_paths=(),
    )

    args = build_parser().parse_args(["qm9"])

    assert _stan_methods_for_artifacts(qm9_artifacts, args) == ("optimize",)


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


def test_featurize_moladt_records_uses_adt_descriptors() -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(molecule, randomSeed=1)
    frame = pd.DataFrame([{"mol_id": "mol_1", "mu": 1.25, "sdf_record_index": 0, "rdkit_mol": molecule}])

    table = featurize_moladt_records(
        frame,
        dataset_name="demo_moladt",
        mol_id_column="mol_id",
        mol_column="rdkit_mol",
        target_column="mu",
        record_index_column="sdf_record_index",
    )

    assert not table.rows.empty
    assert table.failures == ()
    assert "weight" in table.feature_names
    assert "radius_of_gyration" not in table.feature_names
    assert float(table.rows.iloc[0]["weight"]) > 0.0


def test_featurize_moladt_featurized_records_adds_pair_angle_and_torsion_features() -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(molecule, randomSeed=1)
    frame = pd.DataFrame([{"mol_id": "mol_1", "mu": 1.25, "sdf_record_index": 0, "rdkit_mol": molecule}])

    table = featurize_moladt_featurized_records(
        frame,
        dataset_name="demo_moladt_featurized",
        mol_id_column="mol_id",
        mol_column="rdkit_mol",
        target_column="mu",
        record_index_column="sdf_record_index",
    )

    assert not table.rows.empty
    assert table.failures == ()
    assert "pair_count_c_o" in table.feature_names
    assert "pair_interaction_c_o" in table.feature_names
    assert "aprdf_all_1p5a" in table.feature_names
    assert "bond_angle_all_120d" in table.feature_names
    assert "torsion_all_180d" in table.feature_names
    assert "system_shared_electrons_sum" in table.feature_names
    assert float(table.rows.iloc[0]["pair_count_c_o"]) >= 1.0
    assert float(table.rows.iloc[0]["bond_angle_all_120d"]) > 0.0
    torsion_columns = [name for name in table.feature_names if name.startswith("torsion_all_")]
    assert sum(float(table.rows.iloc[0][name]) for name in torsion_columns) > 0.0


def test_moladt_parse_render_supports_silicon_molecules() -> None:
    molecule = Chem.MolFromSmiles("C[Si](C)(C)C", sanitize=True)

    rendered = _moladt_parse_render_from_rdkit_mol(molecule)

    assert "Si" in rendered


def test_prepare_timing_library_creates_matched_local_corpus(tmp_path, monkeypatch) -> None:
    import scripts.benchmark_zinc as benchmark_zinc

    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(benchmark_zinc, "PROCESSED_DATA_DIR", processed_dir)
    molecules = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]

    library, stage = _prepare_timing_library(
        molecules=molecules,
        dataset_size="demo",
        dataset_dimension="2D",
        limit=2,
        source_path=tmp_path / "zinc_demo.smi",
        force=True,
    )

    manifest = _read_timing_library_manifest(library)

    assert library.smiles_path.exists()
    assert library.manifest_path.exists()
    assert len(manifest) == 2
    assert stage.stage == "timing_library_prepare"
    assert stage.success_count == 2
    assert (library.library_root / manifest.iloc[0]["moladt_relative_path"]).exists()


def test_timing_library_parse_stages_succeed_on_matched_entries(tmp_path, monkeypatch) -> None:
    import scripts.benchmark_zinc as benchmark_zinc

    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(benchmark_zinc, "PROCESSED_DATA_DIR", processed_dir)
    molecules = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCN")]

    library, _ = _prepare_timing_library(
        molecules=molecules,
        dataset_size="demo",
        dataset_dimension="2D",
        limit=2,
        source_path=tmp_path / "zinc_demo.smi",
        force=True,
    )
    manifest = _read_timing_library_manifest(library)

    csv_string_items, csv_string_stage = _measure_smiles_csv_string_parse(
        library,
        manifest=manifest,
        dataset_size="demo",
        dataset_dimension="2D",
    )
    smiles_items, smiles_stage = _measure_smiles_library_parse(
        library,
        manifest=manifest,
        dataset_size="demo",
        dataset_dimension="2D",
    )
    moladt_items, moladt_stage = _measure_moladt_library_parse(
        library,
        manifest=manifest,
        dataset_size="demo",
        dataset_dimension="2D",
    )

    assert csv_string_stage.failure_count == 0
    assert smiles_stage.failure_count == 0
    assert moladt_stage.failure_count == 0
    assert len(csv_string_items) == len(manifest)
    assert len(smiles_items) == len(manifest)
    assert len(moladt_items) == len(manifest)
    assert all(item.success for item in csv_string_items)
    assert all(item.success for item in smiles_items)
    assert all(item.success for item in moladt_items)


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
        "_load_freesolv_sdf_dataset",
        lambda downloads: (
            pd.DataFrame([{"mol_id": "freesolv_0001", "smiles": "O", "expt": -1.0}]),
            [],
            1,
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

    artifacts = process_freesolv_dataset(include_moladt=False)

    assert processed_dir.is_dir()
    assert artifacts.processed_csv_path.exists()


def test_process_freesolv_exports_sdf_backed_moladt_featurized_bundle(tmp_path, monkeypatch) -> None:
    import scripts.process_freesolv as process_freesolv
    import scripts.splits as splits

    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(process_freesolv, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(splits, "PROCESSED_DATA_DIR", processed_dir)

    class FakeDownloads:
        csv_path = tmp_path / "SAMPL.csv"
        repo_extract_dir = tmp_path / "FreeSolv-master"

    fake_frame = pd.DataFrame(
        [
            {
                "mol_id": f"mobley_{index:04d}",
                "smiles": "O",
                "smiles_canonical": "O",
                "expt": float(index),
                "sdf_relpath": f"sdffiles/mobley_{index:04d}.sdf",
                "sdf_record_index": 0,
                "moladt_molecule": object(),
            }
            for index in range(12)
        ]
    )

    fake_table = FeatureTable(
        rows=pd.DataFrame(
            [
                {"mol_id": f"mobley_{index:04d}", "smiles": "O", "expt": float(index), "x1": float(index), "x2": float(index + 1)}
                for index in range(12)
            ]
        ),
        feature_names=("x1", "x2"),
        feature_groups={"x1": "group_a", "x2": "group_b"},
        failures=(),
    )

    monkeypatch.setattr(process_freesolv, "download_freesolv", lambda force=False: FakeDownloads())
    monkeypatch.setattr(process_freesolv, "_load_freesolv_sdf_dataset", lambda downloads: (fake_frame, [], 12))
    monkeypatch.setattr(process_freesolv, "featurize_smiles_dataframe", lambda *args, **kwargs: fake_table)
    monkeypatch.setattr(process_freesolv, "featurize_moladt_smiles_dataframe", lambda *args, **kwargs: fake_table)
    monkeypatch.setattr(process_freesolv, "featurize_moladt_featurized_records", lambda *args, **kwargs: fake_table)
    monkeypatch.setattr(
        process_freesolv,
        "featurize_sdf_geometry_records",
        lambda *args, **kwargs: process_freesolv.GeometricFeatureTable(
            rows=pd.DataFrame(
                [
                    {"mol_id": f"mobley_{index:04d}", "smiles": "O", "expt": float(index), "sdf_record_index": 0}
                    for index in range(12)
                ]
            ),
            atomic_numbers=tuple(np.asarray([8], dtype=np.int64) for _ in range(12)),
            coordinates=tuple(np.asarray([[0.0, 0.0, 0.0]], dtype=float) for _ in range(12)),
            global_feature_names=("z_mean",),
            global_feature_groups={"z_mean": "geometry_atoms"},
            global_features=np.asarray([[8.0] for _ in range(12)], dtype=float),
            failures=(),
        ),
    )
    monkeypatch.setattr(
        process_freesolv,
        "featurize_moladt_geometry_records",
        lambda *args, **kwargs: process_freesolv.GeometricFeatureTable(
            rows=pd.DataFrame(
                [
                    {"mol_id": f"mobley_{index:04d}", "smiles": "O", "expt": float(index), "sdf_record_index": 0}
                    for index in range(12)
                ]
            ),
            atomic_numbers=tuple(np.asarray([8], dtype=np.int64) for _ in range(12)),
            coordinates=tuple(np.asarray([[0.0, 0.0, 0.0]], dtype=float) for _ in range(12)),
            global_feature_names=("weight",),
            global_feature_groups={"weight": "adt_composition"},
            global_features=np.asarray([[18.0] for _ in range(12)], dtype=float),
            failures=(),
        ),
    )

    artifacts = process_freesolv_dataset(include_moladt=True)

    assert "moladt_featurized" in artifacts.tabular_exports
    assert artifacts.moladt_featurized_export is not None
    assert artifacts.moladt_featurized_export.source_row_count == 12
    assert artifacts.moladt_featurized_export.used_row_count == 12


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
            {"mol_id": f"qm9_{index:06d}", "smiles": "O", "mu": float(index), "sdf_record_index": index, "moladt_molecule": object()}
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
        "featurize_moladt_smiles_dataframe",
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
    monkeypatch.setattr(
        process_qm9,
        "featurize_moladt_featurized_records",
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
    monkeypatch.setattr(
        process_qm9,
        "featurize_sdf_geometry_records",
        lambda *args, **kwargs: process_qm9.GeometricFeatureTable(
            rows=pd.DataFrame(
                [
                    {"mol_id": f"qm9_{index:06d}", "smiles": "O", "mu": float(index), "sdf_record_index": index}
                    for index in range(12)
                ]
            ),
            atomic_numbers=tuple(np.asarray([8], dtype=np.int64) for _ in range(12)),
            coordinates=tuple(np.asarray([[0.0, 0.0, 0.0]], dtype=float) for _ in range(12)),
            global_feature_names=("z_mean",),
            global_feature_groups={"z_mean": "geometry_atoms"},
            global_features=np.asarray([[8.0] for _ in range(12)], dtype=float),
            failures=(),
        ),
    )
    monkeypatch.setattr(
        process_qm9,
        "featurize_moladt_geometry_records",
        lambda *args, **kwargs: process_qm9.GeometricFeatureTable(
            rows=pd.DataFrame(
                [
                    {"mol_id": f"qm9_{index:06d}", "smiles": "O", "mu": float(index), "sdf_record_index": index}
                    for index in range(12)
                ]
            ),
            atomic_numbers=tuple(np.asarray([8], dtype=np.int64) for _ in range(12)),
            coordinates=tuple(np.asarray([[0.0, 0.0, 0.0]], dtype=float) for _ in range(12)),
            global_feature_names=("weight",),
            global_feature_groups={"weight": "adt_composition"},
            global_features=np.asarray([[18.0] for _ in range(12)], dtype=float),
            failures=(),
        ),
    )

    def fake_export_standardized_splits(*args, **kwargs):
        assert processed_dir.is_dir()
        return export_standardized_splits(args[0], dataset_name=kwargs["dataset_name"], representation=kwargs["representation"], target_name=kwargs["target_name"], seed=kwargs["seed"])

    monkeypatch.setattr(process_qm9, "export_standardized_splits", fake_export_standardized_splits)

    artifacts = process_qm9_dataset(limit=12, split_mode="subset")

    assert processed_dir.is_dir()
    assert artifacts.processed_csv_path.exists()
    assert artifacts.moladt_featurized_export is not None
    assert "moladt_featurized" in artifacts.tabular_exports


def test_process_qm9_fixed_contract_skips_legacy_smiles_exports(tmp_path, monkeypatch) -> None:
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
            {"mol_id": f"qm9_{index:06d}", "smiles": "O", "mu": float(index), "sdf_record_index": index, "moladt_molecule": object()}
            for index in range(12)
        ]
    )
    monkeypatch.setattr(process_qm9, "_build_qm9_aligned_frame", lambda *args, **kwargs: (aligned_frame, []))

    def should_not_run(*args, **kwargs):
        raise AssertionError("legacy SMILES featurizers should not run for the fixed QM9 benchmark contract")

    monkeypatch.setattr(process_qm9, "featurize_smiles_dataframe", should_not_run)
    monkeypatch.setattr(process_qm9, "featurize_moladt_smiles_dataframe", should_not_run)
    monkeypatch.setattr(
        process_qm9,
        "featurize_moladt_featurized_records",
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
    monkeypatch.setattr(
        process_qm9,
        "featurize_sdf_geometry_records",
        lambda *args, **kwargs: process_qm9.GeometricFeatureTable(
            rows=pd.DataFrame(
                [
                    {"mol_id": f"qm9_{index:06d}", "smiles": "O", "mu": float(index), "sdf_record_index": index}
                    for index in range(12)
                ]
            ),
            atomic_numbers=tuple(np.asarray([8], dtype=np.int64) for _ in range(12)),
            coordinates=tuple(np.asarray([[0.0, 0.0, 0.0]], dtype=float) for _ in range(12)),
            global_feature_names=("z_mean",),
            global_feature_groups={"z_mean": "geometry_atoms"},
            global_features=np.asarray([[8.0] for _ in range(12)], dtype=float),
            failures=(),
        ),
    )
    monkeypatch.setattr(
        process_qm9,
        "featurize_moladt_geometry_records",
        lambda *args, **kwargs: process_qm9.GeometricFeatureTable(
            rows=pd.DataFrame(
                [
                    {"mol_id": f"qm9_{index:06d}", "smiles": "O", "mu": float(index), "sdf_record_index": index}
                    for index in range(12)
                ]
            ),
            atomic_numbers=tuple(np.asarray([8], dtype=np.int64) for _ in range(12)),
            coordinates=tuple(np.asarray([[0.0, 0.0, 0.0]], dtype=float) for _ in range(12)),
            global_feature_names=("weight",),
            global_feature_groups={"weight": "adt_composition"},
            global_features=np.asarray([[18.0] for _ in range(12)], dtype=float),
            failures=(),
        ),
    )

    artifacts = process_qm9.process_qm9_dataset(limit=12, split_mode="subset", include_legacy_tabular=False)

    assert set(artifacts.tabular_exports) == {"moladt_featurized"}
    assert artifacts.moladt_featurized_export is not None


def test_load_freesolv_sdf_dataset_falls_back_when_database_json_is_missing(tmp_path, monkeypatch) -> None:
    import scripts.process_freesolv as process_freesolv

    sdf_dir = tmp_path / "sdffiles"
    sdf_dir.mkdir(parents=True)
    (sdf_dir / "mobley_1.sdf").write_text("", encoding="utf-8")
    (sdf_dir / "mobley_2.sdf").write_text("", encoding="utf-8")

    class FakeDownloads:
        repo_extract_dir = tmp_path
        csv_path = tmp_path / "SAMPL.csv"

    aligned = pd.DataFrame(
        [
            {
                "mol_id": "freesolv_0001",
                "smiles": "O",
                "smiles_canonical": "O",
                "iupac": "water",
                "expt": -6.0,
                "sdf_relpath": "sdffiles/mobley_1.sdf",
                "sdf_record_index": 0,
                "moladt_molecule": object(),
            }
        ]
    )
    monkeypatch.setattr(process_freesolv, "_find_freesolv_database_json", lambda downloads: None)
    monkeypatch.setattr(process_freesolv, "_find_freesolv_sdf_dir", lambda downloads: sdf_dir)
    monkeypatch.setattr(process_freesolv, "_canonicalize_freesolv_csv", lambda downloads: (pd.DataFrame([{"mol_id": "freesolv_0001", "smiles": "O", "smiles_canonical": "O", "iupac": "water", "expt": -6.0}]), []))
    monkeypatch.setattr(process_freesolv, "_align_freesolv_sdf", lambda downloads, processed_frame: (aligned, []))

    frame, failures, source_sdf_count = process_freesolv._load_freesolv_sdf_dataset(FakeDownloads())

    assert len(frame) == 1
    assert failures == []
    assert source_sdf_count == 2


def test_find_freesolv_database_json_recurses_into_nested_snapshot(tmp_path) -> None:
    from scripts.download_data import FreeSolvDownloads
    from scripts.process_freesolv import _find_freesolv_database_json

    nested = tmp_path / "FreeSolv-master" / "FreeSolv-master" / "metadata"
    nested.mkdir(parents=True)
    database_path = nested / "database.json"
    database_path.write_text("{}", encoding="utf-8")

    downloads = FreeSolvDownloads(csv_path=tmp_path / "SAMPL.csv", repo_archive_path=None, repo_extract_dir=tmp_path)

    assert _find_freesolv_database_json(downloads) == database_path


def test_find_freesolv_sdf_dir_recurses_into_nested_snapshot(tmp_path) -> None:
    from scripts.download_data import FreeSolvDownloads
    from scripts.process_freesolv import _find_freesolv_sdf_dir

    sdf_dir = tmp_path / "FreeSolv-master" / "FreeSolv-master" / "sdffiles" / "sdffiles"
    sdf_dir.mkdir(parents=True)
    (sdf_dir / "mobley_1.sdf").write_text("", encoding="utf-8")

    downloads = FreeSolvDownloads(csv_path=tmp_path / "SAMPL.csv", repo_archive_path=None, repo_extract_dir=tmp_path)

    assert _find_freesolv_sdf_dir(downloads) == sdf_dir


def test_freesolv_split_partition_matches_baseline_counts_for_full_dataset() -> None:
    from scripts.process_freesolv import _freesolv_split_partition

    partition = _freesolv_split_partition(642, seed=1)

    assert len(partition.train_indices) == 513
    assert len(partition.valid_indices) == 64
    assert len(partition.test_indices) == 65
    assert partition.scheme == "moleculenet_random_like:513/64/65"


def test_freesolv_split_partition_scales_baseline_counts_when_rows_are_missing() -> None:
    from scripts.process_freesolv import _freesolv_split_partition

    partition = _freesolv_split_partition(186, seed=1)

    assert len(partition.train_indices) == 149
    assert len(partition.valid_indices) == 18
    assert len(partition.test_indices) == 19
    assert partition.scheme.startswith("moleculenet_random_like_scaled:")


def test_featurize_moladt_smiles_dataframe_uses_plain_moladt_feature_schema() -> None:
    frame = pd.DataFrame([{"mol_id": "mol-1", "smiles": "O", "target": 1.0}])

    table = featurize_moladt_smiles_dataframe(
        frame,
        dataset_name="unit_moladt",
        mol_id_column="mol_id",
        smiles_column="smiles",
        target_column="target",
    )

    assert table.feature_names == tuple(MOLADT_FEATURE_GROUPS)
    assert table.feature_groups == MOLADT_FEATURE_GROUPS
    assert "radius_of_gyration" not in table.rows.columns
