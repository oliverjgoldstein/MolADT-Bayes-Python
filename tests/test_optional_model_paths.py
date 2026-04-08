from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from scripts.features import FeatureTable, featurize_moladt_geometry_records, featurize_moladt_typed_geometry_records
from scripts.geometry_runner import _import_geometry_stack
from scripts.model_errors import OptionalModelDependencyError
from scripts.model_registry import RegisteredModel
from scripts.run_all import _extend_with_property_results, _parse_extra_models, build_parser
from scripts.splits import export_geometric_splits, export_standardized_splits
from scripts.tabular_runner import CATBOOST_METHOD, CATBOOST_MODEL, CatBoostRunConfig, run_catboost_uncertainty


def test_run_all_parser_accepts_optional_model_flags() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "--extra-models",
            "catboost_uncertainty,visnet_ensemble",
            "--paper-mode",
            "--num-seeds",
            "3",
            "--full-qm9",
            "--geom-model",
            "dimenetpp",
            "--skip-geom",
        ]
    )

    assert args.command == "benchmark"
    assert args.paper_mode is True
    assert args.num_seeds == 3
    assert args.full_qm9 is True
    assert args.geom_model == "dimenetpp"
    assert args.skip_geom is True


def test_run_all_models_parser_defaults_to_models_command() -> None:
    args = build_parser().parse_args(["models", "--verbose"])

    assert args.command == "models"
    assert args.include_moladt_predictive is True


def test_models_command_defaults_include_both_geometry_families() -> None:
    args = build_parser().parse_args(["models"])

    assert _parse_extra_models(args) == ("catboost_uncertainty", "visnet_ensemble", "dimenetpp_ensemble")


def test_moladt_geometry_export_creation(tmp_path, monkeypatch) -> None:
    import scripts.splits as splits

    monkeypatch.setattr(splits, "PROCESSED_DATA_DIR", tmp_path)
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(molecule, randomSeed=1)
    frame = pd.DataFrame(
        [
            {"mol_id": f"mol_{index}", "mu": float(index), "sdf_record_index": index, "rdkit_mol": molecule}
            for index in range(12)
        ]
    )

    geometry_table = featurize_moladt_geometry_records(
        frame,
        dataset_name="demo_geom",
        mol_id_column="mol_id",
        mol_column="rdkit_mol",
        target_column="mu",
        record_index_column="sdf_record_index",
    )
    exported = export_geometric_splits(
        geometry_table,
        dataset_name="demo",
        representation="moladt_geom",
        target_name="mu",
        seed=7,
    )

    assert exported.representation == "moladt_geom"
    assert exported.global_features is not None
    assert "weight" in exported.global_feature_names
    assert len(exported.train_indices) > 0
    assert exported.metadata_path.exists()


def test_moladt_typed_geometry_export_creation(tmp_path, monkeypatch) -> None:
    import scripts.splits as splits

    monkeypatch.setattr(splits, "PROCESSED_DATA_DIR", tmp_path)
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(molecule, randomSeed=1)
    frame = pd.DataFrame(
        [
            {"mol_id": f"mol_{index}", "mu": float(index), "sdf_record_index": index, "rdkit_mol": molecule}
            for index in range(12)
        ]
    )

    geometry_table = featurize_moladt_typed_geometry_records(
        frame,
        dataset_name="demo_geom_typed",
        mol_id_column="mol_id",
        mol_column="rdkit_mol",
        target_column="mu",
        record_index_column="sdf_record_index",
    )
    exported = export_geometric_splits(
        geometry_table,
        dataset_name="demo",
        representation="moladt_typed_geom",
        target_name="mu",
        seed=7,
    )

    assert exported.representation == "moladt_typed_geom"
    assert exported.global_features is not None
    assert "pair_count_c_o" in exported.global_feature_names
    assert "aprdf_all_1p5a" in exported.global_feature_names
    assert "bond_angle_all_120d" in exported.global_feature_names
    assert "torsion_all_180d" in exported.global_feature_names
    assert exported.metadata_path.exists()


def test_catboost_runner_outputs_expected_schema(tmp_path, monkeypatch) -> None:
    import scripts.splits as splits
    import scripts.tabular_runner as tabular_runner

    monkeypatch.setattr(splits, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(tabular_runner, "RESULTS_DIR", tmp_path / "results")

    rows = pd.DataFrame(
        [
            {"mol_id": f"mol_{index}", "smiles": "CCO", "target": float(index), "x1": float(index), "x2": float(index + 1)}
            for index in range(18)
        ]
    )
    bundle = export_standardized_splits(
        FeatureTable(
            rows=rows,
            feature_names=("x1", "x2"),
            feature_groups={"x1": "group_a", "x2": "group_b"},
            failures=(),
        ),
        dataset_name="demo",
        representation="moladt",
        target_name="target",
        seed=3,
    )

    class FakeCatBoostRegressor:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y, eval_set=None, use_best_model=False, verbose=False, early_stopping_rounds=None):
            self.feature_count = X.shape[1]
            self.eval_result = {"learn": {"RMSE": [1.0, 0.8]}, "validation": {"RMSE": [1.1, 0.9]}}
            return self

        def virtual_ensembles_predict(self, X, prediction_type=None, virtual_ensembles_count=None):
            mean = X.to_numpy(dtype=float).sum(axis=1) * 0.01 + 0.5
            data = np.full(len(X), 0.04)
            knowledge = np.full(len(X), 0.01)
            return np.column_stack([mean, data, knowledge])

        def get_feature_importance(self, data=None, type=None):
            if type == "ShapValues":
                base = np.tile(np.array([[0.2, 0.1, 0.0]], dtype=float), (len(data), 1))
                return base
            return np.array([0.7, 0.3], dtype=float)

    monkeypatch.setitem(sys.modules, "catboost", SimpleNamespace(CatBoostRegressor=FakeCatBoostRegressor))

    metrics_rows, prediction_rows, artifact_rows = run_catboost_uncertainty(
        bundle,
        config=CatBoostRunConfig(seeds=(11,), search_hyperparameters=False),
    )

    assert len(metrics_rows) == 3
    assert {row["split"] for row in metrics_rows} == {"train", "valid", "test"}
    assert all(row["model"] == CATBOOST_MODEL for row in metrics_rows)
    assert all(row["method"] == CATBOOST_METHOD for row in metrics_rows)
    assert {"actual", "predicted_mean", "predictive_sd", "data_uncertainty", "knowledge_uncertainty", "total_uncertainty"}.issubset(prediction_rows[0].keys())
    assert artifact_rows


def test_extend_with_property_results_skips_missing_optional_geometry_dependency(monkeypatch) -> None:
    import scripts.run_all as run_all

    def failing_geometry_runner(*args, **kwargs):
        raise OptionalModelDependencyError("Geometry extension stack is incomplete.")

    monkeypatch.setattr(run_all, "write_stan_data_json", lambda *args, **kwargs: None)
    monkeypatch.setitem(
        run_all.GEOMETRIC_MODEL_REGISTRY,
        "visnet_ensemble",
        RegisteredModel(name="visnet_ensemble", input_kind="geometric", runner=failing_geometry_runner),
    )
    artifacts = SimpleNamespace(
        smiles_export=SimpleNamespace(),
        tabular_exports={"smiles": SimpleNamespace()},
        geometric_exports={"moladt_geom": SimpleNamespace()},
    )
    args = SimpleNamespace(
        methods="",
        models="",
        seed=7,
        sample_chains=1,
        sample_warmup=1,
        sample_draws=1,
        approximation_draws=1,
        variational_iterations=1,
        optimize_iterations=1,
        pathfinder_paths=1,
        predictive_draws=1,
        verbose=False,
        num_seeds=1,
        paper_mode=False,
    )
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    training_curve_rows: list[dict[str, object]] = []
    model_artifact_rows: list[dict[str, object]] = []

    _extend_with_property_results(
        artifacts,
        metrics_rows,
        prediction_rows,
        coefficient_rows,
        training_curve_rows,
        model_artifact_rows,
        ("visnet_ensemble",),
        args,
    )

    assert metrics_rows == []
    assert prediction_rows == []
    assert training_curve_rows == []


def test_dimenet_reports_missing_torch_sparse_dependency(monkeypatch) -> None:
    import scripts.geometry_runner as geometry_runner

    def fake_import_module(name: str):
        if name == "torch_sparse":
            raise ModuleNotFoundError("No module named 'torch_sparse'")
        return SimpleNamespace()

    monkeypatch.setattr(geometry_runner.importlib, "import_module", fake_import_module)

    with pytest.raises(OptionalModelDependencyError, match="torch-sparse"):
        _import_geometry_stack(model_name="dimenetpp_ensemble")
