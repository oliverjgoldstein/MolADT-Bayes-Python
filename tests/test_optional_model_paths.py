from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from scripts.features import FeatureTable, featurize_moladt_featurized_geometry_records, featurize_moladt_geometry_records
from scripts.geometry_runner import GeometryRunConfig, _geometry_defaults, _import_geometry_stack, _train_member, run_geometry_ensemble
from scripts.model_errors import OptionalModelDependencyError
from scripts.model_registry import RegisteredModel
from scripts.run_all import _extend_with_property_results, _parse_extra_models, _uses_sdf_only_qm9_predictive_contract, build_parser
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
    assert args.qm9_split_mode == "long"


def test_run_all_qm9_parser_defaults_to_long_split() -> None:
    args = build_parser().parse_args(["qm9"])

    assert args.command == "qm9"
    assert args.split_mode == "long"


def test_models_command_defaults_include_both_geometry_families() -> None:
    args = build_parser().parse_args(["models"])

    assert _parse_extra_models(args) == ("catboost_uncertainty", "visnet_ensemble", "dimenetpp_ensemble")


def test_qm9_predictive_contract_is_sdf_only() -> None:
    args = build_parser().parse_args(
        [
            "qm9",
            "--include-moladt-predictive",
            "--models",
            "",
            "--extra-models",
            "catboost_uncertainty,visnet_ensemble",
        ]
    )

    assert _uses_sdf_only_qm9_predictive_contract(args, _parse_extra_models(args)) is True


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


def test_moladt_featurized_geometry_export_creation(tmp_path, monkeypatch) -> None:
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

    geometry_table = featurize_moladt_featurized_geometry_records(
        frame,
        dataset_name="demo_geom_featurized",
        mol_id_column="mol_id",
        mol_column="rdkit_mol",
        target_column="mu",
        record_index_column="sdf_record_index",
    )
    exported = export_geometric_splits(
        geometry_table,
        dataset_name="demo",
        representation="moladt_featurized_geom",
        target_name="mu",
        seed=7,
    )

    assert exported.representation == "moladt_featurized_geom"
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
        moladt_export=SimpleNamespace(),
        tabular_exports={"moladt": SimpleNamespace()},
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


def test_extend_with_property_results_skips_smiles_tabular_bundle(monkeypatch) -> None:
    import scripts.run_all as run_all

    seen_representations: list[str] = []

    def fake_tabular_runner(bundle, config):
        seen_representations.append(bundle.representation)
        return [], [], []

    monkeypatch.setitem(
        run_all.TABULAR_MODEL_REGISTRY,
        "catboost_uncertainty",
        RegisteredModel(name="catboost_uncertainty", input_kind="tabular", runner=fake_tabular_runner),
    )
    monkeypatch.setattr(run_all, "write_stan_data_json", lambda *args, **kwargs: None)
    artifacts = SimpleNamespace(
        moladt_export=SimpleNamespace(),
        tabular_exports={
            "smiles": SimpleNamespace(representation="smiles"),
            "moladt_featurized": SimpleNamespace(representation="moladt_featurized"),
        },
        geometric_exports={},
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

    _extend_with_property_results(
        artifacts,
        [],
        [],
        [],
        [],
        [],
        ("catboost_uncertainty",),
        args,
    )

    assert seen_representations == ["moladt_featurized"]


def test_extend_with_property_results_prefers_requested_qm9_geometry_representation(monkeypatch) -> None:
    import scripts.run_all as run_all

    seen_representations: list[str] = []

    def fake_geometry_runner(bundle, config):
        del config
        seen_representations.append(bundle.representation)
        return [], [], [], []

    monkeypatch.setitem(
        run_all.GEOMETRIC_MODEL_REGISTRY,
        "visnet_ensemble",
        RegisteredModel(name="visnet_ensemble", input_kind="geometric", runner=fake_geometry_runner),
    )
    monkeypatch.setattr(run_all, "write_stan_data_json", lambda *args, **kwargs: None)
    artifacts = SimpleNamespace(
        moladt_export=SimpleNamespace(),
        tabular_exports={},
        geometric_exports={
            "sdf_geom": SimpleNamespace(representation="sdf_geom"),
            "moladt_geom": SimpleNamespace(representation="moladt_geom"),
            "moladt_featurized_geom": SimpleNamespace(representation="moladt_featurized_geom"),
        },
    )
    args = SimpleNamespace(
        methods="",
        models="",
        seed=102,
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
        preferred_qm9_geometry_representation="moladt_featurized_geom",
    )

    _extend_with_property_results(
        artifacts,
        [],
        [],
        [],
        [],
        [],
        ("visnet_ensemble",),
        args,
    )

    assert seen_representations == ["moladt_featurized_geom"]


def test_dimenet_reports_missing_torch_sparse_dependency(monkeypatch) -> None:
    import scripts.geometry_runner as geometry_runner

    def fake_import_module(name: str):
        if name == "torch_sparse":
            raise ModuleNotFoundError("No module named 'torch_sparse'")
        return SimpleNamespace()

    monkeypatch.setattr(geometry_runner.importlib, "import_module", fake_import_module)

    with pytest.raises(OptionalModelDependencyError, match="torch-sparse"):
        _import_geometry_stack(model_name="dimenetpp_ensemble")


def test_train_member_logs_every_epoch(monkeypatch, capsys) -> None:
    import scripts.geometry_runner as geometry_runner

    torch = pytest.importorskip("torch")

    class DummyBatch:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def to(self, device):
            del device
            return self

    class BatchScaleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, batch):
            return batch.x.view(-1) * self.weight

    validation_predictions = [0.3, 0.2]

    def fake_predict_loader(**kwargs):
        del kwargs
        value = validation_predictions.pop(0)
        return {
            "predicted_mean": np.asarray([value], dtype=float),
            "actual": np.asarray([0.0], dtype=float),
            "mol_ids": ("mol_1",),
        }

    monkeypatch.setattr(geometry_runner, "_predict_loader", fake_predict_loader)

    model = BatchScaleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = DummyBatch(torch.tensor([[1.0]], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float32))

    _train_member(
        torch=torch,
        model=model,
        optimizer=optimizer,
        train_loader=[batch],
        valid_loader=[batch],
        device=torch.device("cpu"),
        target_mean=0.0,
        target_std=1.0,
        max_epochs=2,
        patience=5,
        gradient_clip_norm=5.0,
        progress_label="visnet_ensemble:demo/sdf_geom",
        target_name="mu",
        seed_index=1,
        seed_count=1,
        verbose=True,
    )

    output = capsys.readouterr().out
    assert "epoch 1/2" in output
    assert "epoch 2/2" in output
    assert "valid_mae=" in output
    assert "target=mu" in output


def test_train_member_stops_immediately_on_nan_validation_and_restores_best(monkeypatch) -> None:
    import scripts.geometry_runner as geometry_runner

    torch = pytest.importorskip("torch")

    class DummyBatch:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def to(self, device):
            del device
            return self

    class BatchScaleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, batch):
            return batch.x.view(-1) * self.weight

    saved_weight: dict[str, torch.Tensor] = {}
    validation_predictions = [0.2, float("nan")]

    def fake_predict_loader(*, model, **kwargs):
        del kwargs
        value = validation_predictions.pop(0)
        if np.isfinite(value):
            saved_weight["value"] = model.weight.detach().clone()
        return {
            "predicted_mean": np.asarray([value], dtype=float),
            "actual": np.asarray([0.0], dtype=float),
            "mol_ids": ("mol_1",),
        }

    monkeypatch.setattr(geometry_runner, "_predict_loader", fake_predict_loader)

    model = BatchScaleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = DummyBatch(torch.tensor([[1.0]], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float32))

    history = _train_member(
        torch=torch,
        model=model,
        optimizer=optimizer,
        train_loader=[batch],
        valid_loader=[batch],
        device=torch.device("cpu"),
        target_mean=0.0,
        target_std=1.0,
        max_epochs=5,
        patience=30,
        gradient_clip_norm=5.0,
        progress_label="visnet_ensemble:demo/sdf_geom",
        target_name="mu",
        seed_index=1,
        seed_count=1,
        verbose=False,
    )

    assert len(history["training_curves"]) == 2
    assert history["training_curves"][1]["valid_rmse"] != history["training_curves"][1]["valid_rmse"]
    assert history["training_curves"][1]["valid_mae"] != history["training_curves"][1]["valid_mae"]
    assert torch.allclose(model.weight.detach(), saved_weight["value"])


def test_qm9_geometry_defaults_use_25_epochs() -> None:
    defaults = _geometry_defaults("qm9", "visnet_ensemble")

    assert defaults["max_epochs"] == 25
    assert defaults["patience"] == 25


def test_run_geometry_ensemble_uses_unshuffled_train_loader_for_metrics(monkeypatch) -> None:
    import scripts.geometry_runner as geometry_runner

    torch = pytest.importorskip("torch")

    class FakeData:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def to(self, device):
            del device
            return self

    class FakeDataLoader:
        def __init__(self, data, batch_size, shuffle):
            del batch_size
            self.data = list(data)
            self.shuffle = shuffle

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, batch):
            return batch.y.view(-1) * 0.0

    monkeypatch.setattr(
        geometry_runner,
        "_import_geometry_stack",
        lambda *, model_name: (
            torch,
            SimpleNamespace(Data=FakeData),
            SimpleNamespace(DataLoader=FakeDataLoader),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(geometry_runner, "_build_geometry_model", lambda **kwargs: DummyModel())
    monkeypatch.setattr(
        geometry_runner,
        "_train_member",
        lambda **kwargs: {"training_curves": [{"epoch": 1, "train_loss": 0.01, "valid_rmse": 0.0, "valid_mae": 0.0}]},
    )
    monkeypatch.setattr(geometry_runner, "_write_geometry_artifact_manifest", lambda **kwargs: [])

    train_shuffle_calls = {"count": 0}

    def fake_predict_loader(*, loader, **kwargs):
        del kwargs
        actual = np.asarray([float(item.y.view(-1)[0].item()) for item in loader.data], dtype=float)
        mol_ids = tuple(str(item.mol_id) for item in loader.data)
        if loader.shuffle and len(actual) == 2:
            train_shuffle_calls["count"] += 1
            order = [0, 1] if train_shuffle_calls["count"] % 2 == 1 else [1, 0]
            actual = actual[order]
            mol_ids = tuple(np.asarray(mol_ids)[order].tolist())
        return {
            "predicted_mean": actual.copy(),
            "actual": actual,
            "mol_ids": mol_ids,
        }

    monkeypatch.setattr(geometry_runner, "_predict_loader", fake_predict_loader)

    bundle = SimpleNamespace(
        dataset_name="qm9",
        representation="moladt_featurized_geom",
        rows=pd.DataFrame(
            [
                {"mol_id": "mol_1", "mu": 1.0},
                {"mol_id": "mol_2", "mu": 2.0},
                {"mol_id": "mol_3", "mu": 3.0},
                {"mol_id": "mol_4", "mu": 4.0},
            ]
        ),
        train_indices=np.asarray([0, 1]),
        valid_indices=np.asarray([2]),
        test_indices=np.asarray([3]),
        atomic_numbers=np.asarray([[6], [6], [6], [6]], dtype=np.int64),
        coordinates=np.asarray(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        global_features=None,
        global_feature_names=(),
        target_name="mu",
        split_scheme="long:fractional_0.8/0.1/0.1",
        source_row_count=4,
        used_row_count=4,
    )

    metrics_rows, prediction_rows, training_curve_rows, artifact_rows = run_geometry_ensemble(
        bundle,
        config=GeometryRunConfig(model_name="visnet_ensemble", seeds=(102,), verbose=False),
    )

    del prediction_rows, training_curve_rows, artifact_rows
    train_metrics = next(row for row in metrics_rows if row["split"] == "train")
    assert train_metrics["mae"] == pytest.approx(0.0)
    assert train_metrics["rmse"] == pytest.approx(0.0)
