from __future__ import annotations

import importlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import RESULTS_DIR, ensure_directory, log
from .model_errors import OptionalModelDependencyError
from .predictive_metrics import build_metric_row, build_prediction_rows
from .splits import GeometricDatasetSpec

GEOMETRY_METHOD = "deep_ensemble"


@dataclass(frozen=True, slots=True)
class GeometryRunConfig:
    model_name: str
    seeds: tuple[int, ...]
    verbose: bool = False


def run_geometry_ensemble(
    bundle: GeometricDatasetSpec,
    *,
    config: GeometryRunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    torch, pyg_data, pyg_loader, pyg_models = _import_geometry_stack(model_name=config.model_name)
    dataset_defaults = _geometry_defaults(bundle.dataset_name, config.model_name)
    device = torch.device("cpu")
    train_data = _build_data_objects(torch, pyg_data, bundle, bundle.train_indices)
    valid_data = _build_data_objects(torch, pyg_data, bundle, bundle.valid_indices)
    test_data = _build_data_objects(torch, pyg_data, bundle, bundle.test_indices)
    train_loader = pyg_loader.DataLoader(train_data, batch_size=dataset_defaults["batch_size"], shuffle=True)
    valid_loader = pyg_loader.DataLoader(valid_data, batch_size=dataset_defaults["batch_size"], shuffle=False)
    test_loader = pyg_loader.DataLoader(test_data, batch_size=dataset_defaults["batch_size"], shuffle=False)
    y_train = bundle.rows.loc[bundle.train_indices, bundle.target_name].to_numpy(dtype=float)
    target_mean = float(np.mean(y_train))
    target_std = float(np.std(y_train))
    if target_std <= 0.0:
        target_std = 1.0

    training_curve_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    member_predictions: dict[str, list[np.ndarray]] = {"train": [], "valid": [], "test": []}
    parameter_count: int | None = None
    progress_label = f"{config.model_name}:{bundle.dataset_name}/{bundle.representation}"
    if config.verbose:
        log(f"[{progress_label}] training ensemble with {len(config.seeds)} member(s)")
    start = time.perf_counter()
    for seed_index, seed in enumerate(config.seeds, start=1):
        _seed_everything(torch, seed)
        model = _build_geometry_model(
            torch=torch,
            pyg_models=pyg_models,
            model_name=config.model_name,
            global_dim=bundle.global_features.shape[1] if bundle.global_features is not None else 0,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(dataset_defaults["lr"]),
            weight_decay=float(dataset_defaults["weight_decay"]),
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if config.verbose:
            log(
                f"[{progress_label}] starting member {seed_index}/{len(config.seeds)} "
                f"(seed={seed}, max_epochs={int(dataset_defaults['max_epochs'])})"
            )
        history = _train_member(
            torch=torch,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            max_epochs=int(dataset_defaults["max_epochs"]),
            patience=int(dataset_defaults["patience"]),
            gradient_clip_norm=float(dataset_defaults["gradient_clip_norm"]),
            progress_label=progress_label,
            seed_index=seed_index,
            seed_count=len(config.seeds),
            verbose=config.verbose,
        )
        training_curve_rows.extend(
            {
                "dataset": bundle.dataset_name,
                "representation": bundle.representation,
                "model": config.model_name,
                "method": GEOMETRY_METHOD,
                "seed": seed,
                **row,
            }
            for row in history["training_curves"]
        )
        member_predictions["train"].append(
            _predict_loader(
                torch=torch,
                model=model,
                loader=train_loader,
                device=device,
                target_mean=target_mean,
                target_std=target_std,
            )["predicted_mean"]
        )
        member_predictions["valid"].append(
            _predict_loader(
                torch=torch,
                model=model,
                loader=valid_loader,
                device=device,
                target_mean=target_mean,
                target_std=target_std,
            )["predicted_mean"]
        )
        member_predictions["test"].append(
            _predict_loader(
                torch=torch,
                model=model,
                loader=test_loader,
                device=device,
                target_mean=target_mean,
                target_std=target_std,
            )["predicted_mean"]
        )
        if config.verbose:
            best_valid_rmse = min(float(row["valid_rmse"]) for row in history["training_curves"])
            log(
                f"[{progress_label}] finished member {seed_index}/{len(config.seeds)} "
                f"with best validation RMSE {best_valid_rmse:.4f}"
            )
    runtime_seconds = time.perf_counter() - start
    if config.verbose:
        log(f"[{progress_label}] finished ensemble in {runtime_seconds:.1f}s")
    artifact_rows.extend(
        _write_geometry_artifact_manifest(
            bundle=bundle,
            model_name=config.model_name,
            seeds=config.seeds,
            defaults=dataset_defaults,
        )
    )
    split_payloads = {
        "train": _predict_loader(torch=torch, model=model, loader=train_loader, device=device, target_mean=target_mean, target_std=target_std),
        "valid": _predict_loader(torch=torch, model=model, loader=valid_loader, device=device, target_mean=target_mean, target_std=target_std),
        "test": _predict_loader(torch=torch, model=model, loader=test_loader, device=device, target_mean=target_mean, target_std=target_std),
    }
    for split_name, payload in split_payloads.items():
        ensemble_members = np.stack(member_predictions[split_name], axis=0)
        predicted_mean = np.mean(ensemble_members, axis=0)
        predictive_sd = np.std(ensemble_members, axis=0)
        metrics_rows.append(
            build_metric_row(
                dataset_name=bundle.dataset_name,
                representation=bundle.representation,
                model_name=config.model_name,
                method=GEOMETRY_METHOD,
                split_name=split_name,
                mol_ids=tuple(payload["mol_ids"]),
                actual=payload["actual"],
                predicted_mean=predicted_mean,
                predictive_sd=predictive_sd,
                runtime_seconds=runtime_seconds,
                feature_count=int(len(bundle.global_feature_names)),
                n_train=len(bundle.train_indices),
                split_scheme=bundle.split_scheme,
                source_row_count=bundle.source_row_count,
                used_row_count=bundle.used_row_count,
                seed="ensemble",
                draw_count=len(config.seeds),
                parameter_count=parameter_count,
            )
        )
        prediction_rows.extend(
            build_prediction_rows(
                dataset_name=bundle.dataset_name,
                representation=bundle.representation,
                model_name=config.model_name,
                method=GEOMETRY_METHOD,
                split_name=split_name,
                mol_ids=tuple(payload["mol_ids"]),
                actual=payload["actual"],
                predicted_mean=predicted_mean,
                predictive_sd=predictive_sd,
                seed="ensemble",
                extra_columns={"epistemic_uncertainty": np.square(predictive_sd)},
            )
        )
    return metrics_rows, prediction_rows, training_curve_rows, artifact_rows


def _build_geometry_model(
    *,
    torch: Any,
    pyg_models: Any,
    model_name: str,
    global_dim: int,
) -> Any:
    nn = torch.nn

    class FusionGeometryRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if model_name == "visnet_ensemble":
                self.base = pyg_models.ViSNet(
                    hidden_channels=128,
                    num_layers=6,
                    num_heads=8,
                    num_rbf=32,
                    cutoff=5.0,
                    max_num_neighbors=32,
                    lmax=1,
                    reduce_op="sum",
                    trainable_rbf=False,
                    trainable_vecnorm=False,
                    mean=0.0,
                    std=1.0,
                    derivative=False,
                )
            elif model_name == "dimenetpp_ensemble":
                self.base = pyg_models.DimeNetPlusPlus(
                    hidden_channels=128,
                    out_channels=1,
                    num_blocks=4,
                    int_emb_size=64,
                    basis_emb_size=8,
                    out_emb_channels=256,
                    num_spherical=7,
                    num_radial=6,
                    cutoff=5.0,
                    max_num_neighbors=32,
                )
            else:
                raise ValueError(f"Unsupported geometry model {model_name}")
            if global_dim > 0:
                hidden = max(32, min(256, global_dim * 2))
                self.global_head = nn.Sequential(
                    nn.Linear(global_dim, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
            else:
                self.global_head = None

        def forward(self, batch: Any) -> Any:
            prediction = self.base(batch.z, batch.pos, batch.batch)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            prediction = prediction.view(-1)
            if self.global_head is not None:
                global_x = batch.global_x.view(-1, global_dim)
                prediction = prediction + self.global_head(global_x).view(-1)
            return prediction

    return FusionGeometryRegressor()


def _train_member(
    *,
    torch: Any,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    valid_loader: Any,
    device: Any,
    target_mean: float,
    target_std: float,
    max_epochs: int,
    patience: int,
    gradient_clip_norm: float,
    progress_label: str,
    seed_index: int,
    seed_count: int,
    verbose: bool,
) -> dict[str, Any]:
    nnf = torch.nn.functional
    best_state: dict[str, Any] | None = None
    best_valid_rmse = float("inf")
    epochs_without_improvement = 0
    training_curves: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            target = (batch.y.view(-1) - target_mean) / target_std
            loss = nnf.mse_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        valid_payload = _predict_loader(
            torch=torch,
            model=model,
            loader=valid_loader,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )
        valid_rmse = float(np.sqrt(np.mean(np.square(valid_payload["predicted_mean"] - valid_payload["actual"]))))
        training_curves.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
                "valid_rmse": valid_rmse,
            }
        )
        if verbose and (epoch == 1 or epoch % 25 == 0):
            log(
                f"[{progress_label}] member {seed_index}/{seed_count} epoch {epoch}/{max_epochs} "
                f"train_loss={training_curves[-1]['train_loss']:.4f} valid_rmse={valid_rmse:.4f} "
                f"best={min(float(row['valid_rmse']) for row in training_curves):.4f}"
            )
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    log(
                        f"[{progress_label}] member {seed_index}/{seed_count} early stopped at epoch {epoch} "
                        f"with best validation RMSE {best_valid_rmse:.4f}"
                    )
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"training_curves": training_curves}


def _predict_loader(
    *,
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    target_mean: float,
    target_std: float,
) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    mol_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch)
            prediction = prediction.view(-1).detach().cpu().numpy() * target_std + target_mean
            actual = batch.y.view(-1).detach().cpu().numpy()
            predictions.append(prediction)
            actuals.append(actual)
            mol_ids.extend(str(mol_id) for mol_id in batch.mol_id)
    return {
        "predicted_mean": np.concatenate(predictions, axis=0) if predictions else np.array([], dtype=float),
        "actual": np.concatenate(actuals, axis=0) if actuals else np.array([], dtype=float),
        "mol_ids": tuple(mol_ids),
    }


def _build_data_objects(torch: Any, pyg_data: Any, bundle: GeometricDatasetSpec, indices: np.ndarray) -> list[Any]:
    data_list: list[Any] = []
    for index in indices.tolist():
        data = pyg_data.Data(
            z=torch.tensor(bundle.atomic_numbers[index], dtype=torch.long),
            pos=torch.tensor(bundle.coordinates[index], dtype=torch.float32),
            y=torch.tensor([float(bundle.rows.iloc[index][bundle.target_name])], dtype=torch.float32),
        )
        data.mol_id = str(bundle.rows.iloc[index]["mol_id"])
        if bundle.global_features is not None:
            data.global_x = torch.tensor(bundle.global_features[index][np.newaxis, :], dtype=torch.float32)
        data_list.append(data)
    return data_list


def _geometry_defaults(dataset_name: str, model_name: str) -> dict[str, float | int]:
    base = {
        "lr": 3e-4,
        "weight_decay": 1e-6,
        "gradient_clip_norm": 5.0,
    }
    if dataset_name == "freesolv":
        base.update({"batch_size": 32, "max_epochs": 400, "patience": 50})
    else:
        base.update({"batch_size": 128, "max_epochs": 200, "patience": 30})
    return base


def _write_geometry_artifact_manifest(
    *,
    bundle: GeometricDatasetSpec,
    model_name: str,
    seeds: tuple[int, ...],
    defaults: dict[str, float | int],
) -> list[dict[str, Any]]:
    artifact_dir = ensure_directory(RESULTS_DIR / "model_artifacts" / model_name / bundle.dataset_name / bundle.representation)
    config_path = artifact_dir / "ensemble_config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": bundle.dataset_name,
                "representation": bundle.representation,
                "model": model_name,
                "method": GEOMETRY_METHOD,
                "seeds": list(seeds),
                "defaults": defaults,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return [
        {
            "dataset": bundle.dataset_name,
            "representation": bundle.representation,
            "model": model_name,
            "seed": "ensemble",
            "artifact_type": "ensemble_config",
            "path": str(config_path.relative_to(RESULTS_DIR)),
        }
    ]


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _import_geometry_stack(*, model_name: str) -> tuple[Any, Any, Any, Any]:
    try:
        torch = importlib.import_module("torch")
        pyg_data = importlib.import_module("torch_geometric.data")
        pyg_loader = importlib.import_module("torch_geometric.loader")
        pyg_models = importlib.import_module("torch_geometric.nn.models")
    except ModuleNotFoundError as exc:
        raise OptionalModelDependencyError(
            "Geometry models require the local geometric stack. Re-run `make python-setup` "
            "to install the benchmark model dependencies."
        ) from exc
    try:
        importlib.import_module("torch_cluster")
    except ModuleNotFoundError as exc:
        raise OptionalModelDependencyError(
            "Geometry models require `torch-cluster` in the local repo environment. "
            "Re-run `make python-setup` so the full PyTorch Geometric runtime is installed."
        ) from exc
    if model_name == "dimenetpp_ensemble":
        try:
            importlib.import_module("torch_sparse")
        except ModuleNotFoundError as exc:
            raise OptionalModelDependencyError(
                "DimeNet++ requires `torch-sparse` in the local repo environment. "
                "Re-run `make python-setup` so the full PyTorch Geometric runtime is installed."
            ) from exc
    return torch, pyg_data, pyg_loader, pyg_models
