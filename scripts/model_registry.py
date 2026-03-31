from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .geometry_runner import run_geometry_ensemble
from .tabular_runner import run_catboost_uncertainty


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    name: str
    input_kind: str
    runner: Callable[..., Any]


TABULAR_MODEL_REGISTRY: dict[str, RegisteredModel] = {
    "catboost_uncertainty": RegisteredModel(
        name="catboost_uncertainty",
        input_kind="tabular",
        runner=run_catboost_uncertainty,
    ),
}

GEOMETRIC_MODEL_REGISTRY: dict[str, RegisteredModel] = {
    "visnet_ensemble": RegisteredModel(
        name="visnet_ensemble",
        input_kind="geometric",
        runner=run_geometry_ensemble,
    ),
    "dimenetpp_ensemble": RegisteredModel(
        name="dimenetpp_ensemble",
        input_kind="geometric",
        runner=run_geometry_ensemble,
    ),
}

ALL_REGISTERED_MODELS: dict[str, RegisteredModel] = {
    **TABULAR_MODEL_REGISTRY,
    **GEOMETRIC_MODEL_REGISTRY,
}
