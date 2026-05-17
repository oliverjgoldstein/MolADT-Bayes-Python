from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

MOLECULENET_SOURCE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/"


@dataclass(frozen=True, slots=True)
class LiteratureBaseline:
    dataset: str
    target: str
    model_name: str
    model_family: str
    metric_name: str
    metric_value: float | None
    units: str
    split_protocol: str
    source_title: str
    source_url: str
    directly_comparable: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metric_value"] = payload["metric_value"] if payload["metric_value"] is not None else pd.NA
        return payload


BASELINES: tuple[LiteratureBaseline, ...] = (
    LiteratureBaseline(
        dataset="freesolv",
        target="expt",
        model_name="MPNN",
        model_family="3d_message_passing",
        metric_name="RMSE",
        metric_value=1.15,
        units="kcal/mol",
        split_protocol="MoleculeNet random split (Table 3)",
        source_title="MoleculeNet: a benchmark for molecular machine learning",
        source_url=MOLECULENET_SOURCE_URL,
        directly_comparable=False,
        note="MoleculeNet Table 3 graph-based baseline for FreeSolv. The metric matches, but the local split and model family still differ from the paper.",
    ),
    LiteratureBaseline(
        dataset="qm9",
        target="mu",
        model_name="DTNN",
        model_family="distance_tensor_network",
        metric_name="MAE",
        metric_value=2.35,
        units="milli-Debye",
        split_protocol="MoleculeNet random split (Table 3)",
        source_title="MoleculeNet: a benchmark for molecular machine learning",
        source_url=MOLECULENET_SOURCE_URL,
        directly_comparable=False,
        note="MoleculeNet Table 3 graph-based baseline for QM9. The paper metric is MAE, but the local split and model family still differ from the original benchmark.",
    ),
)


def literature_baselines_frame() -> pd.DataFrame:
    return pd.DataFrame([row.to_dict() for row in BASELINES])
