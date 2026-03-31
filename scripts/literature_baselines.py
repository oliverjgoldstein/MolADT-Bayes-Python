from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

MOLECULENET_SOURCE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/"
GILMER_SOURCE_URL = "https://proceedings.mlr.press/v70/gilmer17a.html"
GILMER_SUPPLEMENT_URL = "https://proceedings.mlr.press/v70/gilmer17a/gilmer17a-supp.pdf"
UNIMOL_SOURCE_URL = "https://openreview.net/forum?id=6K2RM6wVqKu"
UNIMOL_REPO_URL = "https://github.com/deepmodeling/Uni-Mol"
MGNN_SOURCE_URL = "https://www.nature.com/articles/s41524-025-01541-5"
PAINN_SOURCE_URL = "https://proceedings.mlr.press/v139/schutt21a.html"
VISNET_SOURCE_URL = "https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.nn.models.ViSNet.html"
DIMENETPP_SOURCE_URL = "https://pytorch-geometric.readthedocs.io/en/2.5.1/_modules/torch_geometric/nn/models/dimenet.html"


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
        note="Useful external context for FreeSolv, but not directly comparable unless the local split protocol matches MoleculeNet.",
    ),
    LiteratureBaseline(
        dataset="freesolv",
        target="expt",
        model_name="XGBoost",
        model_family="tabular_boosting",
        metric_name="RMSE",
        metric_value=1.74,
        units="kcal/mol",
        split_protocol="MoleculeNet random split (Table 3)",
        source_title="MoleculeNet: a benchmark for molecular machine learning",
        source_url=MOLECULENET_SOURCE_URL,
        directly_comparable=False,
        note="Tabular baseline from the same MoleculeNet FreeSolv table; included as context for descriptor-only models.",
    ),
    LiteratureBaseline(
        dataset="qm9",
        target="mu",
        model_name="MPNN",
        model_family="3d_message_passing",
        metric_name="MAE ratio",
        metric_value=0.30,
        units="fraction of 0.1 Debye chemical-accuracy target",
        split_protocol="110462/10000/10000",
        source_title="Neural Message Passing for Quantum Chemistry",
        source_url=GILMER_SUPPLEMENT_URL,
        directly_comparable=False,
        note="Gilmer et al. report a normalized mu error ratio rather than the exact same scalar metrics used here.",
    ),
    LiteratureBaseline(
        dataset="qm9",
        target="mu",
        model_name="Uni-Mol",
        model_family="3d_pretrained_transformer",
        metric_name="",
        metric_value=None,
        units="",
        split_protocol="paper-specific downstream protocols",
        source_title="Uni-Mol: A Universal 3D Molecular Representation Learning Framework",
        source_url=UNIMOL_SOURCE_URL,
        directly_comparable=False,
        note="The official Uni-Mol paper and repo are included for manuscript context; this repo has not verified an exact QM9 mu row under the same protocol.",
    ),
    LiteratureBaseline(
        dataset="qm9",
        target="mu",
        model_name="Uni-Mol repo",
        model_family="3d_pretrained_transformer",
        metric_name="",
        metric_value=None,
        units="",
        split_protocol="official downstream tooling",
        source_title="Official Uni-Mol repository",
        source_url=UNIMOL_REPO_URL,
        directly_comparable=False,
        note="Repository row kept separate from the paper row to make the source trail explicit.",
    ),
    LiteratureBaseline(
        dataset="qm9",
        target="mu",
        model_name="MGNN",
        model_family="3d_message_passing",
        metric_name="",
        metric_value=None,
        units="",
        split_protocol="paper-specific QM9 protocol",
        source_title="MGNN: Moment Graph Neural Network for Universal Molecular Potentials",
        source_url=MGNN_SOURCE_URL,
        directly_comparable=False,
        note="The 2025 MGNN paper reports strong QM9 results, but this repo has not verified an exact mu metric row under the same split and training recipe.",
    ),
    LiteratureBaseline(
        dataset="qm9",
        target="mu",
        model_name="PaiNN",
        model_family="equivariant_message_passing",
        metric_name="MAE",
        metric_value=0.012,
        units="Debye",
        split_protocol="QM9 protocol reported in PaiNN Table 2",
        source_title="Equivariant message passing for the prediction of tensorial properties and molecular spectra",
        source_url=PAINN_SOURCE_URL,
        directly_comparable=False,
        note="ICML 2021 equivariant message-passing baseline for QM9 mu. Included as external MAE context only; this repo has not verified an exact split-and-training match to the local benchmark.",
    ),
    LiteratureBaseline(
        dataset="qm9",
        target="mu",
        model_name="ViSNet",
        model_family="equivariant_message_passing",
        metric_name="",
        metric_value=None,
        units="",
        split_protocol="implementation family reference",
        source_title="PyTorch Geometric ViSNet reference",
        source_url=VISNET_SOURCE_URL,
        directly_comparable=False,
        note="This row documents the model family used in the optional geometry branch; exact comparable paper metrics are not asserted here.",
    ),
    LiteratureBaseline(
        dataset="qm9",
        target="mu",
        model_name="DimeNet++",
        model_family="equivariant_message_passing",
        metric_name="",
        metric_value=None,
        units="",
        split_protocol="implementation family reference",
        source_title="PyTorch Geometric DimeNet++ reference",
        source_url=DIMENETPP_SOURCE_URL,
        directly_comparable=False,
        note="Fallback geometry family reference only; no exact QM9 mu metric is claimed without a verified matching source row.",
    ),
)


def literature_baselines_frame() -> pd.DataFrame:
    return pd.DataFrame([row.to_dict() for row in BASELINES])
