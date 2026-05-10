from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from moladt.chem.dietz import Edge
from moladt.chem.molecule import Atom, Molecule, molecule_edges
from moladt.chem.molecule_ops import effective_order

from .common import PROCESSED_DATA_DIR, ensure_directory
from .freesolv_wl_system_gp import (
    _Bundle,
    _build_kernel_components,
    _fit_gp,
    _graph_token_counts,
    _load_bundle_from_rows,
    _order_bucket,
    _predict_gp,
    _shell_stats,
    _system_kind,
    _system_token_counts,
    _vectorize,
    train_valid_test_indices,
)
from .predictive_metrics import regression_summary


DEFAULT_SPLIT_COUNT = 20
DEFAULT_SEED_START = 0
WL_RADIUS = 4


@dataclass(frozen=True, slots=True)
class AblationVariant:
    code: str
    label: str
    description: str
    wl_counter_fn: Callable[[Molecule], Counter[str]]
    system_counter_fn: Callable[[Molecule], Counter[str]]


def _atom_symbol(atom: Atom) -> str:
    return atom.attributes.symbol.value


def _edge_symbol_pair(molecule: Molecule, edge: Edge) -> str:
    left = _atom_symbol(molecule.atoms[edge.a])
    right = _atom_symbol(molecule.atoms[edge.b])
    return "-".join(sorted((left, right)))


def _charge_bucket(charge: int) -> str:
    if charge <= -3:
        return "neg3plus"
    if charge < 0:
        return f"neg{abs(charge)}"
    if charge == 0:
        return "neutral"
    if charge >= 3:
        return "pos3plus"
    return f"pos{charge}"


def _atom_label(atom: Atom) -> str:
    shell_count, orbital_count, electron_count, shell_signature = _shell_stats(atom)
    return (
        f"{_atom_symbol(atom)}:{_charge_bucket(atom.formal_charge)}:"
        f"sh{shell_count}:orb{orbital_count}:e{electron_count}:{shell_signature}"
    )


def _edge_multiplicity(order: float) -> int:
    if order <= 0.25:
        return 1
    if order < 1.25:
        return 1
    if order < 2.50:
        return 2
    if order < 3.50:
        return 3
    return 4


def _edge_labels_for_mode(molecule: Molecule, edge: Edge, mode: str) -> list[str]:
    pair = _edge_symbol_pair(molecule, edge)
    order = effective_order(molecule, edge)
    order_bucket = _order_bucket(order)
    systems = [system for _, system in molecule.systems]
    containing = [system for system in systems if edge in system.member_edges]

    if mode == "simple_graph_wl":
        return [f"{pair}:adjacency"]
    if mode == "bond_order_graph_wl":
        return [f"{pair}:{order_bucket}"]
    if mode == "multigraph_multiplicity_wl":
        return [f"{pair}:parallel_edge"] * _edge_multiplicity(order)
    if mode == "dietz_edge_wl":
        signature = ".".join(
            sorted(
                f"{system.shared_electrons.value}e:{len(system.member_edges)}m:{_system_kind(system)}"
                for system in containing
            )
        )
        return [f"{pair}:{order_bucket}:overlap{len(containing)}:{signature}"]
    raise ValueError(f"Unsupported graph mode: {mode}")


def _graph_counts(molecule: Molecule, mode: str, radius: int = WL_RADIUS) -> Counter[str]:
    atoms = sorted(molecule.atoms)
    labels = {atom_id: _atom_label(molecule.atoms[atom_id]) for atom_id in atoms}
    counts: Counter[str] = Counter()
    for label in labels.values():
        counts[f"wl0:{label}"] += 1
    if mode == "atom_bag":
        return counts

    adjacency: dict[Any, list[tuple[Any, str]]] = {}
    for edge in sorted(molecule_edges(molecule)):
        for edge_label in _edge_labels_for_mode(molecule, edge, mode):
            counts[f"edge_label:{edge_label}"] += 1
            adjacency.setdefault(edge.a, []).append((edge.b, edge_label))
            adjacency.setdefault(edge.b, []).append((edge.a, edge_label))

    current = labels
    for step in range(1, radius + 1):
        updated: dict[Any, str] = {}
        for atom_id in atoms:
            neighborhood = tuple(
                sorted((edge_label, current[neighbor]) for neighbor, edge_label in adjacency.get(atom_id, ()))
            )
            updated[atom_id] = f"{current[atom_id]}|{neighborhood}"
        current = updated
        for label in current.values():
            counts[f"wl{step}:{label}"] += 1
    return counts


def _empty_counts(_: Molecule) -> Counter[str]:
    return Counter()


def _vectorize_counts(counters: list[Counter[str]]) -> tuple[np.ndarray, tuple[str, ...]]:
    matrix, names = _vectorize(counters)
    if matrix.shape[1] == 0:
        return np.zeros((len(counters), 1), dtype=float), tuple()
    return matrix, tuple(names)


def _tanimoto_kernel(matrix: np.ndarray) -> np.ndarray:
    dot = matrix @ matrix.T
    norm = np.sum(matrix * matrix, axis=1)
    denom = norm[:, None] + norm[None, :] - dot
    return np.nan_to_num(dot / np.maximum(denom, 1.0e-12), nan=0.0, posinf=0.0, neginf=0.0)


def _kernel_components(wl_matrix: np.ndarray, system_matrix: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "wl_system_tanimoto": _tanimoto_kernel(np.hstack([wl_matrix, system_matrix])),
        "system_tanimoto": _tanimoto_kernel(system_matrix),
        "wl_tanimoto": _tanimoto_kernel(wl_matrix),
    }


def _variants() -> tuple[AblationVariant, ...]:
    return (
        AblationVariant(
            code="A",
            label="atom_bag",
            description="atoms only, no connectivity",
            wl_counter_fn=lambda molecule: _graph_counts(molecule, "atom_bag"),
            system_counter_fn=_empty_counts,
        ),
        AblationVariant(
            code="B",
            label="simple_graph_wl",
            description="atoms plus binary adjacency",
            wl_counter_fn=lambda molecule: _graph_counts(molecule, "simple_graph_wl"),
            system_counter_fn=_empty_counts,
        ),
        AblationVariant(
            code="C",
            label="bond_order_graph_wl",
            description="one edge per atom pair with effective bond-order labels",
            wl_counter_fn=lambda molecule: _graph_counts(molecule, "bond_order_graph_wl"),
            system_counter_fn=_empty_counts,
        ),
        AblationVariant(
            code="D",
            label="multigraph_multiplicity_wl",
            description="parallel-edge-style multiplicity from effective bond order",
            wl_counter_fn=lambda molecule: _graph_counts(molecule, "multigraph_multiplicity_wl"),
            system_counter_fn=_empty_counts,
        ),
        AblationVariant(
            code="E",
            label="dietz_edge_wl",
            description="Dietz-derived edge labels without a separate bonding-system view",
            wl_counter_fn=lambda molecule: _graph_counts(molecule, "dietz_edge_wl"),
            system_counter_fn=_empty_counts,
        ),
        AblationVariant(
            code="F",
            label="full_moladt",
            description="Dietz edge labels plus explicit bonding-system token view",
            wl_counter_fn=_graph_token_counts,
            system_counter_fn=_system_token_counts,
        ),
    )


def _evaluate_variant(
    bundle: _Bundle,
    *,
    variant: AblationVariant,
    wl_matrix: np.ndarray,
    system_matrix: np.ndarray,
    wl_token_count: int,
    system_token_count: int,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    components = _kernel_components(wl_matrix, system_matrix)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train_idx, valid_idx, test_idx = train_valid_test_indices(len(bundle.y), int(seed))
        selection_fit = _fit_gp(components, bundle.y, train_idx)
        valid_mean, valid_sd = _predict_gp(selection_fit, valid_idx)
        final_train_idx = np.sort(np.concatenate([train_idx, valid_idx]))
        final_fit = _fit_gp(components, bundle.y, final_train_idx)
        train_mean, train_sd = _predict_gp(final_fit, final_train_idx)
        test_mean, test_sd = _predict_gp(final_fit, test_idx)
        for split, idx, mean, sd, fit in (
            ("train", final_train_idx, train_mean, train_sd, final_fit),
            ("valid", valid_idx, valid_mean, valid_sd, selection_fit),
            ("test", test_idx, test_mean, test_sd, final_fit),
        ):
            metrics = regression_summary(bundle.y[idx], mean, sd)
            rows.append(
                {
                    "variant_code": variant.code,
                    "variant": variant.label,
                    "description": variant.description,
                    "seed": int(seed),
                    "split": split,
                    "n_eval": int(len(idx)),
                    "wl_token_count": int(wl_token_count),
                    "system_token_count": int(system_token_count),
                    "feature_count": int(wl_token_count + system_token_count),
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "mean_log_predictive_density": metrics["mean_log_predictive_density"],
                    "coverage_90": metrics["coverage_90"],
                    "predictive_sd_mean": metrics["predictive_sd_mean"],
                    "nll": fit.nll,
                }
            )
    return rows


def _summary_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby(["variant_code", "variant", "description", "split"], sort=False)
        .agg(
            n_splits=("seed", "nunique"),
            n_eval_mean=("n_eval", "mean"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            mean_log_predictive_density_mean=("mean_log_predictive_density", "mean"),
            coverage_90_mean=("coverage_90", "mean"),
            predictive_sd_mean=("predictive_sd_mean", "mean"),
            wl_token_count=("wl_token_count", "first"),
            system_token_count=("system_token_count", "first"),
            feature_count=("feature_count", "first"),
        )
        .reset_index()
    )
    std_columns = [column for column in summary.columns if column.endswith("_std")]
    summary.loc[:, std_columns] = summary.loc[:, std_columns].fillna(0.0)
    return summary


def _paired_against_full(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics.loc[metrics["split"] == "test"].copy()
    full = test.loc[test["variant"] == "full_moladt", ["seed", "rmse", "mae", "mean_log_predictive_density"]].rename(
        columns={
            "rmse": "full_rmse",
            "mae": "full_mae",
            "mean_log_predictive_density": "full_mlpd",
        }
    )
    rows: list[dict[str, Any]] = []
    for (code, variant, description), frame in test.groupby(["variant_code", "variant", "description"], sort=False):
        if variant == "full_moladt":
            continue
        merged = frame[["seed", "rmse", "mae", "mean_log_predictive_density"]].merge(full, on="seed", how="inner")
        rmse_delta = merged["rmse"] - merged["full_rmse"]
        mae_delta = merged["mae"] - merged["full_mae"]
        rows.append(
            {
                "variant_code": code,
                "variant": variant,
                "description": description,
                "splits": int(len(merged)),
                "full_lower_rmse_splits": int((merged["full_rmse"] < merged["rmse"]).sum()),
                "rmse_variant_minus_full_mean": float(rmse_delta.mean()),
                "rmse_variant_minus_full_std": float(rmse_delta.std(ddof=1)),
                "mae_variant_minus_full_mean": float(mae_delta.mean()),
                "mlpd_full_minus_variant_mean": float((merged["full_mlpd"] - merged["mean_log_predictive_density"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def run_ablation(*, seeds: Sequence[int], output_dir: Path, verbose: bool = False) -> dict[str, Path]:
    start = time.perf_counter()
    rows = pd.read_csv(PROCESSED_DATA_DIR / "freesolv_moladt_featurized_features.csv")
    bundle = _load_bundle_from_rows(rows)
    metric_rows: list[dict[str, Any]] = []
    for variant in _variants():
        wl_counters = [variant.wl_counter_fn(molecule) for molecule in bundle.molecules]
        system_counters = [variant.system_counter_fn(molecule) for molecule in bundle.molecules]
        wl_matrix, wl_names = _vectorize_counts(wl_counters)
        system_matrix, system_names = _vectorize_counts(system_counters)
        if verbose:
            print(
                f"[{variant.code} {variant.label}] "
                f"features={len(wl_names) + len(system_names)} "
                f"wl={len(wl_names)} system={len(system_names)}",
                flush=True,
            )
        metric_rows.extend(
            _evaluate_variant(
                bundle,
                variant=variant,
                wl_matrix=wl_matrix,
                system_matrix=system_matrix,
                wl_token_count=len(wl_names),
                system_token_count=len(system_names),
                seeds=seeds,
            )
        )

    metrics = pd.DataFrame(metric_rows)
    summary = _summary_frame(metrics)
    paired = _paired_against_full(metrics)
    output_dir = ensure_directory(output_dir)
    metrics_path = output_dir / "metrics_by_seed.csv"
    summary_path = output_dir / "summary.csv"
    paired_path = output_dir / "paired_against_full.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    metadata_path = output_dir / "metadata.csv"
    pd.DataFrame(
        [
            {
                "dataset": "freesolv",
                "representation": "moladt",
                "model_family": "representation_ablation",
                "split_count": len(seeds),
                "seed_start": int(seeds[0]) if seeds else "",
                "seed_end": int(seeds[-1]) if seeds else "",
                "runtime_seconds": time.perf_counter() - start,
            }
        ]
    ).to_csv(metadata_path, index=False)
    return {
        "metrics": metrics_path,
        "summary": summary_path,
        "paired_against_full": paired_path,
        "metadata": metadata_path,
    }


def _default_output_dir() -> Path:
    return Path("results") / "freesolv_ablation" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.freesolv_representation_ablation",
        description="Run the FreeSolv A-F representation ablation for the MolADT WL + bonding-system GP.",
    )
    parser.add_argument("--split-count", type=int, default=DEFAULT_SPLIT_COUNT)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split_count <= 0:
        raise ValueError("--split-count must be positive")
    seeds = tuple(range(int(args.seed_start), int(args.seed_start) + int(args.split_count)))
    paths = run_ablation(seeds=seeds, output_dir=args.output_dir, verbose=bool(args.verbose))
    summary = pd.read_csv(paths["summary"])
    test_summary = summary.loc[summary["split"] == "test"].copy()
    print("FreeSolv A-F MolADT representation ablation")
    print(f"  seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} splits)")
    print(f"  output_dir: {args.output_dir}")
    for row in test_summary.itertuples(index=False):
        print(
            f"  {row.variant_code} {row.variant}: "
            f"RMSE {float(row.rmse_mean):.6f} +/- {float(row.rmse_std):.6f} kcal/mol "
            f"features={int(row.feature_count)}"
        )
    print(f"  summary: {paths['summary']}")
    print(f"  paired_against_full: {paths['paired_against_full']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
