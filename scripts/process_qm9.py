from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase

from .common import DEFAULT_SEED, FailureRecord, PROCESSED_DATA_DIR, ensure_directory, write_failure_csv
from .download_data import download_qm9
from .features import (
    canonical_smiles_from_mol,
    featurize_moladt_geometry_records,
    featurize_moladt_records,
    featurize_sdf_geometry_records,
    featurize_smiles_dataframe,
)
from .splits import ExportedDataset, GeometricDatasetSpec, deterministic_split_partition, export_geometric_splits, export_standardized_splits

QM9_PAPER_TRAIN_SIZE = 110_462
QM9_PAPER_VALID_SIZE = 10_000
QM9_PAPER_TEST_SIZE = 10_000
QM9_PAPER_TOTAL = QM9_PAPER_TRAIN_SIZE + QM9_PAPER_VALID_SIZE + QM9_PAPER_TEST_SIZE


@dataclass(frozen=True, slots=True)
class QM9Artifacts:
    processed_csv_path: Path
    moladt_index_path: Path
    tabular_exports: dict[str, ExportedDataset]
    geometric_exports: dict[str, GeometricDatasetSpec]
    smiles_export: ExportedDataset
    moladt_export: ExportedDataset
    failure_csv_paths: tuple[Path, ...]


@contextmanager
def _block_rdkit_logs():
    blocker = rdBase.BlockLogs()
    try:
        yield
    finally:
        del blocker


def process_qm9_dataset(
    *,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    limit: int | None = None,
    split_mode: str = "subset",
) -> QM9Artifacts:
    if split_mode == "paper" and limit is not None and limit < QM9_PAPER_TOTAL:
        raise ValueError(
            f"QM9 paper split requires at least {QM9_PAPER_TOTAL} aligned molecules; "
            "omit --limit or provide a larger value"
        )
    downloads = download_qm9(force=force)
    ensure_directory(PROCESSED_DATA_DIR)
    with _block_rdkit_logs():
        combined_frame, failures = _build_qm9_aligned_frame(downloads.sdf_path, downloads.csv_path, limit=limit)
    processed_csv_path = PROCESSED_DATA_DIR / "qm9_processed.csv"
    combined_frame.loc[:, ["mol_id", "smiles", "mu"]].to_csv(processed_csv_path, index=False)
    moladt_index_path = PROCESSED_DATA_DIR / "qm9_moladt_index.csv"
    combined_frame.loc[:, ["mol_id", "sdf_record_index", "mu"]].to_csv(moladt_index_path, index=False)
    failure_paths: list[Path] = []
    processing_failure_path = PROCESSED_DATA_DIR / "qm9_processing_failures.csv"
    write_failure_csv(processing_failure_path, failures)
    failure_paths.append(processing_failure_path)
    split_partition = _qm9_split_partition(len(combined_frame), seed=seed, split_mode=split_mode)

    with _block_rdkit_logs():
        smiles_table = featurize_smiles_dataframe(
            combined_frame.loc[:, ["mol_id", "smiles", "mu"]],
            dataset_name="qm9_smiles",
            mol_id_column="mol_id",
            smiles_column="smiles",
            target_column="mu",
        )
    smiles_feature_failure_path = PROCESSED_DATA_DIR / "qm9_smiles_feature_failures.csv"
    write_failure_csv(smiles_feature_failure_path, smiles_table.failures)
    failure_paths.append(smiles_feature_failure_path)
    smiles_export = export_standardized_splits(
        smiles_table,
        dataset_name="qm9",
        representation="smiles",
        target_name="mu",
        seed=seed,
        split_partition=split_partition,
    )
    tabular_exports: dict[str, ExportedDataset] = {"smiles": smiles_export}

    with _block_rdkit_logs():
        moladt_table = featurize_moladt_records(
            combined_frame,
            dataset_name="qm9_moladt",
            mol_id_column="mol_id",
            mol_column="rdkit_mol",
            target_column="mu",
            record_index_column="sdf_record_index",
        )
    moladt_feature_failure_path = PROCESSED_DATA_DIR / "qm9_moladt_feature_failures.csv"
    write_failure_csv(moladt_feature_failure_path, moladt_table.failures)
    failure_paths.append(moladt_feature_failure_path)
    moladt_export = export_standardized_splits(
        moladt_table,
        dataset_name="qm9",
        representation="moladt",
        target_name="mu",
        seed=seed,
        split_partition=split_partition,
    )
    tabular_exports["moladt"] = moladt_export

    with _block_rdkit_logs():
        sdf_geom_table = featurize_sdf_geometry_records(
            combined_frame,
            dataset_name="qm9_sdf_geom",
            mol_id_column="mol_id",
            mol_column="rdkit_mol",
            target_column="mu",
            record_index_column="sdf_record_index",
        )
    sdf_geom_failure_path = PROCESSED_DATA_DIR / "qm9_sdf_geometry_failures.csv"
    write_failure_csv(sdf_geom_failure_path, sdf_geom_table.failures)
    failure_paths.append(sdf_geom_failure_path)
    with _block_rdkit_logs():
        moladt_geom_table = featurize_moladt_geometry_records(
            combined_frame,
            dataset_name="qm9_moladt_geom",
            mol_id_column="mol_id",
            mol_column="rdkit_mol",
            target_column="mu",
            record_index_column="sdf_record_index",
        )
    moladt_geom_failure_path = PROCESSED_DATA_DIR / "qm9_moladt_geometry_failures.csv"
    write_failure_csv(moladt_geom_failure_path, moladt_geom_table.failures)
    failure_paths.append(moladt_geom_failure_path)
    geometric_exports: dict[str, GeometricDatasetSpec] = {}
    if not sdf_geom_table.rows.empty:
        geometric_exports["sdf_geom"] = export_geometric_splits(
            sdf_geom_table,
            dataset_name="qm9",
            representation="sdf_geom",
            target_name="mu",
            seed=seed,
            split_partition=split_partition,
        )
    if not moladt_geom_table.rows.empty:
        geometric_exports["moladt_geom"] = export_geometric_splits(
            moladt_geom_table,
            dataset_name="qm9",
            representation="moladt_geom",
            target_name="mu",
            seed=seed,
            split_partition=split_partition,
        )

    return QM9Artifacts(
        processed_csv_path=processed_csv_path,
        moladt_index_path=moladt_index_path,
        tabular_exports=tabular_exports,
        geometric_exports=geometric_exports,
        smiles_export=smiles_export,
        moladt_export=moladt_export,
        failure_csv_paths=tuple(failure_paths),
    )


def _build_qm9_aligned_frame(sdf_path: Path, csv_path: Path, *, limit: int | None) -> tuple[pd.DataFrame, list[FailureRecord]]:
    targets = pd.read_csv(csv_path)
    if "mu" not in targets.columns:
        raise ValueError("QM9 CSV must contain target column `mu`")
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    rows: list[dict[str, object]] = []
    failures: list[FailureRecord] = []
    for index, molecule in enumerate(supplier):
        if limit is not None and len(rows) >= limit:
            break
        if index >= len(targets):
            failures.append(FailureRecord(dataset="qm9", mol_id=f"qm9_{index + 1}", stage="align_rows", error="SDF has more records than target CSV"))
            break
        target_row = targets.iloc[index]
        mol_id = str(target_row["mol_id"]) if "mol_id" in targets.columns else f"qm9_{index + 1:06d}"
        if molecule is None:
            failures.append(FailureRecord(dataset="qm9", mol_id=mol_id, stage="read_sdf", error="RDKit returned None from SDMolSupplier"))
            continue
        try:
            canonical = canonical_smiles_from_mol(molecule)
        except Exception as exc:
            failures.append(FailureRecord(dataset="qm9", mol_id=mol_id, stage="canonicalize_sdf", error=str(exc)))
            continue
        rows.append(
            {
                "mol_id": mol_id,
                "smiles": canonical,
                "mu": float(target_row["mu"]),
                "sdf_record_index": index,
                "rdkit_mol": molecule,
            }
        )
    if not rows:
        raise ValueError("No QM9 rows were processed successfully")
    return pd.DataFrame(rows), failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.process_qm9")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-mode", choices=("subset", "paper"), default="subset")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = process_qm9_dataset(seed=args.seed, force=args.force, limit=args.limit, split_mode=args.split_mode)
    print(artifacts.processed_csv_path)
    print(artifacts.moladt_index_path)
    return 0


def _qm9_split_partition(row_count: int, *, seed: int, split_mode: str):
    if split_mode == "paper":
        return deterministic_split_partition(
            row_count,
            seed=seed,
            train_size=QM9_PAPER_TRAIN_SIZE,
            valid_size=QM9_PAPER_VALID_SIZE,
            test_size=QM9_PAPER_TEST_SIZE,
            scheme="paper:110462/10000/10000",
        )
    if split_mode != "subset":
        raise ValueError(f"Unsupported QM9 split mode {split_mode}")
    return deterministic_split_partition(row_count, seed=seed, scheme="subset:fractional_0.8/0.1/0.1")


if __name__ == "__main__":
    raise SystemExit(main())
