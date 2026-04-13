from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from moladt.io.sdf import iter_sdf_records
from moladt.io.smiles import molecule_to_smiles

from .common import DEFAULT_SEED, FailureRecord, PROCESSED_DATA_DIR, ensure_directory, log, log_stage, write_failure_csv
from .download_data import download_qm9
from .features import (
    FeatureTable,
    GeometricFeatureTable,
    featurize_moladt_featurized_records,
    featurize_moladt_geometry_records,
    featurize_moladt_smiles_dataframe,
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
    moladt_featurized_export: ExportedDataset | None
    failure_csv_paths: tuple[Path, ...]

def _align_feature_tables(tables: dict[str, FeatureTable]) -> dict[str, FeatureTable]:
    non_empty = {name: table for name, table in tables.items() if not table.rows.empty}
    if not non_empty:
        return tables
    common_ids = set.intersection(*(set(table.rows["mol_id"].astype(str)) for table in non_empty.values()))
    aligned: dict[str, FeatureTable] = {}
    for name, table in tables.items():
        if table.rows.empty:
            aligned[name] = table
            continue
        rows = table.rows.copy()
        rows.loc[:, "mol_id"] = rows["mol_id"].astype(str)
        rows = rows.loc[rows["mol_id"].isin(common_ids)].reset_index(drop=True)
        aligned[name] = FeatureTable(
            rows=rows,
            feature_names=table.feature_names,
            feature_groups=table.feature_groups,
            failures=table.failures,
        )
    return aligned


def _align_geometric_tables(tables: dict[str, GeometricFeatureTable]) -> dict[str, GeometricFeatureTable]:
    non_empty = {name: table for name, table in tables.items() if not table.rows.empty}
    if not non_empty:
        return tables
    common_ids = set.intersection(*(set(table.rows["mol_id"].astype(str)) for table in non_empty.values()))
    aligned: dict[str, GeometricFeatureTable] = {}
    for name, table in tables.items():
        if table.rows.empty:
            aligned[name] = table
            continue
        rows = table.rows.copy()
        rows.loc[:, "mol_id"] = rows["mol_id"].astype(str)
        mask = rows["mol_id"].isin(common_ids).to_numpy(dtype=bool)
        filtered_rows = rows.loc[mask].reset_index(drop=True)
        indices = [index for index, keep in enumerate(mask.tolist()) if keep]
        filtered_atomic_numbers = tuple(table.atomic_numbers[index] for index in indices)
        filtered_coordinates = tuple(table.coordinates[index] for index in indices)
        filtered_global_features = None
        if table.global_features is not None:
            filtered_global_features = table.global_features[mask]
        aligned[name] = GeometricFeatureTable(
            rows=filtered_rows,
            atomic_numbers=filtered_atomic_numbers,
            coordinates=filtered_coordinates,
            global_feature_names=table.global_feature_names,
            global_feature_groups=table.global_feature_groups,
            global_features=filtered_global_features,
            failures=table.failures,
        )
    return aligned


def process_qm9_dataset(
    *,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    limit: int | None = None,
    split_mode: str = "long",
    include_legacy_tabular: bool = True,
    verbose: bool = False,
) -> QM9Artifacts:
    if split_mode == "paper" and limit is not None and limit < QM9_PAPER_TOTAL:
        raise ValueError(
            f"QM9 paper split requires at least {QM9_PAPER_TOTAL} aligned molecules; "
            "omit --limit or provide a larger value"
        )
    total_stages = 5 if include_legacy_tabular else 4
    if verbose:
        log_stage(
            "qm9",
            1,
            total_stages,
            f"Preparing QM9 inputs (limit={limit if limit is not None else 'full'}, split_mode={split_mode})",
        )
    downloads = download_qm9(force=force)
    ensure_directory(PROCESSED_DATA_DIR)
    combined_frame, failures = _build_qm9_aligned_frame(
        downloads.sdf_path,
        downloads.csv_path,
        limit=limit,
        progress_total_stages=total_stages,
        verbose=verbose,
    )
    if verbose:
        log(
            f"[qm9 1/{total_stages}] aligned_rows={len(combined_frame)} "
            f"processing_failures={len(failures)}"
        )
    processed_csv_path = PROCESSED_DATA_DIR / "qm9_processed.csv"
    combined_frame.loc[:, ["mol_id", "smiles", "mu"]].to_csv(processed_csv_path, index=False)
    moladt_index_path = PROCESSED_DATA_DIR / "qm9_moladt_index.csv"
    combined_frame.loc[:, ["mol_id", "sdf_record_index", "mu"]].to_csv(moladt_index_path, index=False)
    failure_paths: list[Path] = []
    processing_failure_path = PROCESSED_DATA_DIR / "qm9_processing_failures.csv"
    write_failure_csv(processing_failure_path, failures)
    failure_paths.append(processing_failure_path)
    smiles_table: FeatureTable | None = None
    moladt_table: FeatureTable | None = None
    stage_three_index = 3 if include_legacy_tabular else 2
    if include_legacy_tabular:
        if verbose:
            log_stage("qm9", 2, total_stages, f"Featurizing QM9 SMILES records (rows={len(combined_frame)})")
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

        if verbose:
            log_stage("qm9", 3, total_stages, f"Featurizing QM9 MolADT-from-SMILES tables (rows={len(combined_frame)})")
        moladt_table = featurize_moladt_smiles_dataframe(
            combined_frame.loc[:, ["mol_id", "smiles", "mu"]],
            dataset_name="qm9_moladt",
            mol_id_column="mol_id",
            smiles_column="smiles",
            target_column="mu",
        )
        moladt_feature_failure_path = PROCESSED_DATA_DIR / "qm9_moladt_feature_failures.csv"
        write_failure_csv(moladt_feature_failure_path, moladt_table.failures)
        failure_paths.append(moladt_feature_failure_path)
    elif verbose:
        log_stage("qm9", stage_three_index, total_stages, f"Featurizing QM9 MolADT featurized tables from SDF-backed records (rows={len(combined_frame)})")

    moladt_featurized_table = featurize_moladt_featurized_records(
        combined_frame.loc[:, ["mol_id", "mu", "sdf_record_index", "moladt_molecule"]],
        dataset_name="qm9_moladt_featurized",
        mol_id_column="mol_id",
        mol_column="moladt_molecule",
        target_column="mu",
        record_index_column="sdf_record_index",
    )
    moladt_featurized_feature_failure_path = PROCESSED_DATA_DIR / "qm9_moladt_featurized_feature_failures.csv"
    write_failure_csv(moladt_featurized_feature_failure_path, moladt_featurized_table.failures)
    failure_paths.append(moladt_featurized_feature_failure_path)
    if include_legacy_tabular:
        assert smiles_table is not None
        assert moladt_table is not None
        aligned_tabular = _align_feature_tables(
            {
                "smiles": smiles_table,
                "moladt": moladt_table,
                "moladt_featurized": moladt_featurized_table,
            }
        )
        smiles_table = aligned_tabular["smiles"]
        moladt_table = aligned_tabular["moladt"]
        moladt_featurized_table = aligned_tabular["moladt_featurized"]
    split_partition = _qm9_split_partition(len(moladt_featurized_table.rows), seed=seed, split_mode=split_mode)
    if include_legacy_tabular and verbose:
        assert smiles_table is not None
        log(
            f"[qm9 3/{total_stages}] shared_tabular_rows={len(smiles_table.rows)} "
            f"split_train={len(split_partition.train_indices)} "
            f"split_valid={len(split_partition.valid_indices)} "
            f"split_test={len(split_partition.test_indices)}"
        )
    if include_legacy_tabular:
        assert smiles_table is not None
        assert moladt_table is not None
        smiles_export = export_standardized_splits(
            smiles_table,
            dataset_name="qm9",
            representation="smiles",
            target_name="mu",
            seed=seed,
            split_partition=split_partition,
        )
        moladt_export = export_standardized_splits(
            moladt_table,
            dataset_name="qm9",
            representation="moladt",
            target_name="mu",
            seed=seed,
            split_partition=split_partition,
        )
    else:
        smiles_export = None
        moladt_export = None
    moladt_featurized_export = export_standardized_splits(
        moladt_featurized_table,
        dataset_name="qm9",
        representation="moladt_featurized",
        target_name="mu",
        seed=seed,
        split_partition=split_partition,
    )
    if include_legacy_tabular:
        assert smiles_export is not None
        assert moladt_export is not None
        tabular_exports: dict[str, ExportedDataset] = {
            "smiles": smiles_export,
            "moladt": moladt_export,
            "moladt_featurized": moladt_featurized_export,
        }
        if verbose:
            assert smiles_table is not None
            assert moladt_table is not None
            log(
                f"[qm9 2/{total_stages}] smiles_rows={len(smiles_table.rows)} "
                f"smiles_feature_failures={len(smiles_table.failures)} "
                f"train={len(smiles_export.y_train)} valid={len(smiles_export.y_valid)} test={len(smiles_export.y_test)}"
            )
            log(
                f"[qm9 3/{total_stages}] moladt_rows={len(moladt_table.rows)} "
                f"moladt_feature_failures={len(moladt_table.failures)} "
                f"train={len(moladt_export.y_train)} valid={len(moladt_export.y_valid)} test={len(moladt_export.y_test)}"
            )
            log(
                f"[qm9 {stage_three_index}/{total_stages}] moladt_featurized_rows={len(moladt_featurized_table.rows)} "
                f"moladt_featurized_failures={len(moladt_featurized_table.failures)} "
                f"train={len(moladt_featurized_export.y_train)} valid={len(moladt_featurized_export.y_valid)} test={len(moladt_featurized_export.y_test)}"
            )
    else:
        tabular_exports = {"moladt_featurized": moladt_featurized_export}
        smiles_export = moladt_featurized_export
        moladt_export = moladt_featurized_export
        if verbose:
            log(
                f"[qm9 {stage_three_index}/{total_stages}] moladt_featurized_rows={len(moladt_featurized_table.rows)} "
                f"moladt_featurized_failures={len(moladt_featurized_table.failures)} "
                f"train={len(moladt_featurized_export.y_train)} valid={len(moladt_featurized_export.y_valid)} test={len(moladt_featurized_export.y_test)}"
            )

    geometry_stage_index = 4 if include_legacy_tabular else 3
    prepared_stage_index = 5 if include_legacy_tabular else 4
    if verbose:
        log_stage("qm9", geometry_stage_index, total_stages, f"Featurizing QM9 geometry tables (rows={len(combined_frame)})")
    sdf_geom_table = featurize_sdf_geometry_records(
        combined_frame,
        dataset_name="qm9_sdf_geom",
        mol_id_column="mol_id",
        mol_column="moladt_molecule",
        target_column="mu",
        record_index_column="sdf_record_index",
    )
    sdf_geom_failure_path = PROCESSED_DATA_DIR / "qm9_sdf_geometry_failures.csv"
    write_failure_csv(sdf_geom_failure_path, sdf_geom_table.failures)
    failure_paths.append(sdf_geom_failure_path)
    moladt_geom_table = featurize_moladt_geometry_records(
        combined_frame,
        dataset_name="qm9_moladt_geom",
        mol_id_column="mol_id",
        mol_column="moladt_molecule",
        target_column="mu",
        record_index_column="sdf_record_index",
    )
    moladt_geom_failure_path = PROCESSED_DATA_DIR / "qm9_moladt_geometry_failures.csv"
    write_failure_csv(moladt_geom_failure_path, moladt_geom_table.failures)
    failure_paths.append(moladt_geom_failure_path)
    aligned_geom = _align_geometric_tables({"sdf_geom": sdf_geom_table, "moladt_geom": moladt_geom_table})
    sdf_geom_table = aligned_geom["sdf_geom"]
    moladt_geom_table = aligned_geom["moladt_geom"]
    geometric_exports: dict[str, GeometricDatasetSpec] = {}
    geom_row_count = len(sdf_geom_table.rows) if not sdf_geom_table.rows.empty else len(moladt_geom_table.rows)
    geom_split_partition = _qm9_split_partition(geom_row_count, seed=seed, split_mode=split_mode) if geom_row_count else None
    if not sdf_geom_table.rows.empty:
        geometric_exports["sdf_geom"] = export_geometric_splits(
            sdf_geom_table,
            dataset_name="qm9",
            representation="sdf_geom",
            target_name="mu",
            seed=seed,
            split_partition=geom_split_partition,
        )
        if verbose:
            geom_export = geometric_exports["sdf_geom"]
            log(
                f"[qm9 {geometry_stage_index}/{total_stages}] sdf_geom_rows={len(sdf_geom_table.rows)} "
                f"sdf_geom_feature_failures={len(sdf_geom_table.failures)} "
                f"train={len(geom_export.train_indices)} valid={len(geom_export.valid_indices)} test={len(geom_export.test_indices)}"
            )
    if not moladt_geom_table.rows.empty:
        geometric_exports["moladt_geom"] = export_geometric_splits(
            moladt_geom_table,
            dataset_name="qm9",
            representation="moladt_geom",
            target_name="mu",
            seed=seed,
            split_partition=geom_split_partition,
        )
        if verbose:
            geom_export = geometric_exports["moladt_geom"]
            log(
                f"[qm9 {geometry_stage_index}/{total_stages}] moladt_geom_rows={len(moladt_geom_table.rows)} "
                f"moladt_geom_feature_failures={len(moladt_geom_table.failures)} "
                f"train={len(geom_export.train_indices)} valid={len(geom_export.valid_indices)} test={len(geom_export.test_indices)}"
            )
    if verbose:
        tabular_keys = ", ".join(sorted(tabular_exports))
        geometric_keys = ", ".join(sorted(geometric_exports)) or "(none)"
        log_stage("qm9", prepared_stage_index, total_stages, f"Prepared exports: tabular={tabular_keys}; geometric={geometric_keys}")

    return QM9Artifacts(
        processed_csv_path=processed_csv_path,
        moladt_index_path=moladt_index_path,
        tabular_exports=tabular_exports,
        geometric_exports=geometric_exports,
        smiles_export=smiles_export,
        moladt_export=moladt_export,
        moladt_featurized_export=moladt_featurized_export,
        failure_csv_paths=tuple(failure_paths),
    )


def _build_qm9_aligned_frame(
    sdf_path: Path,
    csv_path: Path,
    *,
    limit: int | None,
    progress_total_stages: int = 5,
    verbose: bool = False,
) -> tuple[pd.DataFrame, list[FailureRecord]]:
    targets = pd.read_csv(csv_path)
    if "mu" not in targets.columns:
        raise ValueError("QM9 CSV must contain target column `mu`")
    rows: list[dict[str, object]] = []
    failures: list[FailureRecord] = []
    for index, record in enumerate(iter_sdf_records(sdf_path, limit=limit)):
        if limit is not None and len(rows) >= limit:
            break
        if index >= len(targets):
            failures.append(FailureRecord(dataset="qm9", mol_id=f"qm9_{index + 1}", stage="align_rows", error="SDF has more records than target CSV"))
            break
        target_row = targets.iloc[index]
        mol_id = str(target_row["mol_id"]) if "mol_id" in targets.columns else f"qm9_{index + 1:06d}"
        try:
            smiles = str(target_row["smiles"]) if "smiles" in targets.columns else molecule_to_smiles(record.molecule)
        except Exception as exc:
            failures.append(FailureRecord(dataset="qm9", mol_id=mol_id, stage="render_smiles", error=str(exc)))
            continue
        rows.append(
            {
                "mol_id": mol_id,
                "smiles": smiles,
                "mu": float(target_row["mu"]),
                "sdf_record_index": index,
                "moladt_molecule": record.molecule,
            }
        )
        if verbose and len(rows) % 5000 == 0:
            target_total = min(limit, len(targets)) if limit is not None else len(targets)
            log(
                f"[qm9 1/{progress_total_stages}] aligned_rows={len(rows)}/{target_total} "
                f"processing_failures={len(failures)}"
            )
    if not rows:
        raise ValueError("No QM9 rows were processed successfully")
    return pd.DataFrame(rows), failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.process_qm9")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-mode", choices=("subset", "paper", "long"), default="long")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = process_qm9_dataset(
        seed=args.seed,
        force=args.force,
        limit=args.limit,
        split_mode=args.split_mode,
        verbose=args.verbose,
    )
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
    if split_mode == "long":
        return deterministic_split_partition(
            row_count,
            seed=seed,
            scheme="long:fractional_0.8/0.1/0.1",
        )
    if split_mode != "subset":
        raise ValueError(f"Unsupported QM9 split mode {split_mode}")
    return deterministic_split_partition(row_count, seed=seed, scheme="subset:fractional_0.8/0.1/0.1")


if __name__ == "__main__":
    raise SystemExit(main())
