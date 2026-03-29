from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem

from .common import DEFAULT_SEED, FailureRecord, PROCESSED_DATA_DIR, write_failure_csv
from .download_data import download_qm9
from .features import canonical_smiles_from_mol, featurize_sdf_records, featurize_smiles_dataframe
from .splits import ExportedDataset, export_standardized_splits


@dataclass(frozen=True, slots=True)
class QM9Artifacts:
    processed_csv_path: Path
    sdf_index_path: Path
    smiles_export: ExportedDataset
    sdf_export: ExportedDataset
    failure_csv_paths: tuple[Path, ...]


def process_qm9_dataset(
    *,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    limit: int | None = None,
) -> QM9Artifacts:
    downloads = download_qm9(force=force)
    combined_frame, failures = _build_qm9_aligned_frame(downloads.sdf_path, downloads.csv_path, limit=limit)
    processed_csv_path = PROCESSED_DATA_DIR / "qm9_processed.csv"
    combined_frame.loc[:, ["mol_id", "smiles", "mu"]].to_csv(processed_csv_path, index=False)
    sdf_index_path = PROCESSED_DATA_DIR / "qm9_sdf_index.csv"
    combined_frame.loc[:, ["mol_id", "sdf_record_index", "mu"]].to_csv(sdf_index_path, index=False)
    failure_paths: list[Path] = []
    processing_failure_path = PROCESSED_DATA_DIR / "qm9_processing_failures.csv"
    write_failure_csv(processing_failure_path, failures)
    failure_paths.append(processing_failure_path)

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
    smiles_export = export_standardized_splits(smiles_table, dataset_name="qm9", representation="smiles", target_name="mu", seed=seed)

    sdf_table = featurize_sdf_records(
        combined_frame,
        dataset_name="qm9_sdf",
        mol_id_column="mol_id",
        mol_column="rdkit_mol",
        target_column="mu",
        record_index_column="sdf_record_index",
    )
    sdf_feature_failure_path = PROCESSED_DATA_DIR / "qm9_sdf_feature_failures.csv"
    write_failure_csv(sdf_feature_failure_path, sdf_table.failures)
    failure_paths.append(sdf_feature_failure_path)
    sdf_export = export_standardized_splits(sdf_table, dataset_name="qm9", representation="sdf", target_name="mu", seed=seed)

    return QM9Artifacts(
        processed_csv_path=processed_csv_path,
        sdf_index_path=sdf_index_path,
        smiles_export=smiles_export,
        sdf_export=sdf_export,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = process_qm9_dataset(seed=args.seed, force=args.force, limit=args.limit)
    print(artifacts.processed_csv_path)
    print(artifacts.sdf_index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

