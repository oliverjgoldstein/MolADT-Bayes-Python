from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem

from .common import DEFAULT_SEED, FailureRecord, PROCESSED_DATA_DIR, find_files, write_failure_csv
from .download_data import FreeSolvDownloads, download_freesolv
from .features import canonical_smiles_from_mol, canonicalize_smiles, featurize_sdf_records, featurize_smiles_dataframe
from .splits import ExportedDataset, export_standardized_splits


@dataclass(frozen=True, slots=True)
class FreeSolvArtifacts:
    processed_csv_path: Path
    sdf_index_path: Path | None
    smiles_export: ExportedDataset
    sdf_export: ExportedDataset | None
    failure_csv_paths: tuple[Path, ...]


def process_freesolv_dataset(
    *,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    include_sdf: bool = True,
) -> FreeSolvArtifacts:
    downloads = download_freesolv(force=force)
    processed_frame, canonical_failures = _canonicalize_freesolv_csv(downloads)
    processed_csv_path = PROCESSED_DATA_DIR / "freesolv_processed.csv"
    processed_frame.to_csv(processed_csv_path, index=False)
    failure_paths: list[Path] = []
    canonical_failure_path = PROCESSED_DATA_DIR / "freesolv_processing_failures.csv"
    write_failure_csv(canonical_failure_path, canonical_failures)
    failure_paths.append(canonical_failure_path)

    smiles_table = featurize_smiles_dataframe(
        processed_frame,
        dataset_name="freesolv_smiles",
        mol_id_column="mol_id",
        smiles_column="smiles",
        target_column="expt",
    )
    smiles_feature_failure_path = PROCESSED_DATA_DIR / "freesolv_smiles_feature_failures.csv"
    write_failure_csv(smiles_feature_failure_path, smiles_table.failures)
    failure_paths.append(smiles_feature_failure_path)
    smiles_export = export_standardized_splits(smiles_table, dataset_name="freesolv", representation="smiles", target_name="expt", seed=seed)

    sdf_index_path: Path | None = None
    sdf_export: ExportedDataset | None = None
    if include_sdf:
        sdf_frame, sdf_failures = _align_freesolv_sdf(downloads, processed_frame)
        sdf_failure_path = PROCESSED_DATA_DIR / "freesolv_sdf_alignment_failures.csv"
        write_failure_csv(sdf_failure_path, sdf_failures)
        failure_paths.append(sdf_failure_path)
        if not sdf_frame.empty:
            sdf_index_path = PROCESSED_DATA_DIR / "freesolv_sdf_index.csv"
            sdf_frame.drop(columns=["rdkit_mol"]).to_csv(sdf_index_path, index=False)
            sdf_table = featurize_sdf_records(
                sdf_frame,
                dataset_name="freesolv_sdf",
                mol_id_column="mol_id",
                mol_column="rdkit_mol",
                target_column="expt",
                record_index_column="sdf_record_index",
            )
            sdf_feature_failure_path = PROCESSED_DATA_DIR / "freesolv_sdf_feature_failures.csv"
            write_failure_csv(sdf_feature_failure_path, sdf_table.failures)
            failure_paths.append(sdf_feature_failure_path)
            sdf_export = export_standardized_splits(sdf_table, dataset_name="freesolv", representation="sdf", target_name="expt", seed=seed)
    return FreeSolvArtifacts(
        processed_csv_path=processed_csv_path,
        sdf_index_path=sdf_index_path,
        smiles_export=smiles_export,
        sdf_export=sdf_export,
        failure_csv_paths=tuple(failure_paths),
    )


def _canonicalize_freesolv_csv(downloads: FreeSolvDownloads) -> tuple[pd.DataFrame, list[FailureRecord]]:
    raw = pd.read_csv(downloads.csv_path)
    if "smiles" not in raw.columns or "expt" not in raw.columns:
        raise ValueError("FreeSolv SAMPL.csv must contain `smiles` and `expt` columns")
    rows: list[dict[str, object]] = []
    failures: list[FailureRecord] = []
    for index, record in raw.iterrows():
        mol_id = f"freesolv_{index + 1:04d}"
        try:
            canonical = canonicalize_smiles(str(record["smiles"]))
        except Exception as exc:
            failures.append(FailureRecord(dataset="freesolv", mol_id=mol_id, stage="canonicalize_smiles", error=str(exc)))
            continue
        rows.append({"mol_id": mol_id, "smiles": canonical, "expt": float(record["expt"])})
    return pd.DataFrame(rows), failures


def _align_freesolv_sdf(downloads: FreeSolvDownloads, processed_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[FailureRecord]]:
    sdf_paths = find_files(downloads.repo_extract_dir, ("*.sdf",))
    if not sdf_paths:
        return pd.DataFrame(), []
    by_smiles: dict[str, list[dict[str, object]]] = {}
    for record in processed_frame.to_dict(orient="records"):
        by_smiles.setdefault(str(record["smiles"]), []).append(record)
    aligned_rows: list[dict[str, object]] = []
    failures: list[FailureRecord] = []
    for sdf_path in sdf_paths:
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        for record_index, molecule in enumerate(supplier):
            mol_id = f"{sdf_path.stem}:{record_index}"
            if molecule is None:
                failures.append(FailureRecord(dataset="freesolv", mol_id=mol_id, stage="read_sdf", error="RDKit returned None from SDMolSupplier"))
                continue
            try:
                canonical = canonical_smiles_from_mol(molecule)
            except Exception as exc:
                failures.append(FailureRecord(dataset="freesolv", mol_id=mol_id, stage="canonicalize_sdf", error=str(exc)))
                continue
            matches = by_smiles.get(canonical, [])
            if len(matches) == 1:
                match = matches[0]
                aligned_rows.append(
                    {
                        "mol_id": str(match["mol_id"]),
                        "smiles": canonical,
                        "expt": float(match["expt"]),
                        "sdf_relpath": str(sdf_path.relative_to(downloads.repo_extract_dir)),
                        "sdf_record_index": record_index,
                        "rdkit_mol": molecule,
                    }
                )
            elif len(matches) > 1:
                failures.append(FailureRecord(dataset="freesolv", mol_id=mol_id, stage="align_sdf", error=f"Ambiguous canonical SMILES match: {canonical}"))
    if not aligned_rows:
        return pd.DataFrame(), failures
    deduplicated = pd.DataFrame(aligned_rows).sort_values(["mol_id", "sdf_relpath", "sdf_record_index"]).drop_duplicates(subset=["mol_id"], keep="first")
    return deduplicated.reset_index(drop=True), failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.process_freesolv")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-sdf", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = process_freesolv_dataset(seed=args.seed, force=args.force, include_sdf=not args.skip_sdf)
    print(artifacts.processed_csv_path)
    if artifacts.sdf_index_path is not None:
        print(artifacts.sdf_index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

