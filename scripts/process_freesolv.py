from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem

from .common import DEFAULT_SEED, FailureRecord, PROCESSED_DATA_DIR, ensure_directory, extract_archive, find_files, log, log_stage, write_failure_csv
from .download_data import FreeSolvDownloads, download_freesolv
from .features import (
    canonical_smiles_from_mol,
    canonicalize_smiles,
    featurize_moladt_geometry_records,
    featurize_moladt_records,
    featurize_moladt_typed_geometry_records,
    featurize_moladt_typed_records,
    featurize_moladt_smiles_dataframe,
    featurize_sdf_geometry_records,
    featurize_smiles_dataframe,
)
from .splits import ExportedDataset, GeometricDatasetSpec, export_geometric_splits, export_standardized_splits


@dataclass(frozen=True, slots=True)
class FreeSolvArtifacts:
    processed_csv_path: Path
    moladt_index_path: Path | None
    tabular_exports: dict[str, ExportedDataset]
    geometric_exports: dict[str, GeometricDatasetSpec]
    smiles_export: ExportedDataset
    moladt_export: ExportedDataset | None
    failure_csv_paths: tuple[Path, ...]


def process_freesolv_dataset(
    *,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    include_moladt: bool = True,
    verbose: bool = False,
) -> FreeSolvArtifacts:
    total_stages = 4 if include_moladt else 3
    if verbose:
        log_stage("freesolv", 1, total_stages, "Preparing FreeSolv inputs")
    downloads = download_freesolv(force=force)
    ensure_directory(PROCESSED_DATA_DIR)
    if verbose:
        log_stage("freesolv", 2, total_stages, "Canonicalizing FreeSolv CSV")
    processed_frame, canonical_failures = _canonicalize_freesolv_csv(downloads)
    if verbose:
        log(
            f"[freesolv 2/{total_stages}] canonicalized molecules={len(processed_frame)} "
            f"csv_failures={len(canonical_failures)}"
        )
    processed_csv_path = PROCESSED_DATA_DIR / "freesolv_processed.csv"
    processed_frame.to_csv(processed_csv_path, index=False)
    failure_paths: list[Path] = []
    canonical_failure_path = PROCESSED_DATA_DIR / "freesolv_processing_failures.csv"
    write_failure_csv(canonical_failure_path, canonical_failures)
    failure_paths.append(canonical_failure_path)

    if verbose:
        log_stage(
            "freesolv",
            3,
            total_stages,
            f"Featurizing FreeSolv SMILES records (molecules={len(processed_frame)})",
        )
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
    if verbose:
        log(
            f"[freesolv 3/{total_stages}] smiles_rows={len(smiles_table.rows)} "
            f"smiles_feature_failures={len(smiles_table.failures)} "
            f"train={len(smiles_export.y_train)} valid={len(smiles_export.y_valid)} test={len(smiles_export.y_test)}"
        )
    tabular_exports: dict[str, ExportedDataset] = {"smiles": smiles_export}
    geometric_exports: dict[str, GeometricDatasetSpec] = {}

    moladt_index_path: Path | None = None
    moladt_export: ExportedDataset | None = None
    if include_moladt:
        if verbose:
            log_stage("freesolv", 4, total_stages, "Aligning SDF records and building MolADT feature tables")
        sdf_frame, sdf_failures = _align_freesolv_sdf(downloads, processed_frame)
        sdf_failure_path = PROCESSED_DATA_DIR / "freesolv_sdf_alignment_failures.csv"
        write_failure_csv(sdf_failure_path, sdf_failures)
        failure_paths.append(sdf_failure_path)
        if not sdf_frame.empty:
            if verbose:
                log(
                    f"[freesolv 4/{total_stages}] aligned_sdf_records={len(sdf_frame)} "
                    f"sdf_alignment_failures={len(sdf_failures)}"
                )
            moladt_index_path = PROCESSED_DATA_DIR / "freesolv_moladt_index.csv"
            sdf_frame.drop(columns=["rdkit_mol"]).to_csv(moladt_index_path, index=False)
            moladt_table = featurize_moladt_records(
                sdf_frame,
                dataset_name="freesolv_moladt",
                mol_id_column="mol_id",
                mol_column="rdkit_mol",
                target_column="expt",
                record_index_column="sdf_record_index",
            )
            moladt_feature_failure_path = PROCESSED_DATA_DIR / "freesolv_moladt_feature_failures.csv"
            write_failure_csv(moladt_feature_failure_path, moladt_table.failures)
            failure_paths.append(moladt_feature_failure_path)
            moladt_export = export_standardized_splits(
                moladt_table,
                dataset_name="freesolv",
                representation="moladt",
                target_name="expt",
                seed=seed,
            )
            tabular_exports["moladt"] = moladt_export
            if verbose:
                log(
                    f"[freesolv 4/{total_stages}] moladt_rows={len(moladt_table.rows)} "
                    f"moladt_feature_failures={len(moladt_table.failures)} "
                    f"train={len(moladt_export.y_train)} valid={len(moladt_export.y_valid)} test={len(moladt_export.y_test)}"
                )
            moladt_typed_table = featurize_moladt_typed_records(
                sdf_frame,
                dataset_name="freesolv_moladt_typed",
                mol_id_column="mol_id",
                mol_column="rdkit_mol",
                target_column="expt",
                record_index_column="sdf_record_index",
            )
            moladt_typed_failure_path = PROCESSED_DATA_DIR / "freesolv_moladt_typed_feature_failures.csv"
            write_failure_csv(moladt_typed_failure_path, moladt_typed_table.failures)
            failure_paths.append(moladt_typed_failure_path)
            if not moladt_typed_table.rows.empty:
                tabular_exports["moladt_typed"] = export_standardized_splits(
                    moladt_typed_table,
                    dataset_name="freesolv",
                    representation="moladt_typed",
                    target_name="expt",
                    seed=seed,
                )
                if verbose:
                    typed_export = tabular_exports["moladt_typed"]
                    log(
                        f"[freesolv 4/{total_stages}] moladt_typed_rows={len(moladt_typed_table.rows)} "
                        f"moladt_typed_feature_failures={len(moladt_typed_table.failures)} "
                        f"train={len(typed_export.y_train)} valid={len(typed_export.y_valid)} test={len(typed_export.y_test)}"
                    )
            sdf_geom_table = featurize_sdf_geometry_records(
                sdf_frame,
                dataset_name="freesolv_sdf_geom",
                mol_id_column="mol_id",
                mol_column="rdkit_mol",
                target_column="expt",
                record_index_column="sdf_record_index",
            )
            sdf_geom_failure_path = PROCESSED_DATA_DIR / "freesolv_sdf_geometry_failures.csv"
            write_failure_csv(sdf_geom_failure_path, sdf_geom_table.failures)
            failure_paths.append(sdf_geom_failure_path)
            if not sdf_geom_table.rows.empty:
                geometric_exports["sdf_geom"] = export_geometric_splits(
                    sdf_geom_table,
                    dataset_name="freesolv",
                    representation="sdf_geom",
                    target_name="expt",
                    seed=seed,
                )
                if verbose:
                    geom_export = geometric_exports["sdf_geom"]
                    log(
                        f"[freesolv 4/{total_stages}] sdf_geom_rows={len(sdf_geom_table.rows)} "
                        f"sdf_geom_feature_failures={len(sdf_geom_table.failures)} "
                        f"train={len(geom_export.train_indices)} valid={len(geom_export.valid_indices)} test={len(geom_export.test_indices)}"
                    )
            moladt_geom_table = featurize_moladt_geometry_records(
                sdf_frame,
                dataset_name="freesolv_moladt_geom",
                mol_id_column="mol_id",
                mol_column="rdkit_mol",
                target_column="expt",
                record_index_column="sdf_record_index",
            )
            moladt_geom_failure_path = PROCESSED_DATA_DIR / "freesolv_moladt_geometry_failures.csv"
            write_failure_csv(moladt_geom_failure_path, moladt_geom_table.failures)
            failure_paths.append(moladt_geom_failure_path)
            if not moladt_geom_table.rows.empty:
                geometric_exports["moladt_geom"] = export_geometric_splits(
                    moladt_geom_table,
                    dataset_name="freesolv",
                    representation="moladt_geom",
                    target_name="expt",
                    seed=seed,
                )
                if verbose:
                    geom_export = geometric_exports["moladt_geom"]
                    log(
                        f"[freesolv 4/{total_stages}] moladt_geom_rows={len(moladt_geom_table.rows)} "
                        f"moladt_geom_feature_failures={len(moladt_geom_table.failures)} "
                        f"train={len(geom_export.train_indices)} valid={len(geom_export.valid_indices)} test={len(geom_export.test_indices)}"
                    )
            moladt_typed_geom_table = featurize_moladt_typed_geometry_records(
                sdf_frame,
                dataset_name="freesolv_moladt_typed_geom",
                mol_id_column="mol_id",
                mol_column="rdkit_mol",
                target_column="expt",
                record_index_column="sdf_record_index",
            )
            moladt_typed_geom_failure_path = PROCESSED_DATA_DIR / "freesolv_moladt_typed_geometry_failures.csv"
            write_failure_csv(moladt_typed_geom_failure_path, moladt_typed_geom_table.failures)
            failure_paths.append(moladt_typed_geom_failure_path)
            if not moladt_typed_geom_table.rows.empty:
                geometric_exports["moladt_typed_geom"] = export_geometric_splits(
                    moladt_typed_geom_table,
                    dataset_name="freesolv",
                    representation="moladt_typed_geom",
                    target_name="expt",
                    seed=seed,
                )
                if verbose:
                    geom_export = geometric_exports["moladt_typed_geom"]
                    log(
                        f"[freesolv 4/{total_stages}] moladt_typed_geom_rows={len(moladt_typed_geom_table.rows)} "
                        f"moladt_typed_geom_feature_failures={len(moladt_typed_geom_table.failures)} "
                        f"train={len(geom_export.train_indices)} valid={len(geom_export.valid_indices)} test={len(geom_export.test_indices)}"
                    )
        else:
            if verbose:
                log(
                    f"[freesolv 4/{total_stages}] aligned_sdf_records=0 "
                    f"sdf_alignment_failures={len(sdf_failures)}; falling back to SMILES-derived MolADT features"
                )
            moladt_table = featurize_moladt_smiles_dataframe(
                processed_frame,
                dataset_name="freesolv_moladt_smiles_fallback",
                mol_id_column="mol_id",
                smiles_column="smiles",
                target_column="expt",
            )
            moladt_feature_failure_path = PROCESSED_DATA_DIR / "freesolv_moladt_feature_failures.csv"
            write_failure_csv(moladt_feature_failure_path, moladt_table.failures)
            failure_paths.append(moladt_feature_failure_path)
            if not moladt_table.rows.empty:
                moladt_export = export_standardized_splits(
                    moladt_table,
                    dataset_name="freesolv",
                    representation="moladt",
                    target_name="expt",
                    seed=seed,
                )
                tabular_exports["moladt"] = moladt_export
                if verbose:
                    log(
                        f"[freesolv 4/{total_stages}] moladt_smiles_fallback_rows={len(moladt_table.rows)} "
                        f"moladt_feature_failures={len(moladt_table.failures)} "
                        f"train={len(moladt_export.y_train)} valid={len(moladt_export.y_valid)} test={len(moladt_export.y_test)}"
                    )
    if verbose:
        tabular_keys = ", ".join(sorted(tabular_exports))
        geometric_keys = ", ".join(sorted(geometric_exports)) or "(none)"
        log(f"[freesolv] prepared exports: tabular={tabular_keys}; geometric={geometric_keys}")
    return FreeSolvArtifacts(
        processed_csv_path=processed_csv_path,
        moladt_index_path=moladt_index_path,
        tabular_exports=tabular_exports,
        geometric_exports=geometric_exports,
        smiles_export=smiles_export,
        moladt_export=moladt_export,
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
        sdf_archives = find_files(downloads.repo_extract_dir, ("sdffiles*.tar.gz", "*sdffiles*.tgz"))
        for archive_path in sdf_archives:
            extracted_dir = extract_archive(
                archive_path,
                archive_path.parent / archive_path.name.removesuffix(".tar.gz").removesuffix(".tgz"),
            )
            sdf_paths.extend(find_files(extracted_dir, ("*.sdf",)))
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
    parser.add_argument("--skip-moladt", dest="skip_moladt", action="store_true")
    parser.add_argument("--skip-sdf", dest="skip_moladt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = process_freesolv_dataset(
        seed=args.seed,
        force=args.force,
        include_moladt=not args.skip_moladt,
        verbose=args.verbose,
    )
    print(artifacts.processed_csv_path)
    if artifacts.moladt_index_path is not None:
        print(artifacts.moladt_index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
