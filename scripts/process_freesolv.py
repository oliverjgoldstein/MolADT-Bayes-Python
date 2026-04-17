from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from moladt.io.sdf import iter_sdf_records

from .common import DEFAULT_SEED, FailureRecord, PROCESSED_DATA_DIR, ensure_directory, extract_archive, find_files, log, log_stage, write_failure_csv
from .download_data import FreeSolvDownloads, download_freesolv
from .features import (
    FeatureTable,
    GeometricFeatureTable,
    canonicalize_smiles,
    featurize_moladt_geometry_records,
    featurize_moladt_featurized_records,
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
    moladt_featurized_export: ExportedDataset | None
    failure_csv_paths: tuple[Path, ...]


FREESOLV_BASELINE_TRAIN_SIZE = 513
FREESOLV_BASELINE_VALID_SIZE = 64
FREESOLV_BASELINE_TEST_SIZE = 65
FREESOLV_BASELINE_TOTAL = FREESOLV_BASELINE_TRAIN_SIZE + FREESOLV_BASELINE_VALID_SIZE + FREESOLV_BASELINE_TEST_SIZE


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


def process_freesolv_dataset(
    *,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    include_moladt: bool = True,
    include_legacy_tabular: bool = True,
    verbose: bool = False,
) -> FreeSolvArtifacts:
    total_stages = 4 if include_moladt else 3
    if verbose:
        log_stage("freesolv", 1, total_stages, "Preparing FreeSolv inputs")
    downloads = download_freesolv(force=force)
    ensure_directory(PROCESSED_DATA_DIR)
    if verbose:
        log_stage("freesolv", 2, total_stages, "Importing FreeSolv molecules from SDF files")
    processed_frame, import_failures, source_sdf_count = _load_freesolv_sdf_dataset(downloads)
    if verbose:
        log(
            f"[freesolv 2/{total_stages}] source_sdf_files={source_sdf_count} "
            f"imported_rows={len(processed_frame)} import_failures={len(import_failures)}"
        )
    processed_csv_path = PROCESSED_DATA_DIR / "freesolv_processed.csv"
    processed_frame.drop(columns=["moladt_molecule"], errors="ignore").to_csv(processed_csv_path, index=False)
    failure_paths: list[Path] = []
    import_failure_path = PROCESSED_DATA_DIR / "freesolv_sdf_import_failures.csv"
    write_failure_csv(import_failure_path, import_failures)
    failure_paths.append(import_failure_path)

    stage_three_message = (
        f"Featurizing FreeSolv boundary strings with SDF-backed molecules (molecules={len(processed_frame)})"
        if include_legacy_tabular
        else f"Featurizing FreeSolv MolADT featurized tables from SDF-backed molecules (molecules={len(processed_frame)})"
    )
    if verbose:
        log_stage("freesolv", 3, total_stages, stage_three_message)
    smiles_table: FeatureTable | None = None
    if include_legacy_tabular:
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
    moladt_export: ExportedDataset | None = None
    moladt_featurized_export: ExportedDataset | None = None
    if include_moladt:
        moladt_table: FeatureTable | None = None
        if include_legacy_tabular:
            moladt_table = featurize_moladt_smiles_dataframe(
                processed_frame,
                dataset_name="freesolv_moladt",
                mol_id_column="mol_id",
                smiles_column="smiles",
                target_column="expt",
            )
            moladt_feature_failure_path = PROCESSED_DATA_DIR / "freesolv_moladt_feature_failures.csv"
            write_failure_csv(moladt_feature_failure_path, moladt_table.failures)
            failure_paths.append(moladt_feature_failure_path)
        moladt_featurized_table = featurize_moladt_featurized_records(
            processed_frame,
            dataset_name="freesolv_moladt_featurized",
            mol_id_column="mol_id",
            mol_column="moladt_molecule",
            target_column="expt",
        )
        moladt_featurized_feature_failure_path = PROCESSED_DATA_DIR / "freesolv_moladt_featurized_feature_failures.csv"
        write_failure_csv(moladt_featurized_feature_failure_path, moladt_featurized_table.failures)
        failure_paths.append(moladt_featurized_feature_failure_path)
        if include_legacy_tabular:
            assert smiles_table is not None
            assert moladt_table is not None
            aligned_tabular = _align_feature_tables({"smiles": smiles_table, "moladt": moladt_table})
            smiles_table = aligned_tabular["smiles"]
            moladt_table = aligned_tabular["moladt"]
        split_partition = None
        if not moladt_featurized_table.rows.empty:
            split_partition = _freesolv_split_partition(len(moladt_featurized_table.rows), seed=seed)
        if include_legacy_tabular:
            assert smiles_table is not None
            assert moladt_table is not None
            smiles_export = export_standardized_splits(
                smiles_table,
                dataset_name="freesolv",
                representation="smiles",
                target_name="expt",
                seed=seed,
                split_partition=split_partition,
            )
            moladt_export = export_standardized_splits(
                moladt_table,
                dataset_name="freesolv",
                representation="moladt",
                target_name="expt",
                seed=seed,
                split_partition=split_partition,
            )
        else:
            smiles_export = None
            moladt_export = None
        typed_split_partition = None
        if not moladt_featurized_table.rows.empty:
            typed_split_partition = _freesolv_split_partition(len(moladt_featurized_table.rows), seed=seed)
            moladt_featurized_export = export_standardized_splits(
                moladt_featurized_table,
                dataset_name="freesolv",
                representation="moladt_featurized",
                target_name="expt",
                seed=seed,
                split_partition=typed_split_partition,
            )
        if include_legacy_tabular:
            assert smiles_table is not None
            assert moladt_table is not None
            assert smiles_export is not None
            if verbose:
                log(
                    f"[freesolv 3/{total_stages}] shared_tabular_rows={len(smiles_table.rows)} "
                    f"smiles_failures={len(smiles_table.failures)} moladt_failures={len(moladt_table.failures)} "
                    f"train={len(smiles_export.y_train)} valid={len(smiles_export.y_valid)} test={len(smiles_export.y_test)} "
                    f"(usable_rows_from_sdf={len(smiles_table.rows)}/{source_sdf_count})"
                )
                if moladt_featurized_export is not None:
                    log(
                        f"[freesolv 3/{total_stages}] moladt_featurized_rows={len(moladt_featurized_table.rows)} "
                        f"moladt_featurized_failures={len(moladt_featurized_table.failures)} "
                        f"train={len(moladt_featurized_export.y_train)} valid={len(moladt_featurized_export.y_valid)} "
                        f"test={len(moladt_featurized_export.y_test)} "
                        f"(usable_rows_from_sdf={len(moladt_featurized_table.rows)}/{source_sdf_count})"
                    )
            tabular_exports = {
                "smiles": smiles_export,
                "moladt": moladt_export,
            }
            if moladt_featurized_export is not None:
                tabular_exports["moladt_featurized"] = moladt_featurized_export
        else:
            tabular_exports = {}
            if moladt_featurized_export is not None:
                tabular_exports["moladt_featurized"] = moladt_featurized_export
                smiles_export = moladt_featurized_export
                moladt_export = moladt_featurized_export
            if verbose and moladt_featurized_export is not None:
                log(
                    f"[freesolv 3/{total_stages}] moladt_featurized_rows={len(moladt_featurized_table.rows)} "
                    f"moladt_featurized_failures={len(moladt_featurized_table.failures)} "
                    f"train={len(moladt_featurized_export.y_train)} valid={len(moladt_featurized_export.y_valid)} "
                    f"test={len(moladt_featurized_export.y_test)} "
                    f"(usable_rows_from_sdf={len(moladt_featurized_table.rows)}/{source_sdf_count})"
                )
    else:
        assert smiles_table is not None
        smiles_export = export_standardized_splits(
            smiles_table,
            dataset_name="freesolv",
            representation="smiles",
            target_name="expt",
            seed=seed,
        )
        if verbose:
            log(
                f"[freesolv 3/{total_stages}] smiles_rows={len(smiles_table.rows)} "
                f"smiles_feature_failures={len(smiles_table.failures)} "
                f"train={len(smiles_export.y_train)} valid={len(smiles_export.y_valid)} test={len(smiles_export.y_test)} "
                f"(usable_rows_from_sdf={len(smiles_table.rows)}/{source_sdf_count})"
            )
        tabular_exports = {"smiles": smiles_export}
    geometric_exports: dict[str, GeometricDatasetSpec] = {}

    moladt_index_path: Path | None = None
    if include_moladt:
        if verbose:
            log_stage("freesolv", 4, total_stages, "Exporting geometry-backed branches from SDF inputs")
        sdf_frame = processed_frame.loc[
            :,
            ["mol_id", "smiles", "smiles_canonical", "expt", "sdf_relpath", "sdf_record_index", "moladt_molecule"],
        ].copy()
        if not sdf_frame.empty:
            if verbose:
                log(
                    f"[freesolv 4/{total_stages}] aligned_sdf_records={len(sdf_frame)} "
                    f"sdf_alignment_failures=0"
                )
            moladt_index_path = PROCESSED_DATA_DIR / "freesolv_moladt_index.csv"
            sdf_frame.drop(columns=["moladt_molecule"]).to_csv(moladt_index_path, index=False)
            sdf_geom_table = featurize_sdf_geometry_records(
                sdf_frame,
                dataset_name="freesolv_sdf_geom",
                mol_id_column="mol_id",
                mol_column="moladt_molecule",
                target_column="expt",
                record_index_column="sdf_record_index",
            )
            sdf_geom_failure_path = PROCESSED_DATA_DIR / "freesolv_sdf_geometry_failures.csv"
            write_failure_csv(sdf_geom_failure_path, sdf_geom_table.failures)
            failure_paths.append(sdf_geom_failure_path)
            moladt_geom_table = featurize_moladt_geometry_records(
                sdf_frame,
                dataset_name="freesolv_moladt_geom",
                mol_id_column="mol_id",
                mol_column="moladt_molecule",
                target_column="expt",
                record_index_column="sdf_record_index",
            )
            moladt_geom_failure_path = PROCESSED_DATA_DIR / "freesolv_moladt_geometry_failures.csv"
            write_failure_csv(moladt_geom_failure_path, moladt_geom_table.failures)
            failure_paths.append(moladt_geom_failure_path)
            aligned_geom = _align_geometric_tables({"sdf_geom": sdf_geom_table, "moladt_geom": moladt_geom_table})
            sdf_geom_table = aligned_geom["sdf_geom"]
            moladt_geom_table = aligned_geom["moladt_geom"]
            geom_split_partition = None
            if not sdf_geom_table.rows.empty:
                geom_split_partition = _freesolv_split_partition(len(sdf_geom_table.rows), seed=seed)
            if not sdf_geom_table.rows.empty:
                geometric_exports["sdf_geom"] = export_geometric_splits(
                    sdf_geom_table,
                    dataset_name="freesolv",
                    representation="sdf_geom",
                    target_name="expt",
                    seed=seed,
                    split_partition=geom_split_partition,
                )
                if verbose:
                    geom_export = geometric_exports["sdf_geom"]
                    log(
                        f"[freesolv 4/{total_stages}] sdf_geom_rows={len(sdf_geom_table.rows)} "
                        f"sdf_geom_feature_failures={len(sdf_geom_table.failures)} "
                        f"train={len(geom_export.train_indices)} valid={len(geom_export.valid_indices)} test={len(geom_export.test_indices)}"
                    )
            if not moladt_geom_table.rows.empty:
                geometric_exports["moladt_geom"] = export_geometric_splits(
                    moladt_geom_table,
                    dataset_name="freesolv",
                    representation="moladt_geom",
                    target_name="expt",
                    seed=seed,
                    split_partition=geom_split_partition,
                )
                if verbose:
                    geom_export = geometric_exports["moladt_geom"]
                    log(
                        f"[freesolv 4/{total_stages}] moladt_geom_rows={len(moladt_geom_table.rows)} "
                        f"moladt_geom_feature_failures={len(moladt_geom_table.failures)} "
                        f"train={len(geom_export.train_indices)} valid={len(geom_export.valid_indices)} test={len(geom_export.test_indices)}"
                    )
        else:
            if verbose:
                log(
                    f"[freesolv 4/{total_stages}] aligned_sdf_records=0 "
                    f"sdf_alignment_failures=0"
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
        moladt_featurized_export=moladt_featurized_export,
        failure_csv_paths=tuple(failure_paths),
    )


def _candidate_freesolv_roots(downloads: FreeSolvDownloads) -> tuple[Path, ...]:
    candidates: list[Path] = []
    raw_root = downloads.csv_path.parent
    extract_root = downloads.repo_extract_dir
    for candidate in (
        raw_root,
        raw_root / "FreeSolv-master",
        raw_root / "FreeSolv-master" / "FreeSolv-master",
        extract_root,
        extract_root / "FreeSolv-master",
        extract_root / "FreeSolv-master" / "FreeSolv-master",
    ):
        if candidate.is_dir() and candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _find_freesolv_database_json(downloads: FreeSolvDownloads) -> Path | None:
    direct_candidates: list[Path] = []
    for root in _candidate_freesolv_roots(downloads):
        candidate = root / "database.json"
        if candidate.is_file():
            direct_candidates.append(candidate)
    if direct_candidates:
        direct_candidates.sort(key=lambda path: (len(path.parts), path.as_posix()))
        return direct_candidates[0]
    recursive_candidates: list[Path] = []
    for root in _candidate_freesolv_roots(downloads):
        recursive_candidates.extend(find_files(root, ("database.json",)))
    if recursive_candidates:
        recursive_candidates = sorted(dict.fromkeys(recursive_candidates), key=lambda path: (len(path.parts), path.as_posix()))
        return recursive_candidates[0]
    return None


def _freesolv_sdf_preference_key(path: Path) -> tuple[int, int, int, str]:
    text = "/".join(part.lower() for part in path.parts[-3:])
    has_3d_conformer = any(token in text for token in ("3d", "conformer"))
    has_v3000 = "v3000" in text
    return (
        0 if has_3d_conformer else 1,
        0 if has_v3000 else 1,
        len(path.parts),
        text,
    )


def _find_freesolv_sdf_paths(downloads: FreeSolvDownloads) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for root in _candidate_freesolv_roots(downloads):
        candidates.extend(sorted(root.rglob("*.sdf")))
    if not candidates:
        raise FileNotFoundError("Could not find FreeSolv SDF records")
    selected_by_compound_id: dict[str, Path] = {}
    for path in sorted(dict.fromkeys(candidates), key=lambda candidate: candidate.as_posix()):
        compound_id = path.stem
        current = selected_by_compound_id.get(compound_id)
        if current is None or _freesolv_sdf_preference_key(path) < _freesolv_sdf_preference_key(current):
            selected_by_compound_id[compound_id] = path
    return tuple(sorted(selected_by_compound_id.values(), key=lambda path: path.stem))


def _freesolv_sdf_relpath(downloads: FreeSolvDownloads, sdf_path: Path) -> str:
    relative_paths: list[Path] = []
    for root in _candidate_freesolv_roots(downloads):
        try:
            relative_paths.append(sdf_path.relative_to(root))
        except ValueError:
            continue
    if relative_paths:
        relative_paths.sort(key=lambda path: (len(path.parts), path.as_posix()))
        return str(relative_paths[0])
    return sdf_path.name


def _scaled_split_sizes_from_baseline(row_count: int) -> tuple[int, int, int]:
    if row_count <= 2:
        raise ValueError("At least three rows are required for train/valid/test splitting")
    baseline_sizes = (
        FREESOLV_BASELINE_TRAIN_SIZE,
        FREESOLV_BASELINE_VALID_SIZE,
        FREESOLV_BASELINE_TEST_SIZE,
    )
    raw_sizes = tuple(row_count * size / FREESOLV_BASELINE_TOTAL for size in baseline_sizes)
    split_sizes = [max(1, int(raw_size)) for raw_size in raw_sizes]
    total_assigned = sum(split_sizes)
    if total_assigned > row_count:
        while total_assigned > row_count:
            largest_index = max(range(3), key=lambda index: (split_sizes[index], baseline_sizes[index]))
            if split_sizes[largest_index] <= 1:
                break
            split_sizes[largest_index] -= 1
            total_assigned -= 1
    if total_assigned < row_count:
        remainders = [raw_size - int(raw_size) for raw_size in raw_sizes]
        for index in sorted(range(3), key=lambda value: (remainders[value], baseline_sizes[value]), reverse=True):
            if total_assigned >= row_count:
                break
            split_sizes[index] += 1
            total_assigned += 1
    return split_sizes[0], split_sizes[1], split_sizes[2]


def _freesolv_split_partition(row_count: int, *, seed: int = DEFAULT_SEED):
    from .splits import deterministic_split_partition

    train_size, valid_size, test_size = _scaled_split_sizes_from_baseline(row_count)
    if row_count == FREESOLV_BASELINE_TOTAL:
        split_scheme = (
            f"moleculenet_random_like:{FREESOLV_BASELINE_TRAIN_SIZE}/"
            f"{FREESOLV_BASELINE_VALID_SIZE}/{FREESOLV_BASELINE_TEST_SIZE}"
        )
    else:
        split_scheme = (
            f"moleculenet_random_like_scaled:{train_size}/{valid_size}/{test_size}"
            f"_from_{FREESOLV_BASELINE_TRAIN_SIZE}/{FREESOLV_BASELINE_VALID_SIZE}/{FREESOLV_BASELINE_TEST_SIZE}"
        )
    return deterministic_split_partition(
        row_count,
        seed=seed,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
        scheme=split_scheme,
    )


def _load_freesolv_metadata(downloads: FreeSolvDownloads) -> dict[str, dict[str, object]]:
    database_path = _find_freesolv_database_json(downloads)
    if database_path is None:
        raise FileNotFoundError("Could not find FreeSolv database.json alongside the SDF files")
    payload = json.loads(database_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("FreeSolv database.json must decode to an object keyed by compound id")
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def _load_freesolv_sdf_dataset(
    downloads: FreeSolvDownloads,
) -> tuple[pd.DataFrame, list[FailureRecord], int]:
    sdf_paths = list(_find_freesolv_sdf_paths(downloads))
    database_path = _find_freesolv_database_json(downloads)
    if database_path is None:
        raise FileNotFoundError(
            "FreeSolv database.json is missing. The benchmark requires the FreeSolv metadata archive so SDF molecules "
            "can be joined directly to targets instead of falling back to SMILES recovery."
        )

    payload = json.loads(database_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("FreeSolv database.json must decode to an object keyed by compound id")
    metadata_by_id = {str(key): value for key, value in payload.items() if isinstance(value, dict)}
    rows: list[dict[str, object]] = []
    failures: list[FailureRecord] = []
    for sdf_path in sdf_paths:
        compound_id = sdf_path.stem
        metadata = metadata_by_id.get(compound_id)
        if metadata is None:
            failures.append(
                FailureRecord(
                    dataset="freesolv",
                    mol_id=compound_id,
                    stage="load_sdf_metadata",
                    error="No matching FreeSolv metadata row for SDF compound id",
                )
            )
            continue
        try:
            record_iterator = iter_sdf_records(sdf_path)
            record = next(record_iterator)
        except StopIteration:
            failures.append(FailureRecord(dataset="freesolv", mol_id=compound_id, stage="read_sdf", error="SDF file had no records"))
            continue
        except Exception as exc:
            failures.append(FailureRecord(dataset="freesolv", mol_id=compound_id, stage="read_sdf", error=str(exc)))
            continue
        source_smiles = str(metadata.get("smiles", "")).strip()
        if not source_smiles:
            failures.append(
                FailureRecord(
                    dataset="freesolv",
                    mol_id=compound_id,
                    stage="load_sdf_metadata",
                    error="FreeSolv metadata row had an empty SMILES string",
                )
            )
            continue
        try:
            canonical_smiles = canonicalize_smiles(source_smiles)
        except Exception as exc:
            failures.append(FailureRecord(dataset="freesolv", mol_id=compound_id, stage="canonicalize_metadata_smiles", error=str(exc)))
            continue
        iupac = str(metadata.get("iupac") or metadata.get("nickname") or "").strip()
        expt = metadata.get("expt")
        try:
            expt_value = float(expt)
        except Exception as exc:
            failures.append(
                FailureRecord(
                    dataset="freesolv",
                    mol_id=compound_id,
                    stage="load_sdf_metadata",
                    error=f"Invalid experimental value {expt!r}: {exc}",
                )
            )
            continue
        rows.append(
            {
                "mol_id": compound_id,
                "compound_id": compound_id,
                "smiles": source_smiles,
                "smiles_canonical": canonical_smiles,
                "source_smiles": source_smiles,
                "iupac": iupac,
                "expt": expt_value,
                "sdf_relpath": _freesolv_sdf_relpath(downloads, sdf_path),
                "sdf_record_index": 0,
                "moladt_molecule": record.molecule,
            }
        )
    return pd.DataFrame(rows).sort_values("mol_id").reset_index(drop=True), failures, len(sdf_paths)


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
