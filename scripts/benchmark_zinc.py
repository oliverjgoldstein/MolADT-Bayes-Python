from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

from moladt.chem.molecule import Molecule
from moladt.chem.validate import validate_molecule
from moladt.io.molecule_json import molecule_from_json, molecule_to_json_bytes
from moladt.io.sdf import molecule_to_sdf, read_sdf_record
from moladt.io.smiles import molecule_to_smiles, parse_smiles

from .common import PROCESSED_DATA_DIR, RESULTS_DIR, display_path, ensure_directory, format_progress, log, log_stage
from .download_data import download_zinc


@dataclass(frozen=True, slots=True)
class TimingStageResult:
    dataset_size: str
    dataset_dimension: str
    stage: str
    description: str
    source_path: str
    molecule_count: int
    success_count: int
    failure_count: int
    total_runtime_seconds: float
    molecules_per_second: float
    median_latency_us: float
    p95_latency_us: float
    peak_rss_mb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TimingItemResult:
    dataset_size: str
    dataset_dimension: str
    stage: str
    mol_id: str
    item_kind: str
    item_path: str
    item_size_bytes: int
    success: bool
    latency_us: float
    error: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TimingLibraryPaths:
    library_root: Path
    smiles_csv_path: Path
    moladt_csv_path: Path
    sdf_dir: Path
    json_dir: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class TimingSourceEntry:
    source_index: int
    mol_id: str
    smiles: str
    item_size_bytes: int


@dataclass(frozen=True, slots=True)
class TimedMoleculeResult:
    mol_id: str
    item_path: Path
    item_size_bytes: int
    molecule: Molecule


@dataclass(frozen=True, slots=True)
class TimedJsonPayload:
    mol_id: str
    item_path: Path
    item_size_bytes: int


def run_zinc_benchmark(
    *,
    dataset_size: str = "250K",
    dataset_dimension: str = "2D",
    limit: int | None = None,
    include_moladt: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> list[TimingStageResult]:
    total_stages = 8
    downloads = download_zinc(dataset_size=dataset_size, dataset_dimension=dataset_dimension, force=force)
    actual_dimension = downloads.dataset_dimension
    library, _ = _prepare_timing_library(
        dataset_size=dataset_size,
        dataset_dimension=actual_dimension,
        limit=limit,
        source_path=downloads.source_path,
        force=force,
        verbose=verbose,
    )
    manifest = _read_timing_library_manifest(library)
    if verbose:
        log(
            "Starting ZINC timing benchmark "
            f"(dataset_size={dataset_size}, dataset_dimension={actual_dimension}, limit={limit})"
        )
        if not include_moladt:
            log("Timing benchmark now runs the fixed eight-stage paper path; --include-moladt is retained only for CLI compatibility.")
        log(f"Results directory: {display_path(RESULTS_DIR)}")
        log(f"[zinc] timing library root: {display_path(library.library_root)}")
        log_stage("zinc", 1, total_stages, f"Reading matched SMILES CSV rows into strings (rows={len(manifest)})")
    source_rows, smiles_stage = _load_smiles_rows_with_timing(
        library.smiles_csv_path,
        dataset_size=dataset_size,
        dataset_dimension=actual_dimension,
        limit=limit,
        verbose=verbose,
    )
    stages = [smiles_stage]
    if verbose:
        _log_stage_result(smiles_stage, stage_index=1, total_stages=total_stages)
        log_stage("zinc", 2, total_stages, f"Reading cached MolADT CSV rows into MolADT (rows={len(manifest)})")
    moladt_csv_item_rows, moladt_csv_to_moladt_stage = _measure_moladt_csv_to_moladt(
        library.moladt_csv_path,
        dataset_size=dataset_size,
        dataset_dimension=actual_dimension,
        verbose=verbose,
    )
    stages.append(moladt_csv_to_moladt_stage)
    if verbose:
        _log_stage_result(moladt_csv_to_moladt_stage, stage_index=2, total_stages=total_stages)
        log_stage("zinc", 3, total_stages, f"Parsing matched SMILES rows and serializing JSON payloads (rows={len(source_rows)})")
    smiles_json_item_rows, smiles_to_json_stage = _measure_smiles_to_json(
        source_rows,
        dataset_size=dataset_size,
        dataset_dimension=actual_dimension,
        source_path=library.smiles_csv_path,
        verbose=verbose,
    )
    stages.append(smiles_to_json_stage)
    if verbose:
        _log_stage_result(smiles_to_json_stage, stage_index=3, total_stages=total_stages)
        log_stage("zinc", 4, total_stages, f"Parsing matched SDF files into MolADT (records={len(manifest)})")
    sdf_molecules, sdf_item_rows, sdf_stage = _measure_sdf_to_moladt(
        library,
        manifest=manifest,
        dataset_size=dataset_size,
        dataset_dimension=actual_dimension,
        verbose=verbose,
    )
    stages.append(sdf_stage)
    if verbose:
        _log_stage_result(sdf_stage, stage_index=4, total_stages=total_stages)
        log_stage("zinc", 5, total_stages, f"Reading cached SDF files and rendering SMILES (records={len(manifest)})")
    sdf_smiles_item_rows, sdf_to_smiles_stage = _measure_sdf_to_smiles(
        library,
        manifest=manifest,
        dataset_size=dataset_size,
        dataset_dimension=actual_dimension,
        verbose=verbose,
    )
    stages.append(sdf_to_smiles_stage)
    if verbose:
        _log_stage_result(sdf_to_smiles_stage, stage_index=5, total_stages=total_stages)
        log_stage("zinc", 6, total_stages, f"Serializing MolADT objects to JSON (records={len(sdf_molecules)})")
    json_payloads, json_item_rows, json_stage = _measure_moladt_to_json(
        sdf_molecules,
        dataset_size=dataset_size,
        dataset_dimension=actual_dimension,
        destination_dir=library.json_dir,
        verbose=verbose,
    )
    stages.append(json_stage)
    if verbose:
        _log_stage_result(json_stage, stage_index=6, total_stages=total_stages)
        log_stage("zinc", 7, total_stages, f"Decoding JSON files back into MolADT (records={len(json_payloads)})")
    json_roundtrip_rows, json_to_moladt_stage = _measure_json_to_moladt(
        json_payloads,
        dataset_size=dataset_size,
        dataset_dimension=actual_dimension,
        verbose=verbose,
    )
    stages.append(json_to_moladt_stage)
    if verbose:
        _log_stage_result(json_to_moladt_stage, stage_index=7, total_stages=total_stages)
        log_stage("zinc", 8, total_stages, f"Decoding JSON files and rendering SMILES (records={len(json_payloads)})")
    json_smiles_rows, json_to_smiles_stage = _measure_json_to_smiles(
        json_payloads,
        dataset_size=dataset_size,
        dataset_dimension=actual_dimension,
        verbose=verbose,
    )
    stages.append(json_to_smiles_stage)
    if verbose:
        _log_stage_result(json_to_smiles_stage, stage_index=8, total_stages=total_stages)

    details_dir = ensure_directory(RESULTS_DIR / "details")
    results_frame = pd.DataFrame([stage.to_dict() for stage in stages])
    results_csv = details_dir / "zinc_timing.csv"
    results_frame.to_csv(results_csv, index=False)
    item_frame = pd.DataFrame(
        [
            row.to_dict()
            for row in moladt_csv_item_rows + smiles_json_item_rows + sdf_item_rows + sdf_smiles_item_rows + json_item_rows + json_roundtrip_rows + json_smiles_rows
        ]
    )
    items_csv = details_dir / "zinc_timing_items.csv"
    if not item_frame.empty:
        item_frame.to_csv(items_csv, index=False)
    manifest_with_json = _attach_json_payload_paths(manifest, library=library, json_payloads=json_payloads)
    manifest_csv = details_dir / "zinc_timing_corpus_manifest.csv"
    manifest_with_json.to_csv(manifest_csv, index=False)
    _write_timing_result_files_report(
        library=library,
        results_csv=results_csv,
        items_csv=items_csv if item_frame.empty or items_csv.exists() else None,
        manifest_csv=manifest_csv,
    )
    if verbose:
        log(f"[zinc] wrote timing rows={len(results_frame)} to {display_path(results_csv)}")
    return stages


def _timing_library_root(dataset_size: str, dataset_dimension: str, limit: int | None) -> Path:
    scope = "full" if limit is None else f"limit_{limit}"
    return ensure_directory(PROCESSED_DATA_DIR / "zinc_timing" / f"zinc15_{dataset_size}_{dataset_dimension}" / scope)


def _prepare_timing_library(
    *,
    dataset_size: str,
    dataset_dimension: str,
    limit: int | None,
    source_path: Path,
    force: bool,
    verbose: bool = False,
    log_label: str = "setup",
) -> tuple[TimingLibraryPaths, TimingStageResult]:
    library_root = _timing_library_root(dataset_size, dataset_dimension, limit)
    smiles_csv_path = library_root / "smiles_library.csv"
    moladt_csv_path = library_root / "moladt_csv_library.csv"
    sdf_dir = library_root / "sdf_library"
    json_dir = library_root / "json_library"
    manifest_path = library_root / "manifest.csv"
    ensure_directory(sdf_dir)
    ensure_directory(json_dir)
    if not force and _timing_library_is_ready(
        smiles_csv_path=smiles_csv_path,
        moladt_csv_path=moladt_csv_path,
        sdf_dir=sdf_dir,
        manifest_path=manifest_path,
    ):
        manifest = pd.read_csv(manifest_path)
        stage = TimingStageResult(
            dataset_size=dataset_size,
            dataset_dimension=dataset_dimension,
            stage="timing_corpus_prepare",
            description="Reused the cached matched timing corpus: one SMILES CSV row, one MolADT CSV row, and one SDF file for each timed molecule.",
            source_path=str(library_root),
            molecule_count=int(len(manifest)),
            success_count=int(len(manifest)),
            failure_count=0,
            total_runtime_seconds=0.0,
            molecules_per_second=0.0,
            median_latency_us=0.0,
            p95_latency_us=0.0,
            peak_rss_mb=psutil.Process().memory_info().rss / (1024.0 * 1024.0),
        )
        return (
            TimingLibraryPaths(
                library_root=library_root,
                smiles_csv_path=smiles_csv_path,
                moladt_csv_path=moladt_csv_path,
                sdf_dir=sdf_dir,
                json_dir=json_dir,
                manifest_path=manifest_path,
            ),
            stage,
        )

    shutil.rmtree(sdf_dir, ignore_errors=True)
    shutil.rmtree(json_dir, ignore_errors=True)
    ensure_directory(sdf_dir)
    ensure_directory(json_dir)

    source_rows = _read_smiles_source_rows(source_path, limit=limit)
    records: list[dict[str, Any]] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    latencies: list[float] = []
    success_count = 0
    failure_count = 0
    start_total = time.perf_counter()
    with moladt_csv_path.open("w", encoding="utf-8", newline="") as moladt_handle:
        moladt_writer = csv.DictWriter(
            moladt_handle,
            fieldnames=["source_index", "mol_id", "moladt_size_bytes", "moladt_json"],
        )
        moladt_writer.writeheader()
        for row_index, entry in enumerate(source_rows, start=1):
            start_item = time.perf_counter_ns()
            try:
                molecule = parse_smiles(entry.smiles)
                payload = molecule_to_json_bytes(molecule)
                moladt_writer.writerow(
                    {
                        "source_index": entry.source_index,
                        "mol_id": entry.mol_id,
                        "moladt_size_bytes": len(payload),
                        "moladt_json": payload.decode("utf-8"),
                    }
                )
                sdf_relative_path = Path("sdf_library") / f"{entry.mol_id}.sdf"
                sdf_path = library_root / sdf_relative_path
                sdf_text = molecule_to_sdf(molecule, title=entry.mol_id, properties={"smiles": entry.smiles})
                sdf_path.write_text(sdf_text, encoding="latin-1")
                records.append(
                    {
                        "source_index": entry.source_index,
                        "mol_id": entry.mol_id,
                        "smiles": entry.smiles,
                        "smiles_size_bytes": entry.item_size_bytes,
                        "sdf_relative_path": str(sdf_relative_path),
                        "sdf_size_bytes": sdf_path.stat().st_size,
                    }
                )
                success_count += 1
            except Exception:
                failure_count += 1
            latencies.append((time.perf_counter_ns() - start_item) / 1_000.0)
            if verbose and row_index % 500 == 0:
                peak_rss = max(peak_rss, process.memory_info().rss)
                log(
                    f"[zinc {log_label}] built timing corpus rows={format_progress(row_index, len(source_rows))} "
                    f"success={success_count} failure={failure_count}"
                )
    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_path, index=False)
    _write_smiles_library_csv(smiles_csv_path, manifest)
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="timing_corpus_prepare",
        description="Built the matched timing corpus used by the eight-stage paper benchmark: a SMILES CSV, a MolADT CSV, and cached SDF files.",
        source_path=str(source_path),
        molecule_count=len(source_rows),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(source_rows) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return (
        TimingLibraryPaths(
            library_root=library_root,
            smiles_csv_path=smiles_csv_path,
            moladt_csv_path=moladt_csv_path,
            sdf_dir=sdf_dir,
            json_dir=json_dir,
            manifest_path=manifest_path,
        ),
        stage,
    )


def _timing_library_is_ready(*, smiles_csv_path: Path, moladt_csv_path: Path, sdf_dir: Path, manifest_path: Path) -> bool:
    if not smiles_csv_path.exists() or not moladt_csv_path.exists() or not manifest_path.exists() or not sdf_dir.exists():
        return False
    try:
        manifest = pd.read_csv(manifest_path)
    except Exception:
        return False
    required_columns = {"mol_id", "smiles", "sdf_relative_path"}
    if manifest.empty or not required_columns.issubset(manifest.columns):
        return False
    return all((manifest_path.parent / str(path)).exists() for path in manifest["sdf_relative_path"].tolist())


def _read_smiles_source_rows(source_path: Path, *, limit: int | None) -> list[TimingSourceEntry]:
    rows: list[TimingSourceEntry] = []
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            smiles = str(raw_row.get("smiles", "")).strip()
            if not smiles:
                continue
            row_index = len(rows) + 1
            mol_id = str(raw_row.get("zinc_id", "")).strip() or f"zinc_{row_index:07d}"
            rows.append(
                TimingSourceEntry(
                    source_index=row_index,
                    mol_id=mol_id,
                    smiles=smiles,
                    item_size_bytes=len(smiles.encode("utf-8")),
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _write_smiles_library_csv(path: Path, manifest: pd.DataFrame) -> None:
    columns = [column for column in ("source_index", "mol_id", "smiles") if column in manifest.columns]
    manifest.loc[:, columns].to_csv(path, index=False)


def _load_smiles_rows_with_timing(
    source_path: Path,
    *,
    dataset_size: str,
    dataset_dimension: str,
    limit: int | None,
    verbose: bool = False,
) -> tuple[list[TimingSourceEntry], TimingStageResult]:
    rows: list[TimingSourceEntry] = []
    latencies: list[float] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        while True:
            start_item = time.perf_counter_ns()
            try:
                row = next(reader)
            except StopIteration:
                break
            smiles = str(row.get("smiles", "")).strip()
            if smiles:
                row_index = len(rows) + 1
                mol_id = str(row.get("mol_id", "")).strip() or f"zinc_{row_index:07d}"
                rows.append(
                    TimingSourceEntry(
                        source_index=int(row.get("source_index") or row_index),
                        mol_id=mol_id,
                        smiles=smiles,
                        item_size_bytes=len(smiles.encode("utf-8")),
                    )
                )
            latencies.append((time.perf_counter_ns() - start_item) / 1_000.0)
            if verbose and rows and len(rows) % 500 == 0:
                peak_rss = max(peak_rss, process.memory_info().rss)
                target_total = limit if limit is not None else 0
                if target_total > 0:
                    log(f"[zinc] matched SMILES rows={format_progress(len(rows), target_total)}")
                else:
                    log(f"[zinc] matched SMILES rows={len(rows)}")
            if limit is not None and len(rows) >= limit:
                break
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="smiles_csv_to_string",
        description="Read matched SMILES CSV rows into Python strings without chemistry parsing.",
        source_path=str(source_path),
        molecule_count=len(rows),
        success_count=len(rows),
        failure_count=0,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(rows) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return rows, stage


def _read_timing_library_manifest(library: TimingLibraryPaths) -> pd.DataFrame:
    return pd.read_csv(library.manifest_path)


def _measure_smiles_to_json(
    source_rows: list[TimingSourceEntry],
    *,
    dataset_size: str,
    dataset_dimension: str,
    source_path: Path,
    verbose: bool = False,
) -> tuple[list[TimingItemResult], TimingStageResult]:
    latencies: list[float] = []
    item_rows: list[TimingItemResult] = []
    success_count = 0
    failure_count = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    for row_index, entry in enumerate(source_rows, start=1):
        start_item = time.perf_counter_ns()
        error = ""
        success = True
        try:
            payload = molecule_to_json_bytes(parse_smiles(entry.smiles))
            payload_size_bytes = len(payload)
        except Exception as exc:
            success = False
            error = str(exc)
            payload_size_bytes = entry.item_size_bytes
        if success:
            success_count += 1
        else:
            failure_count += 1
        latency_us = (time.perf_counter_ns() - start_item) / 1_000.0
        latencies.append(latency_us)
        item_rows.append(
            TimingItemResult(
                dataset_size=dataset_size,
                dataset_dimension=dataset_dimension,
                stage="smiles_to_json",
                mol_id=entry.mol_id,
                item_kind="smiles_row",
                item_path=f"{source_path}:{entry.source_index}",
                item_size_bytes=payload_size_bytes,
                success=success,
                latency_us=latency_us,
                error=error,
            )
        )
        if verbose and row_index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc] smiles_to_json rows={format_progress(row_index, len(source_rows))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="smiles_to_json",
        description="Parse matched SMILES strings into MolADT and serialize the result to JSON payloads.",
        source_path=str(source_path),
        molecule_count=len(source_rows),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(source_rows) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return item_rows, stage


def _measure_moladt_csv_to_moladt(
    source_path: Path,
    *,
    dataset_size: str,
    dataset_dimension: str,
    verbose: bool = False,
) -> tuple[list[TimingItemResult], TimingStageResult]:
    latencies: list[float] = []
    item_rows: list[TimingItemResult] = []
    success_count = 0
    failure_count = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    row_count = 0
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        while True:
            start_item = time.perf_counter_ns()
            try:
                row = next(reader)
            except StopIteration:
                break
            row_count += 1
            mol_id = str(row.get("mol_id", "")).strip() or f"zinc_{row_count:07d}"
            source_index = int(row.get("source_index") or row_count)
            payload_text = str(row.get("moladt_json", ""))
            item_size_bytes = int(row.get("moladt_size_bytes") or len(payload_text.encode("utf-8")))
            error = ""
            success = True
            try:
                molecule_from_json(payload_text.encode("utf-8"))
            except Exception as exc:
                success = False
                error = str(exc)
            if success:
                success_count += 1
            else:
                failure_count += 1
            latency_us = (time.perf_counter_ns() - start_item) / 1_000.0
            latencies.append(latency_us)
            item_rows.append(
                TimingItemResult(
                    dataset_size=dataset_size,
                    dataset_dimension=dataset_dimension,
                    stage="moladt_csv_to_moladt",
                    mol_id=mol_id,
                    item_kind="moladt_csv_row",
                    item_path=f"{source_path}:{source_index}",
                    item_size_bytes=item_size_bytes,
                    success=success,
                    latency_us=latency_us,
                    error=error,
                )
            )
            if verbose and row_count % 500 == 0:
                peak_rss = max(peak_rss, process.memory_info().rss)
                log(f"[zinc] moladt_csv_to_moladt rows={row_count} success={success_count} failure={failure_count}")
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="moladt_csv_to_moladt",
        description="Read cached MolADT CSV rows and decode the embedded MolADT payload into the local typed Molecule object.",
        source_path=str(source_path),
        molecule_count=row_count,
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(row_count / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return item_rows, stage


def _measure_sdf_to_moladt(
    library: TimingLibraryPaths,
    *,
    manifest: pd.DataFrame,
    dataset_size: str,
    dataset_dimension: str,
    verbose: bool = False,
) -> tuple[list[TimedMoleculeResult], list[TimingItemResult], TimingStageResult]:
    latencies: list[float] = []
    parsed_molecules: list[TimedMoleculeResult] = []
    item_rows: list[TimingItemResult] = []
    success_count = 0
    failure_count = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    for row_index, row in enumerate(manifest.itertuples(index=False), start=1):
        relative_path = Path(str(row.sdf_relative_path))
        sdf_path = library.library_root / relative_path
        start_item = time.perf_counter_ns()
        error = ""
        success = True
        try:
            molecule = read_sdf_record(sdf_path).molecule
        except Exception as exc:
            success = False
            error = str(exc)
        else:
            success_count += 1
            parsed_molecules.append(
                TimedMoleculeResult(
                    mol_id=str(row.mol_id),
                    item_path=sdf_path,
                    item_size_bytes=int(getattr(row, "sdf_size_bytes", sdf_path.stat().st_size)),
                    molecule=molecule,
                )
            )
        if not success:
            failure_count += 1
        latency_us = (time.perf_counter_ns() - start_item) / 1_000.0
        latencies.append(latency_us)
        item_rows.append(
            TimingItemResult(
                dataset_size=dataset_size,
                dataset_dimension=dataset_dimension,
                stage="sdf_to_moladt",
                mol_id=str(row.mol_id),
                item_kind="sdf_file",
                item_path=str(sdf_path),
                item_size_bytes=int(getattr(row, "sdf_size_bytes", sdf_path.stat().st_size)),
                success=success,
                latency_us=latency_us,
                error=error,
            )
        )
        if verbose and row_index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc] sdf_to_moladt rows={format_progress(row_index, len(manifest))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="sdf_to_moladt",
        description="Read cached SDF files and parse them into the local typed Molecule object.",
        source_path=str(library.sdf_dir),
        molecule_count=len(manifest),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(manifest) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return parsed_molecules, item_rows, stage


def _measure_sdf_to_smiles(
    library: TimingLibraryPaths,
    *,
    manifest: pd.DataFrame,
    dataset_size: str,
    dataset_dimension: str,
    verbose: bool = False,
) -> tuple[list[TimingItemResult], TimingStageResult]:
    latencies: list[float] = []
    item_rows: list[TimingItemResult] = []
    success_count = 0
    failure_count = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    for row_index, row in enumerate(manifest.itertuples(index=False), start=1):
        relative_path = Path(str(row.sdf_relative_path))
        sdf_path = library.library_root / relative_path
        start_item = time.perf_counter_ns()
        error = ""
        success = True
        try:
            molecule = read_sdf_record(sdf_path).molecule
            validate_molecule(molecule)
            molecule_to_smiles(molecule)
        except Exception as exc:
            success = False
            error = str(exc)
        if success:
            success_count += 1
        else:
            failure_count += 1
        latency_us = (time.perf_counter_ns() - start_item) / 1_000.0
        latencies.append(latency_us)
        item_rows.append(
            TimingItemResult(
                dataset_size=dataset_size,
                dataset_dimension=dataset_dimension,
                stage="sdf_to_smiles",
                mol_id=str(row.mol_id),
                item_kind="sdf_file",
                item_path=str(sdf_path),
                item_size_bytes=int(getattr(row, "sdf_size_bytes", sdf_path.stat().st_size)),
                success=success,
                latency_us=latency_us,
                error=error,
            )
        )
        if verbose and row_index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc] sdf_to_smiles rows={format_progress(row_index, len(manifest))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="sdf_to_smiles",
        description="Read cached SDF files, validate the decoded molecules, and render them into the supported SMILES subset.",
        source_path=str(library.sdf_dir),
        molecule_count=len(manifest),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(manifest) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return item_rows, stage


def _measure_moladt_to_json(
    molecules: list[TimedMoleculeResult],
    *,
    dataset_size: str,
    dataset_dimension: str,
    destination_dir: Path,
    verbose: bool = False,
) -> tuple[list[TimedJsonPayload], list[TimingItemResult], TimingStageResult]:
    ensure_directory(destination_dir)
    latencies: list[float] = []
    written_payloads: list[TimedJsonPayload] = []
    item_rows: list[TimingItemResult] = []
    success_count = 0
    failure_count = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    for row_index, item in enumerate(molecules, start=1):
        json_path = destination_dir / f"{item.mol_id}.moladt.json"
        start_item = time.perf_counter_ns()
        error = ""
        success = True
        json_size_bytes = 0
        try:
            payload = molecule_to_json_bytes(item.molecule)
            json_path.write_bytes(payload)
            json_size_bytes = len(payload)
        except Exception as exc:
            success = False
            error = str(exc)
        else:
            success_count += 1
            written_payloads.append(
                TimedJsonPayload(
                    mol_id=item.mol_id,
                    item_path=json_path,
                    item_size_bytes=json_size_bytes,
                )
            )
        if not success:
            failure_count += 1
        latency_us = (time.perf_counter_ns() - start_item) / 1_000.0
        latencies.append(latency_us)
        item_rows.append(
            TimingItemResult(
                dataset_size=dataset_size,
                dataset_dimension=dataset_dimension,
                stage="moladt_to_json",
                mol_id=item.mol_id,
                item_kind="moladt_object",
                item_path=str(json_path),
                item_size_bytes=json_size_bytes,
                success=success,
                latency_us=latency_us,
                error=error,
            )
        )
        if verbose and row_index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc] moladt_to_json rows={format_progress(row_index, len(molecules))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="moladt_to_json",
        description="Serialize parsed MolADT objects to JSON and write the JSON files used by the final decode stage.",
        source_path=str(destination_dir),
        molecule_count=len(molecules),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(molecules) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return written_payloads, item_rows, stage


def _measure_json_to_moladt(
    payloads: list[TimedJsonPayload],
    *,
    dataset_size: str,
    dataset_dimension: str,
    verbose: bool = False,
) -> tuple[list[TimingItemResult], TimingStageResult]:
    latencies: list[float] = []
    item_rows: list[TimingItemResult] = []
    success_count = 0
    failure_count = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    for row_index, payload in enumerate(payloads, start=1):
        start_item = time.perf_counter_ns()
        error = ""
        success = True
        try:
            molecule_from_json(payload.item_path.read_bytes())
        except Exception as exc:
            success = False
            error = str(exc)
        if success:
            success_count += 1
        else:
            failure_count += 1
        latency_us = (time.perf_counter_ns() - start_item) / 1_000.0
        latencies.append(latency_us)
        item_rows.append(
            TimingItemResult(
                dataset_size=dataset_size,
                dataset_dimension=dataset_dimension,
                stage="json_to_moladt",
                mol_id=payload.mol_id,
                item_kind="json_file",
                item_path=str(payload.item_path),
                item_size_bytes=payload.item_size_bytes,
                success=success,
                latency_us=latency_us,
                error=error,
            )
        )
        if verbose and row_index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc] json_to_moladt rows={format_progress(row_index, len(payloads))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    source_dir = payloads[0].item_path.parent if payloads else RESULTS_DIR
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="json_to_moladt",
        description="Read JSON files and decode them back into the local typed Molecule object.",
        source_path=str(source_dir),
        molecule_count=len(payloads),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(payloads) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return item_rows, stage


def _measure_json_to_smiles(
    payloads: list[TimedJsonPayload],
    *,
    dataset_size: str,
    dataset_dimension: str,
    verbose: bool = False,
) -> tuple[list[TimingItemResult], TimingStageResult]:
    latencies: list[float] = []
    item_rows: list[TimingItemResult] = []
    success_count = 0
    failure_count = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    for row_index, payload in enumerate(payloads, start=1):
        start_item = time.perf_counter_ns()
        error = ""
        success = True
        try:
            molecule = molecule_from_json(payload.item_path.read_bytes())
            validate_molecule(molecule)
            molecule_to_smiles(molecule)
        except Exception as exc:
            success = False
            error = str(exc)
        if success:
            success_count += 1
        else:
            failure_count += 1
        latency_us = (time.perf_counter_ns() - start_item) / 1_000.0
        latencies.append(latency_us)
        item_rows.append(
            TimingItemResult(
                dataset_size=dataset_size,
                dataset_dimension=dataset_dimension,
                stage="json_to_smiles",
                mol_id=payload.mol_id,
                item_kind="json_file",
                item_path=str(payload.item_path),
                item_size_bytes=payload.item_size_bytes,
                success=success,
                latency_us=latency_us,
                error=error,
            )
        )
        if verbose and row_index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc] json_to_smiles rows={format_progress(row_index, len(payloads))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    source_dir = payloads[0].item_path.parent if payloads else RESULTS_DIR
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="json_to_smiles",
        description="Read JSON files, decode them into Molecule values, validate them, and render the supported SMILES subset.",
        source_path=str(source_dir),
        molecule_count=len(payloads),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(payloads) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return item_rows, stage


def _attach_json_payload_paths(
    manifest: pd.DataFrame,
    *,
    library: TimingLibraryPaths,
    json_payloads: list[TimedJsonPayload],
) -> pd.DataFrame:
    if manifest.empty:
        return manifest
    frame = manifest.copy()
    payload_frame = pd.DataFrame(
        [
            {
                "mol_id": payload.mol_id,
                "json_relative_path": str(payload.item_path.relative_to(library.library_root)),
                "json_size_bytes": payload.item_size_bytes,
            }
            for payload in json_payloads
        ]
    )
    if payload_frame.empty:
        frame["json_relative_path"] = ""
        frame["json_size_bytes"] = 0
        return frame
    return frame.merge(payload_frame, on="mol_id", how="left")


def _write_timing_result_files_report(
    *,
    library: TimingLibraryPaths,
    results_csv: Path,
    items_csv: Path | None,
    manifest_csv: Path,
) -> None:
    lines = [
        "Timing result files",
        "",
        f"matched_smiles_csv: {display_path(library.smiles_csv_path)}",
        f"matched_moladt_csv: {display_path(library.moladt_csv_path)}",
        f"matched_sdf_dir: {display_path(library.sdf_dir)}",
        f"serialized_json_dir: {display_path(library.json_dir)}",
        f"timing_library_manifest: {display_path(library.manifest_path)}",
        f"timing_results_csv: {display_path(results_csv)}",
        f"timing_corpus_manifest_csv: {display_path(manifest_csv)}",
    ]
    if items_csv is not None and items_csv.exists():
        lines.append(f"timing_item_rows_csv: {display_path(items_csv)}")
    (RESULTS_DIR / "timing_result_files.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    series = pd.Series(values, dtype=float)
    return float(series.quantile(percentile / 100.0))


def _log_stage_result(stage: TimingStageResult, *, stage_index: int, total_stages: int) -> None:
    log(
        f"[zinc {stage_index}/{total_stages}] {stage.stage}: "
        f"records={stage.molecule_count} success={stage.success_count} failure={stage.failure_count} "
        f"runtime_s={stage.total_runtime_seconds:.4f} mol_per_s={stage.molecules_per_second:.2f} "
        f"median_us={stage.median_latency_us:.1f} p95_us={stage.p95_latency_us:.1f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.benchmark_zinc")
    parser.add_argument("--dataset-size", default="250K")
    parser.add_argument("--dataset-dimension", default="2D")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-moladt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_zinc_benchmark(
        dataset_size=args.dataset_size,
        dataset_dimension=args.dataset_dimension,
        limit=args.limit,
        include_moladt=args.include_moladt,
        force=args.force,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
