from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
from rdkit import Chem

from moladt.chem.molecule import Molecule
from moladt.chem.pretty import pretty_text
from moladt.io.smiles import parse_smiles

from .common import PROCESSED_DATA_DIR, RESULTS_DIR, display_path, ensure_directory, format_progress, log, log_stage
from .download_data import download_zinc
from .features import canonical_smiles_from_mol, rdkit_mol_to_moladt_record


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
    smiles_path: Path
    moladt_dir: Path
    manifest_path: Path


def run_zinc_benchmark(
    *,
    dataset_size: str = "250K",
    dataset_dimension: str = "2D",
    limit: int | None = None,
    include_moladt: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> list[TimingStageResult]:
    total_stages = 7 if include_moladt else 3
    if verbose:
        log(
            "Starting ZINC timing benchmark "
            f"(dataset_size={dataset_size}, dataset_dimension={dataset_dimension}, limit={limit}, include_moladt={include_moladt})"
        )
        log(f"Results directory: {display_path(RESULTS_DIR)}")
        log_stage("zinc", 1, total_stages, "Reading raw source file")
    downloads = download_zinc(dataset_size=dataset_size, dataset_dimension=dataset_dimension, force=force)
    smiles_strings, raw_stage = _load_smiles_with_timing(
        downloads.source_path,
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        limit=limit,
        verbose=verbose,
    )
    if verbose:
        _log_stage_result(raw_stage, stage_index=1, total_stages=total_stages)
        log_stage("zinc", 2, total_stages, f"Parsing and sanitizing SMILES strings (molecules={len(smiles_strings)})")
    parsed_molecules, parse_stage = _measure_stage(
        smiles_strings,
        lambda smiles: Chem.MolFromSmiles(smiles, sanitize=True),
        stage_name="smiles_parse_sanitize",
        stage_description="An external chemistry toolkit parses each SMILES string and sanitizes the molecular graph for the interoperability timing path.",
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        source_path=downloads.source_path,
        keep_successful_outputs=True,
        verbose=verbose,
    )
    if verbose:
        _log_stage_result(parse_stage, stage_index=2, total_stages=total_stages)
        log_stage("zinc", 3, total_stages, f"Canonicalizing parsed molecules (molecules={len(parsed_molecules)})")
    _, canonical_stage = _measure_stage(
        [molecule for molecule in parsed_molecules if molecule is not None],
        lambda molecule: Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        stage_name="smiles_canonicalization",
        stage_description="The interoperability timing path re-renders each parsed molecule as a canonical isomeric SMILES string using an external chemistry toolkit.",
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        source_path=downloads.source_path,
        keep_successful_outputs=False,
        verbose=verbose,
    )
    if verbose:
        _log_stage_result(canonical_stage, stage_index=3, total_stages=total_stages)
    stages = [raw_stage, parse_stage, canonical_stage]
    item_rows: list[TimingItemResult] = []
    library_manifest: pd.DataFrame | None = None
    if include_moladt:
        if verbose:
            log_stage("zinc", 4, total_stages, f"Building local timing library (molecules={len(parsed_molecules)})")
        library, library_stage = _prepare_timing_library(
            molecules=[molecule for molecule in parsed_molecules if molecule is not None],
            dataset_size=dataset_size,
            dataset_dimension=dataset_dimension,
            limit=limit,
            source_path=downloads.source_path,
            force=force,
            verbose=verbose,
            log_label=f"4/{total_stages}",
        )
        stages.append(library_stage)
        if verbose:
            _log_stage_result(library_stage, stage_index=4, total_stages=total_stages)
            log(f"[zinc] timing library root: {display_path(library.library_root)}")
        library_manifest = _read_timing_library_manifest(library)
        if verbose:
            log_stage("zinc", 5, total_stages, f"Materializing manifest CSV SMILES strings (molecules={len(library_manifest)})")
        csv_string_item_rows, csv_string_stage = _measure_smiles_csv_string_parse(
            library,
            manifest=library_manifest,
            dataset_size=dataset_size,
            dataset_dimension=dataset_dimension,
            verbose=verbose,
            log_label=f"5/{total_stages}",
        )
        item_rows.extend(csv_string_item_rows)
        stages.append(csv_string_stage)
        if verbose:
            _log_stage_result(csv_string_stage, stage_index=5, total_stages=total_stages)
            log_stage("zinc", 6, total_stages, f"Parsing local SMILES timing library (molecules={len(library_manifest)})")
        smiles_item_rows, smiles_stage = _measure_smiles_library_parse(
            library,
            manifest=library_manifest,
            dataset_size=dataset_size,
            dataset_dimension=dataset_dimension,
            verbose=verbose,
            log_label=f"6/{total_stages}",
        )
        item_rows.extend(smiles_item_rows)
        stages.append(smiles_stage)
        if verbose:
            _log_stage_result(smiles_stage, stage_index=6, total_stages=total_stages)
            log_stage("zinc", 7, total_stages, f"Parsing local MolADT timing library (molecules={len(library_manifest)})")
        moladt_item_rows, moladt_stage = _measure_moladt_library_parse(
            library,
            manifest=library_manifest,
            dataset_size=dataset_size,
            dataset_dimension=dataset_dimension,
            verbose=verbose,
            log_label=f"7/{total_stages}",
        )
        item_rows.extend(moladt_item_rows)
        stages.append(moladt_stage)
        if verbose:
            _log_stage_result(moladt_stage, stage_index=7, total_stages=total_stages)
    details_dir = ensure_directory(RESULTS_DIR / "details")
    results_frame = pd.DataFrame([stage.to_dict() for stage in stages])
    results_csv = details_dir / "zinc_timing.csv"
    results_frame.to_csv(results_csv, index=False)
    if item_rows:
        item_frame = pd.DataFrame([row.to_dict() for row in item_rows])
        item_frame.to_csv(details_dir / "zinc_timing_items.csv", index=False)
    if library_manifest is not None:
        library_manifest.to_csv(details_dir / "zinc_timing_library_manifest.csv", index=False)
    if verbose:
        log(f"[zinc] wrote timing rows={len(results_frame)} to {display_path(results_csv)}")
    return stages


def _timing_library_root(dataset_size: str, dataset_dimension: str, limit: int | None) -> Path:
    scope = "full" if limit is None else f"limit_{limit}"
    return ensure_directory(PROCESSED_DATA_DIR / "zinc_timing" / f"zinc15_{dataset_size}_{dataset_dimension}" / scope)


def _prepare_timing_library(
    *,
    molecules: list[Chem.Mol],
    dataset_size: str,
    dataset_dimension: str,
    limit: int | None,
    source_path: Path,
    force: bool,
    verbose: bool = False,
    log_label: str = "4/6",
) -> tuple[TimingLibraryPaths, TimingStageResult]:
    library_root = _timing_library_root(dataset_size, dataset_dimension, limit)
    smiles_path = library_root / "smiles_library.smi"
    moladt_dir = ensure_directory(library_root / "moladt_library")
    manifest_path = library_root / "manifest.csv"
    if not force and manifest_path.exists() and smiles_path.exists():
        manifest = pd.read_csv(manifest_path)
        if not manifest.empty and all((library_root / str(path)).exists() for path in manifest["moladt_relative_path"].tolist()):
            stage = TimingStageResult(
                dataset_size=dataset_size,
                dataset_dimension=dataset_dimension,
                stage="timing_library_prepare",
                description="Reused the cached local timing corpus of matched ADT files plus canonical SMILES strings.",
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
            return TimingLibraryPaths(library_root=library_root, smiles_path=smiles_path, moladt_dir=moladt_dir, manifest_path=manifest_path), stage
    records: list[dict[str, Any]] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    latencies: list[float] = []
    success_count = 0
    failure_count = 0
    start_total = time.perf_counter()
    with smiles_path.open("w", encoding="utf-8") as smiles_handle:
        for source_index, molecule in enumerate(molecules, start=1):
            start_item = time.perf_counter_ns()
            try:
                canonical_smiles = canonical_smiles_from_mol(molecule)
                parse_smiles(canonical_smiles)
                moladt_record = rdkit_mol_to_moladt_record(molecule)
                mol_id = f"zinc_{source_index:07d}"
                relative_path = Path("moladt_library") / f"{mol_id}.moladt.json"
                adt_path = library_root / relative_path
                payload = moladt_record.molecule.to_json_bytes()
                adt_path.write_bytes(payload)
                smiles_handle.write(canonical_smiles + "\n")
                success_count += 1
                records.append(
                    {
                        "mol_id": mol_id,
                        "source_index": source_index,
                        "canonical_smiles": canonical_smiles,
                        "smiles_line_number": success_count,
                        "smiles_size_bytes": len(canonical_smiles.encode("utf-8")),
                        "moladt_relative_path": str(relative_path),
                        "moladt_size_bytes": len(payload),
                    }
                )
            except Exception:
                failure_count += 1
            latencies.append((time.perf_counter_ns() - start_item) / 1_000.0)
            if verbose and source_index % 500 == 0:
                peak_rss = max(peak_rss, process.memory_info().rss)
                log(
                    f"[zinc {log_label}] built timing library molecules={format_progress(source_index, len(molecules))} "
                    f"success={success_count} failure={failure_count}"
                )
    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_path, index=False)
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="timing_library_prepare",
        description="Built the local timing corpus: one MolADT JSON file per matched molecule plus a canonical SMILES file with the same molecule count.",
        source_path=str(source_path),
        molecule_count=len(molecules),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(molecules) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return TimingLibraryPaths(library_root=library_root, smiles_path=smiles_path, moladt_dir=moladt_dir, manifest_path=manifest_path), stage


def _read_timing_library_manifest(library: TimingLibraryPaths) -> pd.DataFrame:
    return pd.read_csv(library.manifest_path)


def _measure_smiles_library_parse(
    library: TimingLibraryPaths,
    *,
    manifest: pd.DataFrame,
    dataset_size: str,
    dataset_dimension: str,
    verbose: bool = False,
    log_label: str = "5/6",
) -> tuple[list[TimingItemResult], TimingStageResult]:
    latencies: list[float] = []
    item_rows: list[TimingItemResult] = []
    success_count = 0
    failure_count = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    for row_index, row in enumerate(manifest.itertuples(index=False), start=1):
        smiles = str(row.canonical_smiles)
        start_item = time.perf_counter_ns()
        error = ""
        success = True
        try:
            parse_smiles(smiles)
        except Exception as exc:
            success = False
            error = str(exc)
        latency_us = (time.perf_counter_ns() - start_item) / 1_000.0
        latencies.append(latency_us)
        if success:
            success_count += 1
        else:
            failure_count += 1
        item_rows.append(
            TimingItemResult(
                dataset_size=dataset_size,
                dataset_dimension=dataset_dimension,
                stage="smiles_library_parse",
                mol_id=str(row.mol_id),
                item_kind="smiles_entry",
                item_path=f"{library.smiles_path}#{int(row.smiles_line_number)}",
                item_size_bytes=int(row.smiles_size_bytes),
                success=success,
                latency_us=latency_us,
                error=error,
            )
        )
        if verbose and row_index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc {log_label}] parsed SMILES library molecules={format_progress(row_index, len(manifest))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="smiles_library_parse",
        description="Parsed each matched canonical SMILES string with the local MolADT SMILES parser.",
        source_path=str(library.smiles_path),
        molecule_count=len(item_rows),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(item_rows) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return item_rows, stage


def _measure_moladt_library_parse(
    library: TimingLibraryPaths,
    *,
    manifest: pd.DataFrame,
    dataset_size: str,
    dataset_dimension: str,
    verbose: bool = False,
    log_label: str = "6/6",
) -> tuple[list[TimingItemResult], TimingStageResult]:
    latencies: list[float] = []
    item_rows: list[TimingItemResult] = []
    success_count = 0
    failure_count = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    for row_index, row in enumerate(manifest.itertuples(index=False), start=1):
        relative_path = Path(str(row.moladt_relative_path))
        adt_path = library.library_root / relative_path
        payload = adt_path.read_bytes()
        start_item = time.perf_counter_ns()
        error = ""
        success = True
        try:
            Molecule.from_json(payload)
        except Exception as exc:
            success = False
            error = str(exc)
        latency_us = (time.perf_counter_ns() - start_item) / 1_000.0
        latencies.append(latency_us)
        if success:
            success_count += 1
        else:
            failure_count += 1
        item_rows.append(
            TimingItemResult(
                dataset_size=dataset_size,
                dataset_dimension=dataset_dimension,
                stage="moladt_file_parse",
                mol_id=str(row.mol_id),
                item_kind="moladt_file",
                item_path=str(adt_path),
                item_size_bytes=int(len(payload)),
                success=success,
                latency_us=latency_us,
                error=error,
            )
        )
        if verbose and row_index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc {log_label}] parsed MolADT library molecules={format_progress(row_index, len(manifest))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="moladt_file_parse",
        description="Read each MolADT JSON file and parsed it into the local Molecule ADT using the fast JSON loader when available.",
        source_path=str(library.moladt_dir),
        molecule_count=len(item_rows),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(item_rows) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return item_rows, stage


def _load_smiles_with_timing(
    source_path: Path,
    *,
    dataset_size: str,
    dataset_dimension: str,
    limit: int | None,
    verbose: bool = False,
) -> tuple[list[str], TimingStageResult]:
    smiles_strings: list[str] = []
    latencies: list[float] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    if source_path.suffix == ".csv":
        with source_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            smiles_column = _detect_smiles_column(tuple(reader.fieldnames or ()))
            for row_index, row in enumerate(reader):
                if limit is not None and len(smiles_strings) >= limit:
                    break
                start_item = time.perf_counter_ns()
                smiles = row.get(smiles_column, "").strip()
                if smiles:
                    smiles_strings.append(smiles)
                latencies.append((time.perf_counter_ns() - start_item) / 1_000.0)
                if verbose and row_index % 500 == 0:
                    peak_rss = max(peak_rss, process.memory_info().rss)
                    target_total = limit if limit is not None else 0
                    if target_total > 0:
                        log(f"[zinc] raw rows={format_progress(len(smiles_strings), target_total)}")
                    else:
                        log(f"[zinc] raw rows={len(smiles_strings)}")
    else:
        with source_path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if limit is not None and len(smiles_strings) >= limit:
                    break
                start_item = time.perf_counter_ns()
                smiles = line.strip().split()[0] if line.strip() else ""
                if smiles:
                    smiles_strings.append(smiles)
                latencies.append((time.perf_counter_ns() - start_item) / 1_000.0)
                if verbose and row_index % 500 == 0:
                    peak_rss = max(peak_rss, process.memory_info().rss)
                    target_total = limit if limit is not None else 0
                    if target_total > 0:
                        log(f"[zinc] raw rows={format_progress(len(smiles_strings), target_total)}")
                    else:
                        log(f"[zinc] raw rows={len(smiles_strings)}")
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="raw_file_read",
        description="Read SMILES strings from the normalized ZINC source file without chemistry work.",
        source_path=str(source_path),
        molecule_count=len(smiles_strings),
        success_count=len(smiles_strings),
        failure_count=0,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(smiles_strings) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return smiles_strings, stage


def _csv_smiles_text_to_string(smiles_text: Any) -> str:
    if pd.isna(smiles_text):
        raise ValueError("Missing SMILES text")
    smiles = str(smiles_text).strip()
    if not smiles:
        raise ValueError("Empty SMILES text")
    return smiles


def _measure_smiles_csv_string_parse(
    library: TimingLibraryPaths,
    *,
    manifest: pd.DataFrame,
    dataset_size: str,
    dataset_dimension: str,
    verbose: bool = False,
    log_label: str = "5/7",
) -> tuple[list[TimingItemResult], TimingStageResult]:
    latencies: list[float] = []
    item_rows: list[TimingItemResult] = []
    success_count = 0
    failure_count = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    for row_index, row in enumerate(manifest.itertuples(index=False), start=1):
        start_item = time.perf_counter_ns()
        error = ""
        success = True
        try:
            _csv_smiles_text_to_string(row.canonical_smiles)
        except Exception as exc:
            success = False
            error = str(exc)
        latency_us = (time.perf_counter_ns() - start_item) / 1_000.0
        latencies.append(latency_us)
        if success:
            success_count += 1
        else:
            failure_count += 1
        item_rows.append(
            TimingItemResult(
                dataset_size=dataset_size,
                dataset_dimension=dataset_dimension,
                stage="smiles_csv_string_parse",
                mol_id=str(row.mol_id),
                item_kind="smiles_csv_field",
                item_path=f"{library.manifest_path}#{row_index}",
                item_size_bytes=int(row.smiles_size_bytes),
                success=success,
                latency_us=latency_us,
                error=error,
            )
        )
        if verbose and row_index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc {log_label}] materialized manifest CSV SMILES={format_progress(row_index, len(manifest))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="smiles_csv_string_parse",
        description="Materialized each canonical SMILES field from the matched manifest CSV as a plain Python string without chemistry parsing.",
        source_path=str(library.manifest_path),
        molecule_count=len(item_rows),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(item_rows) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return item_rows, stage


def _measure_stage(
    items: list[Any],
    function: Any,
    *,
    stage_name: str,
    stage_description: str,
    dataset_size: str,
    dataset_dimension: str,
    source_path: Path,
    keep_successful_outputs: bool,
    verbose: bool = False,
) -> tuple[list[Any], TimingStageResult]:
    latencies: list[float] = []
    outputs: list[Any] = []
    success_count = 0
    failure_count = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    for index, item in enumerate(items):
        start_item = time.perf_counter_ns()
        try:
            result = function(item)
            if result is None:
                raise ValueError("Function returned None")
        except Exception:
            failure_count += 1
        else:
            success_count += 1
            if keep_successful_outputs:
                outputs.append(result)
        latencies.append((time.perf_counter_ns() - start_item) / 1_000.0)
        if verbose and index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc] {stage_name} molecules={format_progress(index + 1, len(items))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage=stage_name,
        description=stage_description,
        source_path=str(source_path),
        molecule_count=len(items),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(items) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return outputs, stage


def _moladt_parse_render_from_rdkit_mol(molecule: Chem.Mol) -> str:
    return pretty_text(rdkit_mol_to_moladt_record(molecule).molecule)


def _detect_smiles_column(fieldnames: tuple[str, ...]) -> str:
    for candidate in ("smiles", "SMILES", "smile", "molecule"):
        if candidate in fieldnames:
            return candidate
    if not fieldnames:
        raise ValueError("CSV file has no header row")
    return fieldnames[0]


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
        f"molecules={stage.molecule_count} success={stage.success_count} failure={stage.failure_count} "
        f"runtime_s={stage.total_runtime_seconds:.4f} mol_per_s={stage.molecules_per_second:.2f} "
        f"median_us={stage.median_latency_us:.1f} p95_us={stage.p95_latency_us:.1f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.benchmark_zinc")
    parser.add_argument("--dataset-size", default="250K")
    parser.add_argument("--dataset-dimension", default="2D")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-moladt", action="store_true")
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
