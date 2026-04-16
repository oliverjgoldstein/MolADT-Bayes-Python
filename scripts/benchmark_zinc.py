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

from moladt.io.molecule_json import molecule_from_json, molecule_to_json_bytes
from moladt.io.smiles import parse_smiles

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
    moladt_dir: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class TimingSourceEntry:
    source_index: int
    mol_id: str
    smiles: str
    item_size_bytes: int


@dataclass(frozen=True, slots=True)
class ParsedSmilesEntry:
    source_index: int
    mol_id: str
    smiles: str
    item_size_bytes: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class LoadedMoladtPayload:
    mol_id: str
    item_path: Path
    item_size_bytes: int
    payload: bytes


def run_zinc_benchmark(
    *,
    dataset_size: str = "250K",
    dataset_dimension: str = "2D",
    limit: int | None = None,
    include_moladt: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> list[TimingStageResult]:
    total_stages = 4
    downloads = download_zinc(dataset_size=dataset_size, dataset_dimension=dataset_dimension, force=force)
    actual_dimension = downloads.dataset_dimension
    if verbose:
        log(
            "Starting ZINC timing benchmark "
            f"(dataset_size={dataset_size}, dataset_dimension={actual_dimension}, limit={limit})"
        )
        if not include_moladt:
            log("Timing benchmark now always runs the four SMILES-vs-MolADT paper stages; --include-moladt is retained only for CLI compatibility.")
        log(f"Results directory: {display_path(RESULTS_DIR)}")
        log_stage("zinc", 1, total_stages, "Reading source SMILES rows")
    source_rows, smiles_read_stage = _load_smiles_rows_with_timing(
        downloads.source_path,
        dataset_size=dataset_size,
        dataset_dimension=actual_dimension,
        limit=limit,
        verbose=verbose,
    )
    stages = [smiles_read_stage]
    if verbose:
        _log_stage_result(smiles_read_stage, stage_index=1, total_stages=total_stages)
        log_stage("zinc", 2, total_stages, f"Parsing SMILES strings into MolADT (rows={len(source_rows)})")
    parsed_entries, smiles_item_rows, smiles_parse_stage = _measure_smiles_parse(
        source_rows,
        dataset_size=dataset_size,
        dataset_dimension=actual_dimension,
        source_path=downloads.source_path,
        verbose=verbose,
    )
    stages.append(smiles_parse_stage)
    if verbose:
        _log_stage_result(smiles_parse_stage, stage_index=2, total_stages=total_stages)
        log(f"[zinc setup] ensuring matched MolADT timing corpus for {len(parsed_entries)} parsed SMILES rows")
    library, _ = _prepare_timing_library(
        entries=parsed_entries,
        dataset_size=dataset_size,
        dataset_dimension=actual_dimension,
        limit=limit,
        source_path=downloads.source_path,
        force=force,
        verbose=verbose,
    )
    manifest = _read_timing_library_manifest(library)
    if verbose:
        log(f"[zinc] timing library root: {display_path(library.library_root)}")
        log_stage("zinc", 3, total_stages, f"Reading cached MolADT JSON payloads (records={len(manifest)})")
    loaded_payloads, moladt_read_stage = _measure_moladt_library_read(
        library,
        manifest=manifest,
        dataset_size=dataset_size,
        dataset_dimension=actual_dimension,
        verbose=verbose,
    )
    stages.append(moladt_read_stage)
    if verbose:
        _log_stage_result(moladt_read_stage, stage_index=3, total_stages=total_stages)
        log_stage("zinc", 4, total_stages, f"Decoding cached MolADT JSON payloads (records={len(loaded_payloads)})")
    moladt_item_rows, moladt_parse_stage = _measure_moladt_library_parse(
        loaded_payloads,
        dataset_size=dataset_size,
        dataset_dimension=actual_dimension,
        source_dir=library.moladt_dir,
        verbose=verbose,
    )
    stages.append(moladt_parse_stage)
    if verbose:
        _log_stage_result(moladt_parse_stage, stage_index=4, total_stages=total_stages)

    details_dir = ensure_directory(RESULTS_DIR / "details")
    results_frame = pd.DataFrame([stage.to_dict() for stage in stages])
    results_csv = details_dir / "zinc_timing.csv"
    results_frame.to_csv(results_csv, index=False)
    item_frame = pd.DataFrame([row.to_dict() for row in smiles_item_rows + moladt_item_rows])
    if not item_frame.empty:
        item_frame.to_csv(details_dir / "zinc_timing_items.csv", index=False)
    manifest.to_csv(details_dir / "zinc_timing_library_manifest.csv", index=False)
    if verbose:
        log(f"[zinc] wrote timing rows={len(results_frame)} to {display_path(results_csv)}")
    return stages


def _timing_library_root(dataset_size: str, dataset_dimension: str, limit: int | None) -> Path:
    scope = "full" if limit is None else f"limit_{limit}"
    return ensure_directory(PROCESSED_DATA_DIR / "zinc_timing" / f"zinc15_{dataset_size}_{dataset_dimension}" / scope)


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
                mol_id = str(row.get("zinc_id", "")).strip() or f"zinc_{row_index:07d}"
                rows.append(
                    TimingSourceEntry(
                        source_index=row_index,
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
                    log(f"[zinc] source SMILES rows={format_progress(len(rows), target_total)}")
                else:
                    log(f"[zinc] source SMILES rows={len(rows)}")
            if limit is not None and len(rows) >= limit:
                break
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="smiles_csv_read",
        description="Read SMILES rows from the normalized ZINC CSV without chemistry parsing.",
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


def _measure_smiles_parse(
    rows: list[TimingSourceEntry],
    *,
    dataset_size: str,
    dataset_dimension: str,
    source_path: Path,
    verbose: bool = False,
) -> tuple[list[ParsedSmilesEntry], list[TimingItemResult], TimingStageResult]:
    latencies: list[float] = []
    parsed_entries: list[ParsedSmilesEntry] = []
    item_rows: list[TimingItemResult] = []
    success_count = 0
    failure_count = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    for row_index, row in enumerate(rows, start=1):
        start_item = time.perf_counter_ns()
        error = ""
        success = True
        try:
            payload = molecule_to_json_bytes(parse_smiles(row.smiles))
        except Exception as exc:
            success = False
            error = str(exc)
        else:
            success_count += 1
            parsed_entries.append(
                ParsedSmilesEntry(
                    source_index=row.source_index,
                    mol_id=row.mol_id,
                    smiles=row.smiles,
                    item_size_bytes=row.item_size_bytes,
                    payload=payload,
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
                stage="smiles_parse",
                mol_id=row.mol_id,
                item_kind="smiles_string",
                item_path=f"{source_path}#{row.source_index}",
                item_size_bytes=row.item_size_bytes,
                success=success,
                latency_us=latency_us,
                error=error,
            )
        )
        if verbose and row_index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc] smiles_parse rows={format_progress(row_index, len(rows))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="smiles_parse",
        description="Parsed each source SMILES string into the local MolADT object with the project SMILES reader.",
        source_path=str(source_path),
        molecule_count=len(rows),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(rows) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return parsed_entries, item_rows, stage


def _prepare_timing_library(
    *,
    entries: list[ParsedSmilesEntry],
    dataset_size: str,
    dataset_dimension: str,
    limit: int | None,
    source_path: Path,
    force: bool,
    verbose: bool = False,
    log_label: str = "setup",
) -> tuple[TimingLibraryPaths, TimingStageResult]:
    library_root = _timing_library_root(dataset_size, dataset_dimension, limit)
    moladt_dir = ensure_directory(library_root / "moladt_library")
    manifest_path = library_root / "manifest.csv"
    if not force and manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if not manifest.empty and all((library_root / str(path)).exists() for path in manifest["moladt_relative_path"].tolist()):
            stage = TimingStageResult(
                dataset_size=dataset_size,
                dataset_dimension=dataset_dimension,
                stage="timing_library_prepare",
                description="Reused the cached local timing corpus of MolADT JSON files derived from the source SMILES rows.",
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
            return TimingLibraryPaths(library_root=library_root, moladt_dir=moladt_dir, manifest_path=manifest_path), stage
    records: list[dict[str, Any]] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    latencies: list[float] = []
    success_count = 0
    failure_count = 0
    start_total = time.perf_counter()
    for row_index, entry in enumerate(entries, start=1):
        start_item = time.perf_counter_ns()
        try:
            moladt_relative_path = Path("moladt_library") / f"{entry.mol_id}.moladt.json"
            (library_root / moladt_relative_path).write_bytes(entry.payload)
            records.append(
                {
                    "mol_id": entry.mol_id,
                    "source_index": entry.source_index,
                    "smiles": entry.smiles,
                    "smiles_size_bytes": entry.item_size_bytes,
                    "moladt_relative_path": str(moladt_relative_path),
                    "moladt_size_bytes": len(entry.payload),
                }
            )
            success_count += 1
        except Exception:
            failure_count += 1
        latencies.append((time.perf_counter_ns() - start_item) / 1_000.0)
        if verbose and row_index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc {log_label}] built timing library rows={format_progress(row_index, len(entries))} "
                f"success={success_count} failure={failure_count}"
            )
    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_path, index=False)
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="timing_library_prepare",
        description="Built the matched timing corpus: one MolADT JSON file for each successfully parsed source SMILES string.",
        source_path=str(source_path),
        molecule_count=len(entries),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(entries) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return TimingLibraryPaths(library_root=library_root, moladt_dir=moladt_dir, manifest_path=manifest_path), stage


def _read_timing_library_manifest(library: TimingLibraryPaths) -> pd.DataFrame:
    return pd.read_csv(library.manifest_path)


def _measure_moladt_library_read(
    library: TimingLibraryPaths,
    *,
    manifest: pd.DataFrame,
    dataset_size: str,
    dataset_dimension: str,
    verbose: bool = False,
) -> tuple[list[LoadedMoladtPayload], TimingStageResult]:
    latencies: list[float] = []
    payloads: list[LoadedMoladtPayload] = []
    success_count = 0
    failure_count = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    for row_index, row in enumerate(manifest.itertuples(index=False), start=1):
        relative_path = Path(str(row.moladt_relative_path))
        adt_path = library.library_root / relative_path
        start_item = time.perf_counter_ns()
        try:
            payload = adt_path.read_bytes()
        except Exception:
            failure_count += 1
        else:
            success_count += 1
            payloads.append(
                LoadedMoladtPayload(
                    mol_id=str(row.mol_id),
                    item_path=adt_path,
                    item_size_bytes=int(len(payload)),
                    payload=payload,
                )
            )
        latencies.append((time.perf_counter_ns() - start_item) / 1_000.0)
        if verbose and row_index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc] moladt_json_read rows={format_progress(row_index, len(manifest))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="moladt_json_read",
        description="Read cached MolADT JSON payloads from the matched local corpus without decoding them into objects.",
        source_path=str(library.moladt_dir),
        molecule_count=len(manifest),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(manifest) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return payloads, stage


def _measure_moladt_library_parse(
    payloads: list[LoadedMoladtPayload],
    *,
    dataset_size: str,
    dataset_dimension: str,
    source_dir: Path,
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
            molecule_from_json(payload.payload)
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
                mol_id=payload.mol_id,
                item_kind="moladt_json",
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
                f"[zinc] moladt_file_parse rows={format_progress(row_index, len(payloads))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="moladt_file_parse",
        description="Decoded cached MolADT JSON payloads into the local typed Molecule object with the fast JSON loader when available.",
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
