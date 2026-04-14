from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

from moladt.chem.molecule import Molecule
from moladt.io.sdf import SDFRecord, parse_sdf_record

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
    sdf_dir: Path
    moladt_dir: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class ParsedSDFEntry:
    source_index: int
    block_text: str
    record: SDFRecord


def run_zinc_benchmark(
    *,
    dataset_size: str = "250K",
    dataset_dimension: str = "3D",
    limit: int | None = None,
    include_moladt: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> list[TimingStageResult]:
    total_stages = 4 if include_moladt else 2
    if verbose:
        log(
            "Starting ZINC timing benchmark "
            f"(dataset_size={dataset_size}, dataset_dimension={dataset_dimension}, limit={limit}, include_moladt={include_moladt})"
        )
        log(f"Results directory: {display_path(RESULTS_DIR)}")
        log_stage("zinc", 1, total_stages, "Reading raw SDF source file")
    downloads = download_zinc(dataset_size=dataset_size, dataset_dimension=dataset_dimension, force=force)
    blocks, raw_stage = _load_sdf_blocks_with_timing(
        downloads.source_path,
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        limit=limit,
        verbose=verbose,
    )
    stages = [raw_stage]
    if verbose:
        _log_stage_result(raw_stage, stage_index=1, total_stages=total_stages)
        log_stage("zinc", 2, total_stages, f"Parsing SDF records into MolADT (records={len(blocks)})")
    parsed_entries, parse_stage = _measure_sdf_record_parse(
        blocks,
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        source_path=downloads.source_path,
        verbose=verbose,
    )
    stages.append(parse_stage)
    if verbose:
        _log_stage_result(parse_stage, stage_index=2, total_stages=total_stages)

    item_rows: list[TimingItemResult] = []
    manifest: pd.DataFrame | None = None
    if include_moladt:
        if verbose:
            log_stage("zinc", 3, total_stages, f"Building matched local timing corpus (records={len(parsed_entries)})")
        library, library_stage = _prepare_timing_library(
            entries=parsed_entries,
            dataset_size=dataset_size,
            dataset_dimension=dataset_dimension,
            limit=limit,
            source_path=downloads.source_path,
            force=force,
            verbose=verbose,
            log_label=f"3/{total_stages}",
        )
        stages.append(library_stage)
        manifest = _read_timing_library_manifest(library)
        if verbose:
            _log_stage_result(library_stage, stage_index=3, total_stages=total_stages)
            log(f"[zinc] timing library root: {display_path(library.library_root)}")
            log_stage("zinc", 4, total_stages, f"Parsing local MolADT timing library (records={len(manifest)})")
        moladt_item_rows, moladt_stage = _measure_moladt_library_parse(
            library,
            manifest=manifest,
            dataset_size=dataset_size,
            dataset_dimension=dataset_dimension,
            verbose=verbose,
            log_label=f"4/{total_stages}",
        )
        item_rows.extend(moladt_item_rows)
        stages.append(moladt_stage)
        if verbose:
            _log_stage_result(moladt_stage, stage_index=4, total_stages=total_stages)

    details_dir = ensure_directory(RESULTS_DIR / "details")
    results_frame = pd.DataFrame([stage.to_dict() for stage in stages])
    results_csv = details_dir / "zinc_timing.csv"
    results_frame.to_csv(results_csv, index=False)
    if item_rows:
        item_frame = pd.DataFrame([row.to_dict() for row in item_rows])
        item_frame.to_csv(details_dir / "zinc_timing_items.csv", index=False)
    if manifest is not None:
        manifest.to_csv(details_dir / "zinc_timing_library_manifest.csv", index=False)
    if verbose:
        log(f"[zinc] wrote timing rows={len(results_frame)} to {display_path(results_csv)}")
    return stages


def _timing_library_root(dataset_size: str, dataset_dimension: str, limit: int | None) -> Path:
    scope = "full" if limit is None else f"limit_{limit}"
    return ensure_directory(PROCESSED_DATA_DIR / "zinc_timing" / f"zinc15_{dataset_size}_{dataset_dimension}" / scope)


def _load_sdf_blocks_with_timing(
    source_path: Path,
    *,
    dataset_size: str,
    dataset_dimension: str,
    limit: int | None,
    verbose: bool = False,
) -> tuple[list[str], TimingStageResult]:
    blocks: list[str] = []
    latencies: list[float] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    count = 0
    block_lines: list[str] = []
    with source_path.open("r", encoding="latin-1") as handle:
        for line_index, line in enumerate(handle, start=1):
            if line.rstrip("\n\r") == "$$$$":
                start_item = time.perf_counter_ns()
                block = "".join(block_lines).strip("\n")
                block_lines.clear()
                if block.strip():
                    blocks.append(block)
                    count += 1
                latencies.append((time.perf_counter_ns() - start_item) / 1_000.0)
                if verbose and count and count % 500 == 0:
                    peak_rss = max(peak_rss, process.memory_info().rss)
                    target_total = limit if limit is not None else 0
                    if target_total > 0:
                        log(f"[zinc] raw SDF records={format_progress(count, target_total)}")
                    else:
                        log(f"[zinc] raw SDF records={count}")
                if limit is not None and count >= limit:
                    break
                continue
            block_lines.append(line)
        if (limit is None or count < limit) and block_lines:
            start_item = time.perf_counter_ns()
            block = "".join(block_lines).strip("\n")
            if block.strip():
                blocks.append(block)
            latencies.append((time.perf_counter_ns() - start_item) / 1_000.0)
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="raw_file_read",
        description="Read raw single-record SDF blocks from the normalized ZINC source file without chemistry parsing.",
        source_path=str(source_path),
        molecule_count=len(blocks),
        success_count=len(blocks),
        failure_count=0,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(blocks) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return blocks, stage


def _measure_sdf_record_parse(
    blocks: list[str],
    *,
    dataset_size: str,
    dataset_dimension: str,
    source_path: Path,
    verbose: bool = False,
) -> tuple[list[ParsedSDFEntry], TimingStageResult]:
    latencies: list[float] = []
    entries: list[ParsedSDFEntry] = []
    success_count = 0
    failure_count = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_total = time.perf_counter()
    for index, block in enumerate(blocks, start=1):
        start_item = time.perf_counter_ns()
        try:
            record = parse_sdf_record(block)
        except Exception:
            failure_count += 1
        else:
            success_count += 1
            entries.append(ParsedSDFEntry(source_index=index, block_text=block, record=record))
        latencies.append((time.perf_counter_ns() - start_item) / 1_000.0)
        if verbose and index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc] sdf_record_parse records={format_progress(index, len(blocks))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="sdf_record_parse",
        description="Parsed each source SDF record into the local MolADT object with the project SDF reader.",
        source_path=str(source_path),
        molecule_count=len(blocks),
        success_count=success_count,
        failure_count=failure_count,
        total_runtime_seconds=elapsed,
        molecules_per_second=(len(blocks) / elapsed) if elapsed > 0 else 0.0,
        median_latency_us=_median(latencies),
        p95_latency_us=_percentile(latencies, 95.0),
        peak_rss_mb=peak_rss / (1024.0 * 1024.0),
    )
    return entries, stage


def _prepare_timing_library(
    *,
    entries: list[ParsedSDFEntry],
    dataset_size: str,
    dataset_dimension: str,
    limit: int | None,
    source_path: Path,
    force: bool,
    verbose: bool = False,
    log_label: str = "3/4",
) -> tuple[TimingLibraryPaths, TimingStageResult]:
    library_root = _timing_library_root(dataset_size, dataset_dimension, limit)
    sdf_dir = ensure_directory(library_root / "sdf_library")
    moladt_dir = ensure_directory(library_root / "moladt_library")
    manifest_path = library_root / "manifest.csv"
    if not force and manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if (
            not manifest.empty
            and all((library_root / str(path)).exists() for path in manifest["sdf_relative_path"].tolist())
            and all((library_root / str(path)).exists() for path in manifest["moladt_relative_path"].tolist())
        ):
            stage = TimingStageResult(
                dataset_size=dataset_size,
                dataset_dimension=dataset_dimension,
                stage="timing_library_prepare",
                description="Reused the cached local timing corpus of matched single-record SDF files plus MolADT JSON files.",
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
            return TimingLibraryPaths(library_root=library_root, sdf_dir=sdf_dir, moladt_dir=moladt_dir, manifest_path=manifest_path), stage
    records: list[dict[str, Any]] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    latencies: list[float] = []
    success_count = 0
    failure_count = 0
    start_total = time.perf_counter()
    for entry in entries:
        start_item = time.perf_counter_ns()
        try:
            mol_id = f"zinc_{entry.source_index:07d}"
            sdf_relative_path = Path("sdf_library") / f"{mol_id}.sdf"
            moladt_relative_path = Path("moladt_library") / f"{mol_id}.moladt.json"
            sdf_payload = entry.block_text.rstrip("\n") + "\n$$$$\n"
            moladt_payload = entry.record.molecule.to_json_bytes()
            (library_root / sdf_relative_path).write_text(sdf_payload, encoding="latin-1")
            (library_root / moladt_relative_path).write_bytes(moladt_payload)
            records.append(
                {
                    "mol_id": mol_id,
                    "source_index": entry.source_index,
                    "title": entry.record.title,
                    "sdf_relative_path": str(sdf_relative_path),
                    "sdf_size_bytes": len(sdf_payload.encode("latin-1")),
                    "moladt_relative_path": str(moladt_relative_path),
                    "moladt_size_bytes": len(moladt_payload),
                }
            )
            success_count += 1
        except Exception:
            failure_count += 1
        latencies.append((time.perf_counter_ns() - start_item) / 1_000.0)
        if verbose and entry.source_index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
            log(
                f"[zinc {log_label}] built timing library records={format_progress(entry.source_index, len(entries))} "
                f"success={success_count} failure={failure_count}"
            )
    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_path, index=False)
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="timing_library_prepare",
        description="Built the matched timing corpus: one single-record SDF file plus one MolADT JSON file for each parsed source molecule.",
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
    return TimingLibraryPaths(library_root=library_root, sdf_dir=sdf_dir, moladt_dir=moladt_dir, manifest_path=manifest_path), stage


def _read_timing_library_manifest(library: TimingLibraryPaths) -> pd.DataFrame:
    return pd.read_csv(library.manifest_path)


def _measure_moladt_library_parse(
    library: TimingLibraryPaths,
    *,
    manifest: pd.DataFrame,
    dataset_size: str,
    dataset_dimension: str,
    verbose: bool = False,
    log_label: str = "4/4",
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
                f"[zinc {log_label}] parsed MolADT library records={format_progress(row_index, len(manifest))} "
                f"success={success_count} failure={failure_count}"
            )
    elapsed = time.perf_counter() - start_total
    stage = TimingStageResult(
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        stage="moladt_file_parse",
        description="Read each MolADT JSON file and reconstructed the local Molecule object with the fast JSON loader when available.",
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
