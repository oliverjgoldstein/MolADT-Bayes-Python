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

from moladt.chem.pretty import pretty_text

from .common import RESULTS_DIR, display_path, ensure_directory, log
from .download_data import download_zinc
from .features import rdkit_mol_to_moladt_record


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


def run_zinc_benchmark(
    *,
    dataset_size: str = "250K",
    dataset_dimension: str = "2D",
    limit: int | None = None,
    include_moladt: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> list[TimingStageResult]:
    if verbose:
        log(
            "Starting ZINC timing benchmark "
            f"(dataset_size={dataset_size}, dataset_dimension={dataset_dimension}, limit={limit}, include_moladt={include_moladt})"
        )
    downloads = download_zinc(dataset_size=dataset_size, dataset_dimension=dataset_dimension, force=force)
    smiles_strings, raw_stage = _load_smiles_with_timing(downloads.source_path, dataset_size=dataset_size, dataset_dimension=dataset_dimension, limit=limit)
    if verbose:
        _log_stage_result(raw_stage)
    parsed_molecules, parse_stage = _measure_stage(
        smiles_strings,
        lambda smiles: Chem.MolFromSmiles(smiles, sanitize=True),
        stage_name="smiles_parse_sanitize",
        stage_description="RDKit parses each SMILES string and sanitizes the molecular graph.",
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        source_path=downloads.source_path,
        keep_successful_outputs=True,
    )
    if verbose:
        _log_stage_result(parse_stage)
    canonical_smiles, canonical_stage = _measure_stage(
        [molecule for molecule in parsed_molecules if molecule is not None],
        lambda molecule: Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        stage_name="smiles_canonicalization",
        stage_description="RDKit re-renders each parsed molecule as a canonical isomeric SMILES string.",
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
        source_path=downloads.source_path,
        keep_successful_outputs=True,
    )
    if verbose:
        _log_stage_result(canonical_stage)
    stages = [raw_stage, parse_stage, canonical_stage]
    if include_moladt:
        _, moladt_stage = _measure_stage(
            [molecule for molecule in parsed_molecules if molecule is not None],
            _moladt_parse_render_from_rdkit_mol,
            stage_name="moladt_parse_render",
            stage_description="RDKit molecules are serialized to MolBlock, parsed into MolADT through the SDF reader, then pretty-rendered from the ADT.",
            dataset_size=dataset_size,
            dataset_dimension=dataset_dimension,
            source_path=downloads.source_path,
            keep_successful_outputs=False,
        )
        stages.append(moladt_stage)
        if verbose:
            _log_stage_result(moladt_stage)
    details_dir = ensure_directory(RESULTS_DIR / "details")
    results_frame = pd.DataFrame([stage.to_dict() for stage in stages])
    results_csv = details_dir / "zinc_timing.csv"
    results_frame.to_csv(results_csv, index=False)
    if verbose:
        log(f"Wrote ZINC timing CSV to {display_path(results_csv)}")
    return stages


def _load_smiles_with_timing(
    source_path: Path,
    *,
    dataset_size: str,
    dataset_dimension: str,
    limit: int | None,
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
                if row_index % 500 == 0:
                    peak_rss = max(peak_rss, process.memory_info().rss)
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
                if row_index % 500 == 0:
                    peak_rss = max(peak_rss, process.memory_info().rss)
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
        if index % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
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


def _log_stage_result(stage: TimingStageResult) -> None:
    log(
        f"[zinc] {stage.stage}: "
        f"molecules={stage.molecule_count} success={stage.success_count} failure={stage.failure_count} "
        f"runtime_s={stage.total_runtime_seconds:.4f} mol_per_s={stage.molecules_per_second:.2f}"
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
