from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .common import (
    FREESOLV_CSV_URL,
    FREESOLV_REPO_ZIP_URL,
    QM9_CSV_URL,
    QM9_TAR_URLS,
    RAW_DATA_DIR,
    ZINC_URL_TEMPLATE,
    copy_if_needed,
    download_file,
    download_first,
    ensure_directory,
    extract_archive,
    log,
    require_single_file,
)


@dataclass(frozen=True, slots=True)
class FreeSolvDownloads:
    csv_path: Path
    repo_archive_path: Path | None
    repo_extract_dir: Path


@dataclass(frozen=True, slots=True)
class QM9Downloads:
    sdf_path: Path
    csv_path: Path
    archive_path: Path | None
    extract_dir: Path | None


@dataclass(frozen=True, slots=True)
class ZincDownloads:
    source_path: Path
    archive_path: Path | None
    extract_dir: Path | None
    dataset_size: str
    dataset_dimension: str


def freesolv_raw_dir() -> Path:
    return RAW_DATA_DIR / "freesolv"


def qm9_raw_dir() -> Path:
    return RAW_DATA_DIR / "qm9"


def zinc_raw_dir() -> Path:
    return RAW_DATA_DIR / "zinc"


def zinc_archive_filename(dataset_size: str, dataset_dimension: str) -> str:
    return f"zinc15_{dataset_size}_{dataset_dimension}.tar.gz"


def zinc_normalized_source_name(dataset_size: str, dataset_dimension: str, suffix: str) -> str:
    return f"zinc15_{dataset_size}_{dataset_dimension}{suffix}"


def _prefer_v3000_path(candidates: tuple[Path, ...]) -> Path:
    if not candidates:
        raise FileNotFoundError("No candidate paths were provided")
    ordered = sorted(
        candidates,
        key=lambda path: (
            0 if "v3000" in path.name.lower() or "v3000" in path.as_posix().lower() else 1,
            path.as_posix(),
        ),
    )
    return ordered[0]


def _freesolv_snapshot_has_metadata(target_dir: Path) -> bool:
    direct_candidate = target_dir / "database.json"
    if direct_candidate.is_file():
        return True
    return any(path.is_file() for path in target_dir.rglob("database.json"))


def _resolve_qm9_sources(extract_dir: Path) -> tuple[Path, Path]:
    sdf_candidates = tuple(
        sorted((extract_dir / candidate) for candidate in ("qm9_v3000.sdf", "gdb9_v3000.sdf") if (extract_dir / candidate).exists())
    )
    if not sdf_candidates:
        sdf_candidates = tuple(sorted(extract_dir.rglob("*V3000*.sdf"))) + tuple(sorted(extract_dir.rglob("*v3000*.sdf")))
    if sdf_candidates:
        sdf_source = _prefer_v3000_path(sdf_candidates)
    else:
        sdf_source = require_single_file(extract_dir, ("qm9.sdf", "gdb9.sdf", "*.sdf"), "QM9 SDF")
    csv_candidates = ("qm9.sdf.csv", "gdb9.sdf.csv", "qm9.csv", "*.csv")
    try:
        csv_source = require_single_file(extract_dir, csv_candidates, "QM9 target CSV")
    except FileNotFoundError:
        csv_source = download_file(QM9_CSV_URL, qm9_raw_dir() / "qm9.csv", force=False)
    return sdf_source, csv_source


def _qm9_cached_copy_matches_sources(target_dir: Path, sdf_path: Path, csv_path: Path) -> tuple[bool, Path | None, Path | None]:
    extract_dir = target_dir / "extracted"
    if not extract_dir.exists():
        return True, None, None
    try:
        sdf_source, csv_source = _resolve_qm9_sources(extract_dir)
    except FileNotFoundError:
        return True, None, None
    return (
        sdf_path.stat().st_size == sdf_source.stat().st_size and csv_path.stat().st_size == csv_source.stat().st_size,
        sdf_source,
        csv_source,
    )


def download_freesolv(*, force: bool = False) -> FreeSolvDownloads:
    target_dir = ensure_directory(freesolv_raw_dir())
    csv_path = target_dir / "SAMPL.csv"
    vendored_sdf_dir = target_dir / "sdffiles"
    vendored_snapshot_exists = csv_path.exists() and vendored_sdf_dir.is_dir() and any(vendored_sdf_dir.glob("*.sdf"))
    if not force and vendored_snapshot_exists and _freesolv_snapshot_has_metadata(target_dir):
        log(f"Using vendored FreeSolv snapshot under {target_dir}")
        return FreeSolvDownloads(csv_path=csv_path, repo_archive_path=None, repo_extract_dir=target_dir)
    if not force and vendored_snapshot_exists:
        log(f"Vendored FreeSolv snapshot under {target_dir} is missing database.json; downloading FreeSolv metadata archive")
    csv_path = download_file(FREESOLV_CSV_URL, csv_path, force=force)
    repo_archive_path = download_file(FREESOLV_REPO_ZIP_URL, target_dir / "FreeSolv-master.zip", force=force)
    repo_extract_dir = extract_archive(repo_archive_path, target_dir / "FreeSolv-master", force=force)
    return FreeSolvDownloads(csv_path=csv_path, repo_archive_path=repo_archive_path, repo_extract_dir=repo_extract_dir)


def download_qm9(*, force: bool = False) -> QM9Downloads:
    target_dir = ensure_directory(qm9_raw_dir())
    sdf_path = target_dir / "qm9.sdf"
    csv_path = target_dir / "qm9.sdf.csv"
    if not force and sdf_path.exists() and csv_path.exists():
        snapshot_matches, cached_sdf_source, cached_csv_source = _qm9_cached_copy_matches_sources(target_dir, sdf_path, csv_path)
        if snapshot_matches:
            log(f"Using vendored QM9 snapshot under {target_dir}")
            return QM9Downloads(sdf_path=sdf_path, csv_path=csv_path, archive_path=None, extract_dir=target_dir)
        if cached_sdf_source is not None and cached_csv_source is not None:
            log(f"Refreshing cached QM9 snapshot under {target_dir} because the copied files do not match the extracted source")
            sdf_path = copy_if_needed(cached_sdf_source, sdf_path, force=True)
            csv_path = copy_if_needed(cached_csv_source, csv_path, force=True)
            return QM9Downloads(sdf_path=sdf_path, csv_path=csv_path, archive_path=None, extract_dir=target_dir / "extracted")
    archive_path = download_first(QM9_TAR_URLS, target_dir / "qm9.tar.gz", force=force)
    extract_dir = extract_archive(archive_path, target_dir / "extracted", force=force)
    sdf_source, csv_source = _resolve_qm9_sources(extract_dir)
    sdf_path = copy_if_needed(sdf_source, sdf_path, force=force)
    csv_path = copy_if_needed(csv_source, csv_path, force=force)
    return QM9Downloads(sdf_path=sdf_path, csv_path=csv_path, archive_path=archive_path, extract_dir=extract_dir)


def download_zinc(*, dataset_size: str = "250K", dataset_dimension: str = "2D", force: bool = False) -> ZincDownloads:
    target_dir = ensure_directory(zinc_raw_dir())
    normalized = target_dir / zinc_normalized_source_name(dataset_size, dataset_dimension, ".csv")
    if not force and normalized.exists():
        log(f"Using vendored ZINC source under {normalized}")
        return ZincDownloads(
            source_path=normalized,
            archive_path=None,
            extract_dir=target_dir,
            dataset_size=dataset_size,
            dataset_dimension=dataset_dimension,
        )
    archive_name = zinc_archive_filename(dataset_size, dataset_dimension)
    archive_path = download_file(
        ZINC_URL_TEMPLATE.format(dataset_size=dataset_size, dataset_dimension=dataset_dimension),
        target_dir / archive_name,
        force=force,
    )
    extract_dir = extract_archive(archive_path, target_dir / f"zinc15_{dataset_size}_{dataset_dimension}", force=force)
    source = require_single_file(extract_dir, ("*.csv", "*.smi", "*.txt"), "ZINC source file")
    normalized = copy_if_needed(source, target_dir / zinc_normalized_source_name(dataset_size, dataset_dimension, source.suffix), force=force)
    return ZincDownloads(
        source_path=normalized,
        archive_path=archive_path,
        extract_dir=extract_dir,
        dataset_size=dataset_size,
        dataset_dimension=dataset_dimension,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.download_data")
    parser.add_argument("dataset", choices=["freesolv", "qm9", "zinc", "all"])
    parser.add_argument("--dataset-size", default="250K")
    parser.add_argument("--dataset-dimension", default="2D")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset in {"freesolv", "all"}:
        downloads = download_freesolv(force=args.force)
        print(downloads.csv_path)
    if args.dataset in {"qm9", "all"}:
        downloads = download_qm9(force=args.force)
        print(downloads.sdf_path)
        print(downloads.csv_path)
    if args.dataset in {"zinc", "all"}:
        downloads = download_zinc(dataset_size=args.dataset_size, dataset_dimension=args.dataset_dimension, force=args.force)
        print(downloads.source_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
