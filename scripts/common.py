from __future__ import annotations

import json
import os
import shutil
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
LOCAL_CMDSTAN_DIR = PROJECT_ROOT / ".cmdstan"

DEFAULT_SEED = 1
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_VALID_FRACTION = 0.1
DEFAULT_TEST_FRACTION = 0.1

FREESOLV_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
FREESOLV_REPO_ZIP_URL = "https://codeload.github.com/MobleyLab/FreeSolv/zip/refs/heads/master"

QM9_TAR_URLS = (
    "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/qm9.tar.gz",
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.tar.gz",
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz",
)
QM9_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"

ZINC_URL_TEMPLATE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/zinc15_{dataset_size}_{dataset_dimension}.tar.gz"


def configured_results_dir() -> Path:
    override = os.environ.get("MOLADT_RESULTS_DIR")
    if not override:
        return PROJECT_ROOT / "results"
    path = Path(override)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


RESULTS_DIR = configured_results_dir()


@dataclass(frozen=True, slots=True)
class FailureRecord:
    dataset: str
    mol_id: str
    stage: str
    error: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def log(message: str) -> None:
    print(message, flush=True)


def download_file(url: str, destination: Path, *, force: bool = False, timeout_seconds: int = 120) -> Path:
    ensure_directory(destination.parent)
    if destination.exists() and not force:
        log(f"Using cached download {destination}")
        return destination
    log(f"Downloading {url} -> {destination}")
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


def download_first(urls: Sequence[str], destination: Path, *, force: bool = False) -> Path:
    last_error: Exception | None = None
    for url in urls:
        try:
            return download_file(url, destination, force=force)
        except Exception as exc:
            last_error = exc
            log(f"Download failed from {url}: {exc}")
    if last_error is None:
        raise RuntimeError("No download URLs provided")
    raise RuntimeError(f"All download URLs failed for {destination.name}") from last_error


def extract_archive(archive_path: Path, destination_dir: Path, *, force: bool = False) -> Path:
    ensure_directory(destination_dir)
    sentinel = destination_dir / ".extracted"
    if sentinel.exists() and not force:
        log(f"Using cached extraction {destination_dir}")
        return destination_dir
    log(f"Extracting {archive_path} -> {destination_dir}")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination_dir)
    elif archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, mode="r:*") as archive:
            archive.extractall(destination_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")
    sentinel.touch()
    return destination_dir


def copy_if_needed(source: Path, destination: Path, *, force: bool = False) -> Path:
    ensure_directory(destination.parent)
    if destination.exists() and not force:
        return destination
    shutil.copy2(source, destination)
    return destination


def find_files(root: Path, patterns: Sequence[str]) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(root.rglob(pattern)))
    return matches


def require_single_file(root: Path, patterns: Sequence[str], description: str) -> Path:
    matches = find_files(root, patterns)
    if not matches:
        raise FileNotFoundError(f"Could not find {description} under {root}")
    return matches[0]


def write_json(path: Path, payload: Any) -> Path:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_failure_csv(path: Path, failures: Iterable[FailureRecord]) -> Path:
    import pandas as pd

    frame = pd.DataFrame([failure.to_dict() for failure in failures], columns=["dataset", "mol_id", "stage", "error"])
    ensure_directory(path.parent)
    frame.to_csv(path, index=False)
    return path
