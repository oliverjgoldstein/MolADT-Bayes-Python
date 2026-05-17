from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import time
import zipfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence

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
GITHUB_FILE_SIZE_LIMIT_BYTES = 100_000_000
TRANSFER_CHUNK_SIZE_BYTES = 4 * 1024 * 1024


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


def log_stage(scope: str, stage_index: int, total_stages: int, message: str) -> None:
    log(f"[{scope} {stage_index}/{total_stages}] {message}")


def format_progress(current: int, total: int) -> str:
    if total <= 0:
        return str(current)
    return f"{current}/{total}"


def _format_bytes(num_bytes: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _parse_content_length(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None
    try:
        parsed = int(raw_value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


class _ProgressReporter:
    def __init__(
        self,
        label: str,
        *,
        total_bytes: int | None = None,
        total_items: int | None = None,
        item_label: str = "items",
    ) -> None:
        self.label = label
        self.total_bytes = total_bytes if total_bytes is not None and total_bytes > 0 else None
        self.total_items = total_items if total_items is not None and total_items > 0 else None
        self.item_label = item_label
        self.bytes_done = 0
        self.items_done = 0
        self.start_time = time.monotonic()
        self.last_emit_time = 0.0
        self.dynamic = sys.stdout.isatty()
        self.last_line_length = 0
        self._emit(force=True)

    def update(self, *, bytes_delta: int = 0, items_delta: int = 0) -> None:
        self.bytes_done += bytes_delta
        self.items_done += items_delta
        self._emit(force=False)

    def finish(self) -> None:
        self._emit(force=True)
        if self.dynamic:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def abort(self) -> None:
        if self.dynamic:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _emit(self, *, force: bool) -> None:
        now = time.monotonic()
        interval = 0.2 if self.dynamic else 1.0
        if not force and now - self.last_emit_time < interval:
            return
        message = self._format_message(now)
        if self.dynamic:
            padded = message.ljust(self.last_line_length)
            sys.stdout.write("\r" + padded)
            sys.stdout.flush()
            self.last_line_length = max(self.last_line_length, len(message))
        else:
            log(message)
        self.last_emit_time = now

    def _format_message(self, now: float) -> str:
        parts: list[str] = []
        if self.total_bytes is not None:
            completed_bytes = min(self.bytes_done, self.total_bytes)
            parts.append(
                f"{_format_bytes(completed_bytes)} / {_format_bytes(self.total_bytes)} "
                f"({completed_bytes / self.total_bytes:.1%})"
            )
        else:
            parts.append(_format_bytes(self.bytes_done))
        if self.total_items is not None:
            parts.append(f"{min(self.items_done, self.total_items)}/{self.total_items} {self.item_label}")
        elif self.items_done:
            parts.append(f"{self.items_done} {self.item_label}")
        elapsed = max(now - self.start_time, 1e-9)
        bytes_per_second = self.bytes_done / elapsed
        parts.append(f"{_format_bytes(bytes_per_second)}/s")
        parts.append(f"elapsed {_format_duration(elapsed)}")
        if self.total_bytes is not None and self.bytes_done > 0 and self.bytes_done < self.total_bytes:
            remaining_bytes = self.total_bytes - self.bytes_done
            eta_seconds = remaining_bytes / max(bytes_per_second, 1e-9)
            parts.append(f"eta {_format_duration(eta_seconds)}")
        return f"{self.label}: {', '.join(parts)}"


def _should_show_large_file_progress(total_bytes: int | None) -> bool:
    return total_bytes is not None and total_bytes >= GITHUB_FILE_SIZE_LIMIT_BYTES


def _build_progress_reporter(
    label: str,
    *,
    total_bytes: int | None = None,
    total_items: int | None = None,
    item_label: str = "items",
) -> _ProgressReporter | None:
    if not _should_show_large_file_progress(total_bytes):
        return None
    return _ProgressReporter(
        label,
        total_bytes=total_bytes,
        total_items=total_items,
        item_label=item_label,
    )


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.part")


def _safe_member_destination(destination_dir: Path, member_name: str) -> Path:
    root = destination_dir.resolve()
    target = (destination_dir / Path(member_name)).resolve()
    if os.path.commonpath((str(root), str(target))) != str(root):
        raise ValueError(f"Archive member escapes destination directory: {member_name}")
    return target


def _copy_stream(
    source: BinaryIO,
    destination: BinaryIO,
    reporter: _ProgressReporter | None,
    *,
    chunk_size: int = TRANSFER_CHUNK_SIZE_BYTES,
) -> None:
    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            return
        destination.write(chunk)
        if reporter is not None:
            reporter.update(bytes_delta=len(chunk))


def download_file(url: str, destination: Path, *, force: bool = False, timeout_seconds: int = 120) -> Path:
    ensure_directory(destination.parent)
    if destination.exists() and not force:
        log(f"Using cached download {display_path(destination)}")
        return destination
    temp_path = _temporary_path(destination)
    temp_path.unlink(missing_ok=True)
    reporter: _ProgressReporter | None = None
    try:
        with requests.get(url, stream=True, timeout=timeout_seconds) as response:
            response.raise_for_status()
            total_bytes = _parse_content_length(response.headers.get("content-length"))
            reporter = _build_progress_reporter(
                f"Downloading {display_path(destination)}",
                total_bytes=total_bytes,
            )
            if reporter is None:
                log(f"Downloading {url} -> {display_path(destination)}")
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=TRANSFER_CHUNK_SIZE_BYTES):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    if reporter is not None:
                        reporter.update(bytes_delta=len(chunk))
        temp_path.replace(destination)
        if reporter is not None:
            reporter.finish()
    except Exception:
        if reporter is not None:
            reporter.abort()
        temp_path.unlink(missing_ok=True)
        raise
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
        log(f"Using cached extraction {display_path(destination_dir)}")
        return destination_dir
    reporter: _ProgressReporter | None = None
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            total_bytes = sum(info.file_size for info in members if not info.is_dir())
            reporter = _build_progress_reporter(
                f"Extracting {display_path(archive_path)} -> {display_path(destination_dir)}",
                total_bytes=total_bytes,
                total_items=len(members),
                item_label="entries",
            )
            if reporter is None:
                log(f"Extracting {display_path(archive_path)} -> {display_path(destination_dir)}")
            try:
                for info in members:
                    target_path = _safe_member_destination(destination_dir, info.filename)
                    if info.is_dir():
                        ensure_directory(target_path)
                        if reporter is not None:
                            reporter.update(items_delta=1)
                        continue
                    ensure_directory(target_path.parent)
                    with archive.open(info) as source, target_path.open("wb") as handle:
                        _copy_stream(source, handle, reporter)
                    if reporter is not None:
                        reporter.update(items_delta=1)
            except Exception:
                if reporter is not None:
                    reporter.abort()
                raise
    elif archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = archive.getmembers()
            total_bytes = sum(member.size for member in members if member.isfile())
            reporter = _build_progress_reporter(
                f"Extracting {display_path(archive_path)} -> {display_path(destination_dir)}",
                total_bytes=total_bytes,
                total_items=len(members),
                item_label="entries",
            )
            if reporter is None:
                log(f"Extracting {display_path(archive_path)} -> {display_path(destination_dir)}")
            try:
                for member in members:
                    target_path = _safe_member_destination(destination_dir, member.name)
                    if member.isdir():
                        ensure_directory(target_path)
                        if reporter is not None:
                            reporter.update(items_delta=1)
                        continue
                    if member.isfile():
                        ensure_directory(target_path.parent)
                        source = archive.extractfile(member)
                        if source is None:
                            raise FileNotFoundError(f"Could not extract {member.name} from {archive_path}")
                        with source, target_path.open("wb") as handle:
                            _copy_stream(source, handle, reporter)
                        if member.mode:
                            os.chmod(target_path, member.mode)
                        if reporter is not None:
                            reporter.update(items_delta=1)
                        continue
                    archive.extract(member, path=destination_dir)
                    if reporter is not None:
                        reporter.update(items_delta=1)
            except Exception:
                if reporter is not None:
                    reporter.abort()
                raise
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")
    if reporter is not None:
        reporter.finish()
    sentinel.touch()
    return destination_dir


def copy_if_needed(source: Path, destination: Path, *, force: bool = False) -> Path:
    ensure_directory(destination.parent)
    if destination.exists() and not force:
        return destination
    temp_path = _temporary_path(destination)
    temp_path.unlink(missing_ok=True)
    source_size = source.stat().st_size
    reporter = _build_progress_reporter(
        f"Copying {display_path(source)} -> {display_path(destination)}",
        total_bytes=source_size,
    )
    if reporter is None:
        log(f"Copying {display_path(source)} -> {display_path(destination)}")
    try:
        with source.open("rb") as source_handle, temp_path.open("wb") as destination_handle:
            _copy_stream(source_handle, destination_handle, reporter)
        copied_size = temp_path.stat().st_size
        if copied_size != source_size:
            raise IOError(f"Copied {display_path(source)} to {display_path(temp_path)} with {copied_size} bytes, expected {source_size}")
        shutil.copystat(source, temp_path)
        temp_path.replace(destination)
        final_size = destination.stat().st_size
        if final_size != source_size:
            raise IOError(f"Copied {display_path(source)} to {display_path(destination)} with {final_size} bytes, expected {source_size}")
        if reporter is not None:
            reporter.finish()
    except Exception:
        if reporter is not None:
            reporter.abort()
        temp_path.unlink(missing_ok=True)
        raise
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
