from __future__ import annotations

import io
import tarfile
from pathlib import Path

from scripts import common


class _FakeResponse:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        content_length: int | None = None,
        fail_after_chunks: int | None = None,
    ) -> None:
        self._chunks = chunks
        self._fail_after_chunks = fail_after_chunks
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, *, chunk_size: int):  # type: ignore[override]
        del chunk_size
        for index, chunk in enumerate(self._chunks):
            if self._fail_after_chunks is not None and index >= self._fail_after_chunks:
                raise RuntimeError("simulated download failure")
            yield chunk
        if self._fail_after_chunks is not None and self._fail_after_chunks >= len(self._chunks):
            raise RuntimeError("simulated download failure")


def test_download_file_reports_progress(capsys, monkeypatch, tmp_path: Path) -> None:
    payload = [b"abcd", b"efgh"]
    destination = tmp_path / "demo.bin"
    monkeypatch.setattr(common, "GITHUB_FILE_SIZE_LIMIT_BYTES", 8)

    monkeypatch.setattr(
        common.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(payload, content_length=8),
    )

    result = common.download_file("https://example.test/demo.bin", destination)

    assert result == destination
    assert destination.read_bytes() == b"abcdefgh"
    output = capsys.readouterr().out
    assert "Downloading" in output
    assert str(destination) in output
    assert "8 B / 8 B" in output
    assert "100.0%" in output
    assert "/s" in output
    assert "elapsed" in output


def test_download_first_retries_after_partial_failure(capsys, monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "payload.bin"
    monkeypatch.setattr(common, "GITHUB_FILE_SIZE_LIMIT_BYTES", 5)
    responses = {
        "https://example.test/first.bin": _FakeResponse([b"partial"], content_length=7, fail_after_chunks=1),
        "https://example.test/second.bin": _FakeResponse([b"final"], content_length=5),
    }

    def fake_get(url: str, *args, **kwargs) -> _FakeResponse:
        return responses[url]

    monkeypatch.setattr(common.requests, "get", fake_get)

    result = common.download_first(
        ("https://example.test/first.bin", "https://example.test/second.bin"),
        destination,
    )

    assert result == destination
    assert destination.read_bytes() == b"final"
    assert not (tmp_path / "payload.bin.part").exists()
    output = capsys.readouterr().out
    assert "Download failed from https://example.test/first.bin" in output


def test_extract_archive_reports_progress(capsys, tmp_path: Path) -> None:
    archive_path = tmp_path / "demo.tar.gz"
    extracted_dir = tmp_path / "extracted"
    payload = b"moladt-progress" * 128
    original_limit = common.GITHUB_FILE_SIZE_LIMIT_BYTES
    common.GITHUB_FILE_SIZE_LIMIT_BYTES = len(payload)

    try:
        with tarfile.open(archive_path, mode="w:gz") as archive:
            info = tarfile.TarInfo("nested/demo.txt")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

        result = common.extract_archive(archive_path, extracted_dir)

        assert result == extracted_dir
        assert (extracted_dir / "nested" / "demo.txt").read_bytes() == payload
        assert (extracted_dir / ".extracted").exists()
        output = capsys.readouterr().out
        assert "Extracting" in output
        assert str(archive_path) in output
        assert "1/1 entries" in output
        assert "100.0%" in output
        assert "/s" in output
        assert "elapsed" in output
    finally:
        common.GITHUB_FILE_SIZE_LIMIT_BYTES = original_limit


def test_copy_if_needed_reports_progress(capsys, tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "copied.bin"
    source.write_bytes(b"x" * 8192)
    monkeypatch_limit = len(source.read_bytes())
    original_limit = common.GITHUB_FILE_SIZE_LIMIT_BYTES
    common.GITHUB_FILE_SIZE_LIMIT_BYTES = monkeypatch_limit

    try:
        result = common.copy_if_needed(source, destination)

        assert result == destination
        assert destination.read_bytes() == source.read_bytes()
        output = capsys.readouterr().out
        assert "Copying" in output
        assert str(destination) in output
        assert "100.0%" in output
        assert "/s" in output
        assert "elapsed" in output
    finally:
        common.GITHUB_FILE_SIZE_LIMIT_BYTES = original_limit


def test_small_download_does_not_emit_progress_meter(capsys, monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "small.bin"
    monkeypatch.setattr(common, "GITHUB_FILE_SIZE_LIMIT_BYTES", 1024)
    monkeypatch.setattr(
        common.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse([b"tiny"], content_length=4),
    )

    common.download_file("https://example.test/small.bin", destination)

    output = capsys.readouterr().out
    assert "Downloading https://example.test/small.bin" in output
    assert "100.0%" not in output
