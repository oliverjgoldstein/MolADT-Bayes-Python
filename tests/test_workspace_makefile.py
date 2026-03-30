from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_MAKEFILE_PATH = WORKSPACE_ROOT / "Makefile"


def _write_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _copy_workspace_makefile(tmp_path: Path) -> None:
    shutil.copy(WORKSPACE_MAKEFILE_PATH, tmp_path / "Makefile")


def test_workspace_makefile_benchmark_uses_timestamped_results_directory(tmp_path: Path) -> None:
    _copy_workspace_makefile(tmp_path)
    _write_executable(tmp_path / "MolADT-Bayes-Python" / ".venv313" / "bin" / "python")

    result = subprocess.run(
        ["make", "-C", str(tmp_path), "-n", "benchmark", "SYSTEM_PYTHON=python3", "RUN_TIMESTAMP=20260330_170000"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "MOLADT_RESULTS_DIR=results/run_20260330_170000" in result.stdout


def test_workspace_makefile_paper_benchmark_uses_paper_results_directory(tmp_path: Path) -> None:
    _copy_workspace_makefile(tmp_path)
    _write_executable(tmp_path / "MolADT-Bayes-Python" / ".venv313" / "bin" / "python")

    result = subprocess.run(
        [
            "make",
            "-C",
            str(tmp_path),
            "-n",
            "benchmark",
            "SYSTEM_PYTHON=python3",
            "INFERENCE_PRESET=paper",
            "RUN_TIMESTAMP=20260330_170000",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "MOLADT_RESULTS_DIR=results/paper/run_20260330_170000" in result.stdout


def test_workspace_makefile_showcase_passes_results_subdir(tmp_path: Path) -> None:
    _copy_workspace_makefile(tmp_path)
    _write_executable(tmp_path / "MolADT-Bayes-Python" / ".venv313" / "bin" / "python")

    result = subprocess.run(
        ["make", "-C", str(tmp_path), "-n", "showcase", "SYSTEM_PYTHON=python3", "RUN_TIMESTAMP=20260330_170000"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--results-subdir run_20260330_170000" in result.stdout
