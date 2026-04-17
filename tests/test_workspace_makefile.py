from __future__ import annotations

import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_MAKEFILE_PATH = WORKSPACE_ROOT / "Makefile"


def _write_executable(path: Path, contents: str = "#!/bin/sh\nexit 0\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _copy_workspace_makefile(tmp_path: Path) -> None:
    shutil.copy(WORKSPACE_MAKEFILE_PATH, tmp_path / "Makefile")


def test_workspace_makefile_benchmark_uses_timestamped_results_directory(tmp_path: Path) -> None:
    _copy_workspace_makefile(tmp_path)
    _write_executable(tmp_path / "MolADT-Bayes-Python" / ".venv313" / "bin" / "python")

    result = subprocess.run(
        ["make", "-C", str(tmp_path), "-n", "benchmark", "SYSTEM_PYTHON=python3"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert re.search(r"MOLADT_RESULTS_DIR=results/run_\d{8}_\d{6}", result.stdout)
    assert "Running workspace combined MolADT benchmark bundle." in result.stdout


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
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert re.search(r"MOLADT_RESULTS_DIR=results/paper/run_\d{8}_\d{6}", result.stdout)


def test_workspace_makefile_supports_windows_style_venv_python(tmp_path: Path) -> None:
    _copy_workspace_makefile(tmp_path)
    _write_executable(tmp_path / "MolADT-Bayes-Python" / ".venv313" / "Scripts" / "python.exe")

    result = subprocess.run(
        ["make", "-C", str(tmp_path), "-n", "python-parse", "SYSTEM_PYTHON=python3"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "./.venv313/Scripts/python.exe -m moladt.cli parse molecules/benzene.sdf" in result.stdout


def test_workspace_makefile_showcase_passes_results_subdir(tmp_path: Path) -> None:
    _copy_workspace_makefile(tmp_path)
    _write_executable(tmp_path / "MolADT-Bayes-Python" / ".venv313" / "bin" / "python")

    result = subprocess.run(
        ["make", "-C", str(tmp_path), "-n", "showcase", "SYSTEM_PYTHON=python3"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert re.search(r"--results-subdir run_\d{8}_\d{6}", result.stdout)
    assert "Running workspace showcase bundle." in result.stdout


@pytest.mark.parametrize(
    ("venv_python_relpath", "expected_python"),
    [
        (".venv/bin/python", ".venv/bin/python"),
        (".venv/Scripts/python.exe", ".venv/Scripts/python.exe"),
    ],
)
def test_workspace_makefile_python_setup_uses_created_venv_python(
    tmp_path: Path,
    venv_python_relpath: str,
    expected_python: str,
) -> None:
    _copy_workspace_makefile(tmp_path)
    python_repo = tmp_path / "MolADT-Bayes-Python"
    python_repo.mkdir()
    venv_python = Path(venv_python_relpath)

    _write_executable(
        tmp_path / "fake-python",
        "#!/bin/sh\n"
        "set -e\n"
        "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"venv\" ]; then\n"
        f"  mkdir -p '{venv_python.parent.as_posix()}'\n"
        f"  cat <<'EOF' > '{venv_python.as_posix()}'\n"
        "#!/bin/sh\n"
        "printf \"%s\\n\" \"$@\" >> setup.log\n"
        "exit 0\n"
        "EOF\n"
        f"  chmod +x '{venv_python.as_posix()}'\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
    )

    result = subprocess.run(
        ["make", "-C", str(tmp_path), "python-setup", "SYSTEM_PYTHON=../fake-python"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    setup_log = (python_repo / "setup.log").read_text(encoding="utf-8")
    assert "-m" in setup_log
    assert "pip" in setup_log
    assert (python_repo / expected_python).exists()
