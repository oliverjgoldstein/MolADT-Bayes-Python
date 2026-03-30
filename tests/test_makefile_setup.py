from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
MAKEFILE_PATH = REPO_ROOT / "Makefile"


def _write_executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _copy_makefile(tmp_path: Path) -> None:
    shutil.copy(MAKEFILE_PATH, tmp_path / "Makefile")


def test_makefile_prefers_unix_style_venv_python(tmp_path: Path) -> None:
    _copy_makefile(tmp_path)
    _write_executable(tmp_path / ".venv" / "bin" / "python", "#!/bin/sh\nexit 0\n")

    result = subprocess.run(
        ["make", "-C", str(tmp_path), "-n", "python-parse", "SYSTEM_PYTHON=python3"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf" in result.stdout


def test_makefile_supports_windows_style_venv_python(tmp_path: Path) -> None:
    _copy_makefile(tmp_path)
    _write_executable(tmp_path / ".venv" / "Scripts" / "python.exe", "#!/bin/sh\nexit 0\n")

    result = subprocess.run(
        ["make", "-C", str(tmp_path), "-n", "python-parse", "SYSTEM_PYTHON=python3"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "./.venv/Scripts/python.exe -m moladt.cli parse molecules/benzene.sdf" in result.stdout


def test_makefile_benchmark_defaults_to_verbose_output(tmp_path: Path) -> None:
    _copy_makefile(tmp_path)
    _write_executable(tmp_path / ".venv" / "bin" / "python", "#!/bin/sh\nexit 0\n")

    result = subprocess.run(
        ["make", "-C", str(tmp_path), "-n", "benchmark", "SYSTEM_PYTHON=python3"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "./.venv/bin/python -m scripts.run_all benchmark" in result.stdout
    assert "--verbose" in result.stdout


def test_makefile_prints_windows_style_activate_hint(tmp_path: Path) -> None:
    _copy_makefile(tmp_path)
    _write_executable(tmp_path / ".venv" / "Scripts" / "activate", "# activate\n")

    result = subprocess.run(
        ["make", "-C", str(tmp_path), "python-activate"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "source .venv/Scripts/activate" in result.stdout


def test_makefile_python_setup_reports_missing_venv_support(tmp_path: Path) -> None:
    _copy_makefile(tmp_path)
    _write_executable(
        tmp_path / "fake-python",
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-c\" ]; then\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"venv\" ]; then\n"
        "  exit 1\n"
        "fi\n"
        "exit 0\n",
    )

    result = subprocess.run(
        ["make", "-C", str(tmp_path), "python-setup", "SYSTEM_PYTHON=./fake-python", "AUTO_INSTALL_VENV=0"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Python could not create .venv." in result.stdout
    assert "sudo apt install -y python3-venv" in result.stdout


def test_makefile_python_setup_declined_auto_install_reports_manual_steps(tmp_path: Path) -> None:
    _copy_makefile(tmp_path)
    _write_executable(
        tmp_path / "fake-python",
        "#!/bin/sh\n"
        "set -e\n"
        "if [ \"$1\" = \"-c\" ]; then\n"
        "  case \"$2\" in\n"
        "    *'sys.version_info >= (3, 11)'*) exit 0 ;;\n"
        "    *'print(\"{}.{}\".format'*) printf \"%s\\n\" \"3.11\"; exit 0 ;;\n"
        "    *'shutil.rmtree'*) exit 0 ;;\n"
        "    *) exit 0 ;;\n"
        "  esac\n"
        "fi\n"
        "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"venv\" ]; then\n"
        "  printf \"%s\\n\" \"The virtual environment was not created successfully because ensurepip is not available.\" >&2\n"
        "  exit 1\n"
        "fi\n"
        "exit 1\n",
    )
    _write_executable(
        tmp_path / "sudo",
        "#!/bin/sh\n"
        "printf \"%s\\n\" \"$@\" >> sudo.log\n"
        "exit 0\n",
    )
    _write_executable(
        tmp_path / "apt-get",
        "#!/bin/sh\n"
        "printf \"%s\\n\" \"$@\" >> apt.log\n"
        "exit 0\n",
    )

    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    result = subprocess.run(
        ["make", "-C", str(tmp_path), "python-setup", "SYSTEM_PYTHON=./fake-python"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        input="n\n",
    )

    assert result.returncode != 0
    assert "Install the Linux venv package now? [y/N]" in result.stdout
    assert "Python could not create .venv." in result.stdout
    assert not (tmp_path / "apt.log").exists()
    assert not (tmp_path / "sudo.log").exists()


def test_makefile_python_setup_auto_installs_venv_support_with_apt(tmp_path: Path) -> None:
    _copy_makefile(tmp_path)
    _write_executable(
        tmp_path / "fake-python",
        "#!/bin/sh\n"
        "set -e\n"
        "if [ \"$1\" = \"-c\" ]; then\n"
        "  case \"$2\" in\n"
        "    *'sys.version_info >= (3, 11)'*) exit 0 ;;\n"
        "    *'print(\"{}.{}\".format'*) printf \"%s\\n\" \"3.11\"; exit 0 ;;\n"
        "    *'shutil.rmtree'*) exit 0 ;;\n"
        "    *) exit 0 ;;\n"
        "  esac\n"
        "fi\n"
        "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"venv\" ]; then\n"
        "  if [ -f apt-installed.flag ]; then\n"
        "    mkdir -p .venv/bin\n"
        "    cat <<'EOF' > .venv/bin/python\n"
        "#!/bin/sh\n"
        "printf \"%s\\n\" \"$@\" >> setup.log\n"
        "exit 0\n"
        "EOF\n"
        "    chmod +x .venv/bin/python\n"
        "    exit 0\n"
        "  fi\n"
        "  printf \"%s\\n\" \"The virtual environment was not created successfully because ensurepip is not available.\" >&2\n"
        "  exit 1\n"
        "fi\n"
        "exit 1\n",
    )
    _write_executable(
        tmp_path / "sudo",
        "#!/bin/sh\n"
        "exec \"$@\"\n",
    )
    _write_executable(
        tmp_path / "apt-get",
        "#!/bin/sh\n"
        "printf \"%s\\n\" \"$@\" >> apt.log\n"
        "if [ \"$1\" = \"update\" ]; then\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = \"install\" ] && [ \"$2\" = \"-y\" ] && [ \"$3\" = \"python3-venv\" ]; then\n"
        "  exit 1\n"
        "fi\n"
        "if [ \"$1\" = \"install\" ] && [ \"$2\" = \"-y\" ] && [ \"$3\" = \"python3.11-venv\" ]; then\n"
        "  : > apt-installed.flag\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
    )

    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    result = subprocess.run(
        ["make", "-C", str(tmp_path), "python-setup", "SYSTEM_PYTHON=./fake-python"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        input="y\n",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Install the Linux venv package now? [y/N]" in result.stdout
    assert "Detected missing ensurepip support while creating .venv." in result.stdout
    assert "Trying package: python3-venv" in result.stdout
    assert "Trying package: python3.11-venv" in result.stdout
    apt_log = (tmp_path / "apt.log").read_text(encoding="utf-8")
    assert "update" in apt_log
    assert "install" in apt_log
    assert "python3.11-venv" in apt_log
    setup_log = (tmp_path / "setup.log").read_text(encoding="utf-8")
    assert "pip" in setup_log


def test_makefile_python_setup_rejects_unsupported_python_version(tmp_path: Path) -> None:
    _copy_makefile(tmp_path)
    _write_executable(
        tmp_path / "fake-python",
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-V\" ]; then\n"
        "  printf \"%s\\n\" \"Python 3.10.14\"\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = \"-c\" ]; then\n"
        "  exit 1\n"
        "fi\n"
        "exit 0\n",
    )

    result = subprocess.run(
        ["make", "-C", str(tmp_path), "python-setup", "SYSTEM_PYTHON=./fake-python"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "MolADT-Bayes-Python requires Python 3.11+ for setup." in result.stdout
    assert "Found: Python 3.10.14" in result.stdout


@pytest.mark.parametrize(
    ("venv_python_relpath", "expected_python"),
    [
        (".venv/bin/python", ".venv/bin/python"),
        (".venv/Scripts/python.exe", ".venv/Scripts/python.exe"),
    ],
)
def test_makefile_python_setup_uses_created_venv_python(tmp_path: Path, venv_python_relpath: str, expected_python: str) -> None:
    _copy_makefile(tmp_path)
    venv_python = Path(venv_python_relpath)
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        "#!/bin/sh\n"
        "set -e\n"
        "if [ \"$1\" = \"-c\" ]; then\n"
        "  exit 0\n"
        "fi\n"
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
        ["make", "-C", str(tmp_path), "python-setup", "SYSTEM_PYTHON=./fake-python"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    setup_log = (tmp_path / "setup.log").read_text(encoding="utf-8")
    assert "-m" in setup_log
    assert "pip" in setup_log
    assert (tmp_path / expected_python).exists()
