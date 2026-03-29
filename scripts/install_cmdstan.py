from __future__ import annotations

import argparse
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import cmdstanpy

from .common import LOCAL_CMDSTAN_DIR, ensure_directory, log

DEFAULT_CMDSTAN_VERSION = "2.38.0"


def install_or_repair_cmdstan(*, version: str = DEFAULT_CMDSTAN_VERSION, force: bool = False) -> Path:
    install_root = ensure_directory(LOCAL_CMDSTAN_DIR)
    version_dir = install_root / f"cmdstan-{version}"

    with _cmdstan_build_environment():
        if version_dir.exists():
            log(f"Found existing CmdStan source tree at {version_dir}")
            _build_cmdstan(version_dir)
            _set_cmdstan_path(version_dir)
            log(f"CmdStan is ready at {version_dir}")
            return version_dir

        log(f"Installing CmdStan {version} into {install_root}")
        ok = cmdstanpy.install_cmdstan(
            dir=str(install_root),
            version=version,
            overwrite=force,
            verbose=True,
        )
        if ok and version_dir.exists():
            _set_cmdstan_path(version_dir)
            log(f"CmdStan is ready at {version_dir}")
            return version_dir

        if version_dir.exists():
            log("cmdstanpy reported a failed install, but the source tree exists. Attempting a local build repair.")
            _build_cmdstan(version_dir)
            _set_cmdstan_path(version_dir)
            log(f"CmdStan is ready at {version_dir}")
            return version_dir

    raise RuntimeError(f"CmdStan {version} could not be installed into {install_root}")


@contextmanager
def _cmdstan_build_environment():
    original = os.environ.copy()
    try:
        if sys.platform == "darwin":
            compiler_env = _darwin_compiler_environment()
            if compiler_env:
                for key, value in compiler_env.items():
                    os.environ[key] = value
                log(
                    "Using Apple toolchain for CmdStan build: "
                    f"CC={compiler_env['CC']} CXX={compiler_env['CXX']}"
                )
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def _darwin_compiler_environment() -> dict[str, str]:
    clang = _xcrun_path("clang")
    clangxx = _xcrun_path("clang++")
    sdkroot = _xcrun_sdkroot()
    environment: dict[str, str] = {}
    if clang is not None:
        environment["CC"] = clang
    if clangxx is not None:
        environment["CXX"] = clangxx
    if sdkroot is not None:
        environment["SDKROOT"] = sdkroot
    return environment


def _xcrun_path(tool: str) -> str | None:
    try:
        return subprocess.check_output(["xcrun", "--find", tool], text=True).strip()
    except Exception:
        return None


def _xcrun_sdkroot() -> str | None:
    try:
        return subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    except Exception:
        return None


def _build_cmdstan(version_dir: Path) -> None:
    subprocess.run(["make", "build", "-j1"], cwd=version_dir, check=True)


def _set_cmdstan_path(version_dir: Path) -> None:
    cmdstanpy.set_cmdstan_path(str(version_dir))
    resolved = cmdstanpy.cmdstan_path()
    if Path(resolved).resolve() != version_dir.resolve():
        raise RuntimeError(f"CmdStanPy resolved unexpected path {resolved}, expected {version_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.install_cmdstan")
    parser.add_argument("--version", default=DEFAULT_CMDSTAN_VERSION)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = install_or_repair_cmdstan(version=args.version, force=args.force)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
