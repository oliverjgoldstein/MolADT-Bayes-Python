from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Iterator

from .common import log


@contextmanager
def cmdstan_build_environment(*, verbose: bool = False) -> Iterator[None]:
    original = os.environ.copy()
    try:
        compiler_env = darwin_compiler_environment() if sys.platform == "darwin" else {}
        for key, value in compiler_env.items():
            os.environ[key] = value
        if verbose and compiler_env:
            log(
                "Using Apple toolchain for Stan build: "
                f"CC={compiler_env.get('CC')} CXX={compiler_env.get('CXX')}"
            )
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def darwin_compiler_environment() -> dict[str, str]:
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
        environment["CFLAGS"] = _append_isysroot(os.environ.get("CFLAGS"), sdkroot)
        environment["CXXFLAGS"] = _append_isysroot(os.environ.get("CXXFLAGS"), sdkroot)
    return environment


def _append_isysroot(existing: str | None, sdkroot: str) -> str:
    flag = f"-isysroot {sdkroot}"
    if existing is None or not existing.strip():
        return flag
    if flag in existing:
        return existing
    return f"{existing} {flag}"


def _xcrun_path(tool: str) -> str | None:
    try:
        return subprocess.check_output(["xcrun", "--find", tool], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _xcrun_sdkroot() -> str | None:
    try:
        return subprocess.check_output(["xcrun", "--show-sdk-path"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
