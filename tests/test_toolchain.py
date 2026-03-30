from __future__ import annotations

import os

from scripts import toolchain


def test_darwin_compiler_environment_uses_xcrun_and_sdkroot(monkeypatch) -> None:
    monkeypatch.setattr(toolchain, "_xcrun_path", lambda tool: f"/xcrun/{tool}")
    monkeypatch.setattr(toolchain, "_xcrun_sdkroot", lambda: "/sdk/root")
    monkeypatch.setenv("CFLAGS", "-O2")
    monkeypatch.setenv("CXXFLAGS", "-O3")

    environment = toolchain.darwin_compiler_environment()

    assert environment["CC"] == "/xcrun/clang"
    assert environment["CXX"] == "/xcrun/clang++"
    assert environment["SDKROOT"] == "/sdk/root"
    assert environment["CFLAGS"] == "-O2 -isysroot /sdk/root"
    assert environment["CXXFLAGS"] == "-O3 -isysroot /sdk/root"


def test_cmdstan_build_environment_restores_original_environment(monkeypatch) -> None:
    monkeypatch.setattr(toolchain.sys, "platform", "darwin")
    monkeypatch.setattr(toolchain, "darwin_compiler_environment", lambda: {"CC": "/xcrun/clang", "SDKROOT": "/sdk"})
    monkeypatch.setenv("CC", "/usr/bin/clang")
    monkeypatch.delenv("SDKROOT", raising=False)

    with toolchain.cmdstan_build_environment():
        assert os.environ["CC"] == "/xcrun/clang"
        assert os.environ["SDKROOT"] == "/sdk"

    assert os.environ["CC"] == "/usr/bin/clang"
    assert "SDKROOT" not in os.environ
