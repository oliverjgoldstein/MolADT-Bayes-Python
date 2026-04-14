from __future__ import annotations

from pathlib import Path

from ..io.sdf import read_sdf


_PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _read_example(name: str):
    return read_sdf(_PROJECT_ROOT / "molecules" / f"{name}.sdf")


hydrogen = _read_example("hydrogen")
oxygen = _read_example("oxygen")
water = _read_example("water")
methane = _read_example("methane")
