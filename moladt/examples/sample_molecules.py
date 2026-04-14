from __future__ import annotations

from pathlib import Path

from ..io.sdf import read_sdf


_PROJECT_ROOT = Path(__file__).resolve().parents[2]

hydrogen = read_sdf(_PROJECT_ROOT / "molecules" / "hydrogen.sdf")
oxygen = read_sdf(_PROJECT_ROOT / "molecules" / "oxygen.sdf")
water = read_sdf(_PROJECT_ROOT / "molecules" / "water.sdf")
methane = read_sdf(_PROJECT_ROOT / "molecules" / "methane.sdf")
