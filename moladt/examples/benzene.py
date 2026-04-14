from __future__ import annotations

from pathlib import Path

from ..io.sdf import read_sdf


_PROJECT_ROOT = Path(__file__).resolve().parents[2]

benzene = read_sdf(_PROJECT_ROOT / "molecules" / "benzene.sdf")

benzene_pretty = benzene
