from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True, order=True)
class Angstrom:
    value: float

    def to_dict(self) -> dict[str, float]:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Angstrom:
        return cls(float(data["value"]))


@dataclass(frozen=True, slots=True)
class Coordinate:
    x: Angstrom
    y: Angstrom
    z: Angstrom

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "x": self.x.to_dict(),
            "y": self.y.to_dict(),
            "z": self.z.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Coordinate:
        return cls(
            x=Angstrom.from_dict(data["x"]),
            y=Angstrom.from_dict(data["y"]),
            z=Angstrom.from_dict(data["z"]),
        )


def mk_angstrom(value: float) -> Angstrom:
    return Angstrom(value)

