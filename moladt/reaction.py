from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, assert_never

from .chem.molecule import Molecule


@dataclass(frozen=True, slots=True)
class TempCondition:
    temperature: float


@dataclass(frozen=True, slots=True)
class PressureCondition:
    pressure: float


Condition: TypeAlias = TempCondition | PressureCondition


@dataclass(frozen=True, slots=True)
class ReactionParticipant:
    amount: float
    molecule: Molecule


@dataclass(frozen=True, slots=True)
class Reaction:
    reactants: tuple[ReactionParticipant, ...]
    products: tuple[ReactionParticipant, ...]
    conditions: tuple[Condition, ...]
    rate: float


@dataclass(frozen=True, slots=True)
class Times:
    start_time: float
    end_time: float


def condition_to_dict(condition: Condition) -> dict[str, Any]:
    match condition:
        case TempCondition(temperature=temperature):
            return {"kind": "temperature", "temperature": temperature}
        case PressureCondition(pressure=pressure):
            return {"kind": "pressure", "pressure": pressure}
        case _ as unreachable:
            assert_never(unreachable)


def condition_from_dict(data: dict[str, Any]) -> Condition:
    kind = data["kind"]
    if kind == "temperature":
        return TempCondition(float(data["temperature"]))
    if kind == "pressure":
        return PressureCondition(float(data["pressure"]))
    raise ValueError(f"Unknown condition kind: {kind}")

