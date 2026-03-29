from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from .coordinate import Angstrom, Coordinate, mk_angstrom
from .dietz import AtomId, BondingSystem, Edge, SystemId, mk_edge
from .orbital import Shells


class AtomicSymbol(Enum):
    H = "H"
    C = "C"
    N = "N"
    O = "O"
    S = "S"
    P = "P"
    F = "F"
    Cl = "Cl"
    Br = "Br"
    I = "I"
    Fe = "Fe"
    B = "B"
    Na = "Na"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ElementAttributes:
    symbol: AtomicSymbol
    atomic_number: int
    atomic_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol.value,
            "atomic_number": self.atomic_number,
            "atomic_weight": self.atomic_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ElementAttributes:
        return cls(
            symbol=AtomicSymbol(data["symbol"]),
            atomic_number=int(data["atomic_number"]),
            atomic_weight=float(data["atomic_weight"]),
        )

    def __str__(self) -> str:
        return f"{self.symbol.value}(Z={self.atomic_number}, {self.atomic_weight:.4f} u)"


@dataclass(frozen=True, slots=True)
class Atom:
    atom_id: AtomId
    attributes: ElementAttributes
    coordinate: Coordinate
    shells: Shells
    formal_charge: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id.to_dict(),
            "attributes": self.attributes.to_dict(),
            "coordinate": self.coordinate.to_dict(),
            "shells": [shell.to_dict() for shell in self.shells],
            "formal_charge": self.formal_charge,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Atom:
        from .constants import element_shells

        symbol = AtomicSymbol(data["attributes"]["symbol"])
        return cls(
            atom_id=AtomId.from_dict(data["atom_id"]),
            attributes=ElementAttributes.from_dict(data["attributes"]),
            coordinate=Coordinate.from_dict(data["coordinate"]),
            shells=element_shells(symbol),
            formal_charge=int(data["formal_charge"]),
        )

    def pretty(self) -> str:
        from .pretty import pretty_text

        return pretty_text(self)

    def __str__(self) -> str:
        return self.pretty()


MoleculeSystems: TypeAlias = tuple[tuple[SystemId, BondingSystem], ...]


@dataclass(frozen=True, slots=True)
class Molecule:
    atoms: Mapping[AtomId, Atom]
    local_bonds: frozenset[Edge]
    systems: MoleculeSystems

    def __post_init__(self) -> None:
        atom_map = dict(sorted(self.atoms.items(), key=lambda item: item[0].value))
        for atom_id, atom in atom_map.items():
            if atom.atom_id != atom_id:
                raise ValueError("Atom map keys must match Atom.atom_id")
        object.__setattr__(self, "atoms", MappingProxyType(atom_map))
        object.__setattr__(self, "local_bonds", frozenset(self.local_bonds))
        object.__setattr__(self, "systems", tuple(sorted(self.systems, key=lambda item: item[0].value)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "atoms": [
                {"atom_id": atom_id.to_dict(), "atom": atom.to_dict()}
                for atom_id, atom in self.atoms.items()
            ],
            "local_bonds": [edge.to_dict() for edge in sorted(self.local_bonds)],
            "systems": [
                {"system_id": system_id.to_dict(), "bonding_system": bonding_system.to_dict()}
                for system_id, bonding_system in self.systems
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Molecule:
        atoms = {
            AtomId.from_dict(item["atom_id"]): Atom.from_dict(item["atom"])
            for item in data["atoms"]
        }
        local_bonds = frozenset(Edge.from_dict(item) for item in data["local_bonds"])
        systems = tuple(
            (SystemId.from_dict(item["system_id"]), BondingSystem.from_dict(item["bonding_system"]))
            for item in data["systems"]
        )
        return cls(atoms=atoms, local_bonds=local_bonds, systems=systems)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, payload: str) -> Molecule:
        return cls.from_dict(json.loads(payload))

    def pretty(self) -> str:
        from .pretty import pretty_text

        return pretty_text(self)

    def __str__(self) -> str:
        return self.pretty()


def add_sigma(atom_a: AtomId, atom_b: AtomId, molecule: Molecule) -> Molecule:
    return Molecule(
        atoms=molecule.atoms,
        local_bonds=molecule.local_bonds | {mk_edge(atom_a, atom_b)},
        systems=molecule.systems,
    )


def distance_angstrom(atom_a: Atom, atom_b: Atom) -> Angstrom:
    dx = atom_a.coordinate.x.value - atom_b.coordinate.x.value
    dy = atom_a.coordinate.y.value - atom_b.coordinate.y.value
    dz = atom_a.coordinate.z.value - atom_b.coordinate.z.value
    return mk_angstrom(sqrt(dx * dx + dy * dy + dz * dz))


def neighbors_sigma(molecule: Molecule, atom_id: AtomId) -> tuple[AtomId, ...]:
    neighbors = []
    for edge in molecule.local_bonds:
        if edge.a == atom_id:
            neighbors.append(edge.b)
        elif edge.b == atom_id:
            neighbors.append(edge.a)
    return tuple(sorted(neighbors))


def edge_systems(molecule: Molecule, edge: Edge) -> tuple[SystemId, ...]:
    return tuple(system_id for system_id, system in molecule.systems if edge in system.member_edges)


def effective_order(molecule: Molecule, edge: Edge) -> float:
    sigma = 1.0 if edge in molecule.local_bonds else 0.0
    pi_contribution = 0.0
    for _, system in molecule.systems:
        if edge in system.member_edges and system.member_edges:
            pi_contribution += system.shared_electrons.value / (2.0 * len(system.member_edges))
    return sigma + pi_contribution


def pretty_print_molecule(molecule: Molecule) -> str:
    return molecule.pretty()
