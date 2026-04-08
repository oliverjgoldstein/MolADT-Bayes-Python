from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from types import MappingProxyType
from typing import Any, Mapping, Protocol, TypeAlias, cast

from .coordinate import Angstrom, Coordinate, mk_angstrom
from .dietz import AtomId, BondingSystem, Edge, SystemId, mk_edge
from .orbital import Shells

class _OrjsonModule(Protocol):
    OPT_INDENT_2: int
    OPT_SORT_KEYS: int

    def dumps(self, obj: Any, /, *, option: int = 0) -> bytes: ...

    def loads(self, obj: str | bytes | bytearray | memoryview, /) -> Any: ...


try:
    import orjson as _orjson  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional wheel is missing locally.
    orjson: _OrjsonModule | None = None
else:
    orjson = cast(_OrjsonModule, _orjson)


class AtomicSymbol(Enum):
    H = "H"
    C = "C"
    N = "N"
    O = "O"
    S = "S"
    P = "P"
    Si = "Si"
    F = "F"
    Cl = "Cl"
    Br = "Br"
    I = "I"
    Fe = "Fe"
    B = "B"
    Na = "Na"

    def __str__(self) -> str:
        return self.value


class SmilesAtomStereoClass(Enum):
    TETRAHEDRAL = "TH"
    ALLENE = "AL"
    SQUARE_PLANAR = "SP"
    TRIGONAL_BIPYRAMIDAL = "TB"
    OCTAHEDRAL = "OH"


class SmilesBondStereoDirection(Enum):
    UP = "/"
    DOWN = "\\"


@dataclass(frozen=True, slots=True)
class SmilesAtomStereo:
    center: AtomId
    stereo_class: SmilesAtomStereoClass
    configuration: int
    token: str

    def __post_init__(self) -> None:
        if self.configuration <= 0:
            raise ValueError("SMILES atom stereochemistry configuration must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.to_dict(),
            "stereo_class": self.stereo_class.value,
            "configuration": self.configuration,
            "token": self.token,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SmilesAtomStereo:
        return cls(
            center=AtomId.from_dict(data["center"]),
            stereo_class=SmilesAtomStereoClass(str(data["stereo_class"])),
            configuration=int(data["configuration"]),
            token=str(data["token"]),
        )


@dataclass(frozen=True, slots=True)
class SmilesBondStereo:
    start_atom: AtomId
    end_atom: AtomId
    direction: SmilesBondStereoDirection

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_atom": self.start_atom.to_dict(),
            "end_atom": self.end_atom.to_dict(),
            "direction": self.direction.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SmilesBondStereo:
        return cls(
            start_atom=AtomId.from_dict(data["start_atom"]),
            end_atom=AtomId.from_dict(data["end_atom"]),
            direction=SmilesBondStereoDirection(str(data["direction"])),
        )


@dataclass(frozen=True, slots=True)
class SmilesStereochemistry:
    atom_stereo: tuple[SmilesAtomStereo, ...] = ()
    bond_stereo: tuple[SmilesBondStereo, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "atom_stereo",
            tuple(sorted(self.atom_stereo, key=lambda item: (item.center.value, item.stereo_class.value, item.configuration, item.token))),
        )
        object.__setattr__(
            self,
            "bond_stereo",
            tuple(sorted(self.bond_stereo, key=lambda item: (item.start_atom.value, item.end_atom.value, item.direction.value))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_stereo": [item.to_dict() for item in self.atom_stereo],
            "bond_stereo": [item.to_dict() for item in self.bond_stereo],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SmilesStereochemistry:
        return cls(
            atom_stereo=tuple(SmilesAtomStereo.from_dict(item) for item in data.get("atom_stereo", [])),
            bond_stereo=tuple(SmilesBondStereo.from_dict(item) for item in data.get("bond_stereo", [])),
        )


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
    smiles_stereochemistry: SmilesStereochemistry = field(default_factory=SmilesStereochemistry)

    def __post_init__(self) -> None:
        atom_map = dict(sorted(self.atoms.items(), key=lambda item: item[0].value))
        for atom_id, atom in atom_map.items():
            if atom.atom_id != atom_id:
                raise ValueError("Atom map keys must match Atom.atom_id")
        object.__setattr__(self, "atoms", MappingProxyType(atom_map))
        object.__setattr__(self, "local_bonds", frozenset(self.local_bonds))
        object.__setattr__(self, "systems", tuple(sorted(self.systems, key=lambda item: item[0].value)))
        object.__setattr__(self, "smiles_stereochemistry", self.smiles_stereochemistry)

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
            "smiles_stereochemistry": self.smiles_stereochemistry.to_dict(),
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
        smiles_stereochemistry = SmilesStereochemistry.from_dict(data.get("smiles_stereochemistry", {}))
        return cls(atoms=atoms, local_bonds=local_bonds, systems=systems, smiles_stereochemistry=smiles_stereochemistry)

    def to_json(self) -> str:
        payload = self.to_dict()
        if orjson is not None:
            return orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode("utf-8")
        return json.dumps(payload, indent=2, sort_keys=True)

    def to_json_bytes(self) -> bytes:
        payload = self.to_dict()
        if orjson is not None:
            return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json(cls, payload: str | bytes) -> Molecule:
        if orjson is not None:
            return cls.from_dict(orjson.loads(payload))
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
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
        smiles_stereochemistry=molecule.smiles_stereochemistry,
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
