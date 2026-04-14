from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterator, Mapping, TypeAlias

from .coordinate import Coordinate
from .dietz import AtomId, BondingSystem, Edge, SystemId
from .orbital import Shells


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


@dataclass(frozen=True, slots=True)
class SmilesBondStereo:
    start_atom: AtomId
    end_atom: AtomId
    direction: SmilesBondStereoDirection


@dataclass(frozen=True, slots=True)
class SmilesStereochemistry:
    atom_stereo: tuple[SmilesAtomStereo, ...] = ()
    bond_stereo: tuple[SmilesBondStereo, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "atom_stereo",
            tuple(
                sorted(
                    self.atom_stereo,
                    key=lambda item: (
                        item.center.value,
                        item.stereo_class.value,
                        item.configuration,
                        item.token,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "bond_stereo",
            tuple(
                sorted(
                    self.bond_stereo,
                    key=lambda item: (
                        item.start_atom.value,
                        item.end_atom.value,
                        item.direction.value,
                    ),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ElementAttributes:
    symbol: AtomicSymbol
    atomic_number: int
    atomic_weight: float

    def __str__(self) -> str:
        return f"{self.symbol.value}(Z={self.atomic_number}, {self.atomic_weight:.4f} u)"


@dataclass(frozen=True, slots=True)
class Atom:
    atom_id: AtomId
    attributes: ElementAttributes
    coordinate: Coordinate
    shells: Shells
    formal_charge: int = 0

    def pretty(self) -> str:
        from .pretty import pretty_text

        return pretty_text(self)

    def __str__(self) -> str:
        return self.pretty()


MoleculeSystems: TypeAlias = tuple[tuple[SystemId, BondingSystem], ...]
MoleculeFields: TypeAlias = tuple[
    Mapping[AtomId, Atom],
    frozenset[Edge],
    MoleculeSystems,
    SmilesStereochemistry,
]


def _normalize_atom_map(atoms: Mapping[AtomId, Atom]) -> Mapping[AtomId, Atom]:
    atom_map = dict(sorted(atoms.items(), key=lambda item: item[0].value))
    for atom_id, atom in atom_map.items():
        if atom.atom_id != atom_id:
            raise ValueError("Atom map keys must match Atom.atom_id")
    return MappingProxyType(atom_map)


def _normalize_systems(systems: MoleculeSystems) -> MoleculeSystems:
    return tuple(sorted(systems, key=lambda item: item[0].value))


@dataclass(frozen=True, slots=True)
class Molecule:
    atoms: Mapping[AtomId, Atom]
    local_bonds: frozenset[Edge]
    systems: MoleculeSystems
    smiles_stereochemistry: SmilesStereochemistry = field(default_factory=SmilesStereochemistry)

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", _normalize_atom_map(self.atoms))
        object.__setattr__(self, "local_bonds", frozenset(self.local_bonds))
        object.__setattr__(self, "systems", _normalize_systems(self.systems))

    def __iter__(self) -> Iterator[Mapping[AtomId, Atom] | frozenset[Edge] | MoleculeSystems | SmilesStereochemistry]:
        fields: MoleculeFields = (
            self.atoms,
            self.local_bonds,
            self.systems,
            self.smiles_stereochemistry,
        )
        return iter(fields)

    def pretty(self) -> str:
        from .pretty import pretty_text

        return pretty_text(self)

    def __str__(self) -> str:
        return self.pretty()
