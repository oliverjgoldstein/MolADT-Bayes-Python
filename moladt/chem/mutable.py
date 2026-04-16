from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, TypeAlias

from .dietz import AtomId, BondingSystem, Edge, SystemId
from .molecule import Atom, Molecule, SmilesStereochemistry
from .pretty import pretty_text


MutableMoleculeSystems: TypeAlias = list[tuple[SystemId, BondingSystem]]
MutableMoleculeFields: TypeAlias = tuple[
    dict[AtomId, Atom],
    set[Edge],
    MutableMoleculeSystems,
    SmilesStereochemistry,
]


@dataclass(slots=True)
class MutableMolecule:
    atoms: dict[AtomId, Atom]
    local_bonds: set[Edge]
    systems: MutableMoleculeSystems
    smiles_stereochemistry: SmilesStereochemistry = field(default_factory=SmilesStereochemistry)

    def __post_init__(self) -> None:
        atom_map = dict(sorted(self.atoms.items(), key=lambda item: item[0].value))
        for atom_id, atom in atom_map.items():
            if atom.atom_id != atom_id:
                raise ValueError("Atom map keys must match Atom.atom_id")
        self.atoms = atom_map
        self.local_bonds = set(self.local_bonds)
        self.systems = list(sorted(self.systems, key=lambda item: item[0].value))

    def __iter__(self) -> Iterator[dict[AtomId, Atom] | set[Edge] | MutableMoleculeSystems | SmilesStereochemistry]:
        fields: MutableMoleculeFields = (
            self.atoms,
            self.local_bonds,
            self.systems,
            self.smiles_stereochemistry,
        )
        return iter(fields)

    @classmethod
    def from_molecule(cls, molecule: Molecule) -> MutableMolecule:
        return cls(
            atoms=dict(molecule.atoms),
            local_bonds=set(molecule.local_bonds),
            systems=list(molecule.systems),
            smiles_stereochemistry=molecule.smiles_stereochemistry,
        )

    def freeze(self) -> Molecule:
        return Molecule(
            atoms=self.atoms,
            local_bonds=frozenset(self.local_bonds),
            systems=tuple(self.systems),
            smiles_stereochemistry=self.smiles_stereochemistry,
        )

    def copy(self) -> MutableMolecule:
        return MutableMolecule(
            atoms=dict(self.atoms),
            local_bonds=set(self.local_bonds),
            systems=list(self.systems),
            smiles_stereochemistry=self.smiles_stereochemistry,
        )

    def pretty(self) -> str:
        return pretty_text(self.freeze())

    def __str__(self) -> str:
        return self.pretty()
