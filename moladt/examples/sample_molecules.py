from __future__ import annotations

from ..chem.molecule import AtomicSymbol, Molecule
from ._literal import atom, bond


hydrogen = Molecule(
    atoms={
        atom.atom_id: atom
        for atom in (
            atom(1, AtomicSymbol.H, 0.000, 0.000, -0.370),
            atom(2, AtomicSymbol.H, 0.000, 0.000, 0.370),
        )
    },
    local_bonds=frozenset({bond(1, 2)}),
    systems=(),
)

oxygen = Molecule(
    atoms={
        atom.atom_id: atom
        for atom in (
            atom(1, AtomicSymbol.O, 0.000, 0.000, -0.605),
            atom(2, AtomicSymbol.O, 0.000, 0.000, 0.605),
        )
    },
    local_bonds=frozenset({bond(1, 2)}),
    systems=(),
)

water = Molecule(
    atoms={
        atom.atom_id: atom
        for atom in (
            atom(1, AtomicSymbol.H, 0.002, -0.004, 0.002),
            atom(2, AtomicSymbol.O, -0.011, 0.963, 0.007),
            atom(3, AtomicSymbol.H, 0.867, 1.368, 0.001),
        )
    },
    local_bonds=frozenset({bond(1, 2), bond(2, 3)}),
    systems=(),
)

methane = Molecule(
    atoms={
        atom.atom_id: atom
        for atom in (
            atom(1, AtomicSymbol.C, 0.000, 0.000, 0.000),
            atom(2, AtomicSymbol.H, 0.629, 0.629, 0.629),
            atom(3, AtomicSymbol.H, -0.629, -0.629, 0.629),
            atom(4, AtomicSymbol.H, -0.629, 0.629, -0.629),
            atom(5, AtomicSymbol.H, 0.629, -0.629, -0.629),
        )
    },
    local_bonds=frozenset({bond(1, 2), bond(1, 3), bond(1, 4), bond(1, 5)}),
    systems=(),
)
