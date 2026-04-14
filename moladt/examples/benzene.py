from __future__ import annotations

from ..chem.molecule import AtomicSymbol, Molecule
from ._literal import atom, bond


benzene = Molecule(
    atoms={
        atom.atom_id: atom
        for atom in (
            atom(1, AtomicSymbol.C, 2.866, 1.000, 0.000),
            atom(2, AtomicSymbol.C, 2.000, 0.500, 0.000),
            atom(3, AtomicSymbol.C, 3.732, 0.500, 0.000),
            atom(4, AtomicSymbol.C, 2.000, -0.500, 0.000),
            atom(5, AtomicSymbol.C, 3.732, -0.500, 0.000),
            atom(6, AtomicSymbol.C, 2.866, -1.000, 0.000),
            atom(7, AtomicSymbol.H, 2.866, 1.620, 0.000),
            atom(8, AtomicSymbol.H, 1.463, 0.810, 0.000),
            atom(9, AtomicSymbol.H, 4.269, 0.810, 0.000),
            atom(10, AtomicSymbol.H, 1.463, -0.810, 0.000),
            atom(11, AtomicSymbol.H, 4.269, -0.810, 0.000),
            atom(12, AtomicSymbol.H, 2.866, -1.620, 0.000),
        )
    },
    local_bonds=frozenset(
        {
            bond(1, 2),
            bond(1, 3),
            bond(1, 7),
            bond(2, 4),
            bond(2, 8),
            bond(3, 5),
            bond(3, 9),
            bond(4, 6),
            bond(4, 10),
            bond(5, 6),
            bond(5, 11),
            bond(6, 12),
        }
    ),
    systems=(),
)

benzene_pretty = benzene
