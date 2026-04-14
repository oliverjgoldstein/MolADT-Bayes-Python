from __future__ import annotations

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import AtomicSymbol, Molecule
from ._literal import atom, bond


def _edge(atom_a: AtomId, atom_b: AtomId) -> Edge:
    return Edge(atom_a, atom_b)


b1 = AtomId(1)
b2 = AtomId(2)
h3 = AtomId(3)
h4 = AtomId(4)
h5 = AtomId(5)
h6 = AtomId(6)
h7 = AtomId(7)
h8 = AtomId(8)

diborane_pretty = Molecule(
    atoms={
        atom.atom_id: atom
        for atom in (
            atom(1, AtomicSymbol.B, -0.885, 0.000, 0.000),
            atom(2, AtomicSymbol.B, 0.885, 0.000, 0.000),
            atom(3, AtomicSymbol.H, 0.000, 0.000, 0.993),
            atom(4, AtomicSymbol.H, 0.000, 0.000, -0.993),
            atom(5, AtomicSymbol.H, -0.885, 1.190, 0.000),
            atom(6, AtomicSymbol.H, -0.885, -1.190, 0.000),
            atom(7, AtomicSymbol.H, 0.885, 1.190, 0.000),
            atom(8, AtomicSymbol.H, 0.885, -1.190, 0.000),
        )
    },
    local_bonds=frozenset(
        {
            bond(1, 2),
            bond(1, 5),
            bond(1, 6),
            bond(2, 7),
            bond(2, 8),
        }
    ),
    systems=(
        (
            SystemId(1),
            mk_bonding_system(NonNegative(2), frozenset({_edge(b1, h3), _edge(b2, h3)}), "bridge_h3_3c2e"),
        ),
        (
            SystemId(2),
            mk_bonding_system(NonNegative(2), frozenset({_edge(b1, h4), _edge(b2, h4)}), "bridge_h4_3c2e"),
        ),
    ),
)
