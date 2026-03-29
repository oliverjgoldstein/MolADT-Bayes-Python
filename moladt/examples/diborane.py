from __future__ import annotations

from ..chem.constants import element_attributes, element_shells
from ..chem.coordinate import Coordinate, mk_angstrom
from ..chem.dietz import AtomId, NonNegative, SystemId, mk_bonding_system, mk_edge
from ..chem.molecule import Atom, AtomicSymbol, Molecule


def _coord(x: float, y: float, z: float) -> Coordinate:
    return Coordinate(mk_angstrom(x), mk_angstrom(y), mk_angstrom(z))


def _atom(atom_id: AtomId, symbol: AtomicSymbol, x: float, y: float, z: float) -> Atom:
    return Atom(
        atom_id=atom_id,
        attributes=element_attributes(symbol),
        coordinate=_coord(x, y, z),
        shells=element_shells(symbol),
        formal_charge=0,
    )


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
        b1: _atom(b1, AtomicSymbol.B, -0.8850, 0.0000, 0.0000),
        b2: _atom(b2, AtomicSymbol.B, 0.8850, 0.0000, 0.0000),
        h3: _atom(h3, AtomicSymbol.H, 0.0000, 0.0000, 0.9928),
        h4: _atom(h4, AtomicSymbol.H, 0.0000, 0.0000, -0.9928),
        h5: _atom(h5, AtomicSymbol.H, -0.8850, 1.1900, 0.0000),
        h6: _atom(h6, AtomicSymbol.H, -0.8850, -1.1900, 0.0000),
        h7: _atom(h7, AtomicSymbol.H, 0.8850, 1.1900, 0.0000),
        h8: _atom(h8, AtomicSymbol.H, 0.8850, -1.1900, 0.0000),
    },
    local_bonds=frozenset(
        {
            mk_edge(b1, b2),
            mk_edge(b1, h5),
            mk_edge(b1, h6),
            mk_edge(b2, h7),
            mk_edge(b2, h8),
        }
    ),
    systems=(
        (
            SystemId(1),
            mk_bonding_system(NonNegative(2), frozenset({mk_edge(b1, h3), mk_edge(b2, h3)}), "bridge_h3_3c2e"),
        ),
        (
            SystemId(2),
            mk_bonding_system(NonNegative(2), frozenset({mk_edge(b1, h4), mk_edge(b2, h4)}), "bridge_h4_3c2e"),
        ),
    ),
)

