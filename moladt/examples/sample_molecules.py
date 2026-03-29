from __future__ import annotations

from ..chem.constants import element_attributes, element_shells
from ..chem.coordinate import Coordinate, mk_angstrom
from ..chem.dietz import AtomId, mk_edge
from ..chem.molecule import Atom, AtomicSymbol, Molecule


def _atom(atom_index: int, symbol: AtomicSymbol, x: float, y: float, z: float) -> Atom:
    atom_id = AtomId(atom_index)
    return Atom(
        atom_id=atom_id,
        attributes=element_attributes(symbol),
        coordinate=Coordinate(mk_angstrom(x), mk_angstrom(y), mk_angstrom(z)),
        shells=element_shells(symbol),
        formal_charge=0,
    )


hydrogen = Molecule(
    atoms={
        AtomId(1): _atom(1, AtomicSymbol.H, 0.0, 0.0, 0.0),
        AtomId(2): _atom(2, AtomicSymbol.H, 0.74, 0.0, 0.0),
    },
    local_bonds=frozenset({mk_edge(AtomId(1), AtomId(2))}),
    systems=(),
)


oxygen = Molecule(
    atoms={
        AtomId(1): _atom(1, AtomicSymbol.O, 0.0, 0.0, 0.0),
        AtomId(2): _atom(2, AtomicSymbol.O, 1.21, 0.0, 0.0),
    },
    local_bonds=frozenset({mk_edge(AtomId(1), AtomId(2))}),
    systems=(),
)


water = Molecule(
    atoms={
        AtomId(1): _atom(1, AtomicSymbol.O, 0.0, 0.0, 0.0),
        AtomId(2): _atom(2, AtomicSymbol.H, 0.96, 0.0, 0.0),
        AtomId(3): _atom(3, AtomicSymbol.H, -0.32, 0.90, 0.0),
    },
    local_bonds=frozenset({mk_edge(AtomId(1), AtomId(2)), mk_edge(AtomId(1), AtomId(3))}),
    systems=(),
)


methane = Molecule(
    atoms={
        AtomId(1): _atom(1, AtomicSymbol.C, 0.0, 0.0, 0.0),
        AtomId(2): _atom(2, AtomicSymbol.H, 1.09, 0.0, 0.0),
        AtomId(3): _atom(3, AtomicSymbol.H, -1.09, 0.0, 0.0),
        AtomId(4): _atom(4, AtomicSymbol.H, 0.0, 1.09, 0.0),
        AtomId(5): _atom(5, AtomicSymbol.H, 0.0, -1.09, 0.0),
    },
    local_bonds=frozenset(
        {
            mk_edge(AtomId(1), AtomId(2)),
            mk_edge(AtomId(1), AtomId(3)),
            mk_edge(AtomId(1), AtomId(4)),
            mk_edge(AtomId(1), AtomId(5)),
        }
    ),
    systems=(),
)

