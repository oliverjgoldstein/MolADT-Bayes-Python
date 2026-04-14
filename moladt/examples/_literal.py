from __future__ import annotations

from ..chem.constants import element_attributes, element_shells
from ..chem.coordinate import Coordinate, mk_angstrom
from ..chem.dietz import AtomId, Edge
from ..chem.molecule import Atom, AtomicSymbol


def atom(
    atom_index: int,
    symbol: AtomicSymbol,
    x: float,
    y: float,
    z: float,
    *,
    formal_charge: int = 0,
) -> Atom:
    atom_id = AtomId(atom_index)
    return Atom(
        atom_id=atom_id,
        attributes=element_attributes(symbol),
        coordinate=Coordinate(mk_angstrom(x), mk_angstrom(y), mk_angstrom(z)),
        shells=element_shells(symbol),
        formal_charge=formal_charge,
    )


def bond(atom_a: int, atom_b: int) -> Edge:
    return Edge(AtomId(atom_a), AtomId(atom_b))
