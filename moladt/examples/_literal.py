"""Helpers for the paper-facing literal example molecules.

Every example atom built here includes the element's orbital shell table, so
the manuscript examples stay fully typed instead of dropping orbital data for
brevity.
"""

from __future__ import annotations

from ..chem.constants import element_attributes, element_shells
from ..chem.coordinate import Coordinate, mk_angstrom
from ..chem.dietz import AtomId
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
