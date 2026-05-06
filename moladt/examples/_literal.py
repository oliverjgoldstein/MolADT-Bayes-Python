"""Helpers for the paper-facing literal example molecules."""

from __future__ import annotations

from ..chem.constants import element_attributes
from ..chem.coordinate import Coordinate, mk_angstrom
from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
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
        formal_charge=formal_charge,
    )


def single_covalent_systems(start_id: int, edges: tuple[Edge, ...]):
    def build(item: tuple[int, Edge]):
        offset, edge = item
        return (
            SystemId(start_id + offset),
            mk_bonding_system(NonNegative(2), frozenset({edge})),
        )

    return tuple(map(build, enumerate(edges)))
