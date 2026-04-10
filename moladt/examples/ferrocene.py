from __future__ import annotations

from ..chem.constants import element_attributes, element_shells
from ..chem.coordinate import Coordinate, mk_angstrom
from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
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


def _edge_set(atom_pairs: tuple[tuple[AtomId, AtomId], ...]) -> frozenset[Edge]:
    return frozenset(Edge(atom_a, atom_b) for atom_a, atom_b in atom_pairs)


fe = AtomId(1)
ring1_c = tuple(AtomId(index) for index in range(2, 7))
ring2_c = tuple(AtomId(index) for index in range(7, 12))
ring1_h = tuple(AtomId(index) for index in range(12, 17))
ring2_h = tuple(AtomId(index) for index in range(17, 22))

_RING1_CARBON_COORDS = (
    (1.1800, 0.0000, 1.6600),
    (0.3647, 1.1220, 1.6600),
    (-0.9547, 0.6935, 1.6600),
    (-0.9547, -0.6935, 1.6600),
    (0.3647, -1.1220, 1.6600),
)
_RING2_CARBON_COORDS = (
    (0.9547, 0.6935, -1.6600),
    (-0.3647, 1.1220, -1.6600),
    (-1.1800, 0.0000, -1.6600),
    (-0.3647, -1.1220, -1.6600),
    (0.9547, -0.6935, -1.6600),
)
_RING1_HYDROGEN_COORDS = (
    (2.2700, 0.0000, 1.6600),
    (0.7016, 2.1582, 1.6600),
    (-1.8364, 1.3338, 1.6600),
    (-1.8364, -1.3338, 1.6600),
    (0.7016, -2.1582, 1.6600),
)
_RING2_HYDROGEN_COORDS = (
    (1.8364, 1.3338, -1.6600),
    (-0.7016, 2.1582, -1.6600),
    (-2.2700, 0.0000, -1.6600),
    (-0.7016, -2.1582, -1.6600),
    (1.8364, -1.3338, -1.6600),
)


def _ring_pairs(atom_ids: tuple[AtomId, ...]) -> tuple[tuple[AtomId, AtomId], ...]:
    return tuple(zip(atom_ids, atom_ids[1:] + atom_ids[:1]))


ring1_cc = _ring_pairs(ring1_c)
ring2_cc = _ring_pairs(ring2_c)
ring1_ch = tuple(zip(ring1_c, ring1_h))
ring2_ch = tuple(zip(ring2_c, ring2_h))
fe_to_ring1 = tuple((fe, atom_id) for atom_id in ring1_c)
fe_to_ring2 = tuple((fe, atom_id) for atom_id in ring2_c)


ferrocene_pretty = Molecule(
    atoms={
        fe: _atom(fe, AtomicSymbol.Fe, 0.0000, 0.0000, 0.0000),
        **{atom_id: _atom(atom_id, AtomicSymbol.C, *xyz) for atom_id, xyz in zip(ring1_c, _RING1_CARBON_COORDS)},
        **{atom_id: _atom(atom_id, AtomicSymbol.C, *xyz) for atom_id, xyz in zip(ring2_c, _RING2_CARBON_COORDS)},
        **{atom_id: _atom(atom_id, AtomicSymbol.H, *xyz) for atom_id, xyz in zip(ring1_h, _RING1_HYDROGEN_COORDS)},
        **{atom_id: _atom(atom_id, AtomicSymbol.H, *xyz) for atom_id, xyz in zip(ring2_h, _RING2_HYDROGEN_COORDS)},
    },
    local_bonds=_edge_set(ring1_cc + ring2_cc + ring1_ch + ring2_ch),
    systems=(
        (
            SystemId(1),
            mk_bonding_system(NonNegative(6), _edge_set(fe_to_ring1 + ring1_cc), "cp1_pi"),
        ),
        (
            SystemId(2),
            mk_bonding_system(NonNegative(6), _edge_set(fe_to_ring2 + ring2_cc), "cp2_pi"),
        ),
        (
            SystemId(3),
            mk_bonding_system(NonNegative(6), _edge_set(fe_to_ring1 + fe_to_ring2), "fe_backdonation"),
        ),
    ),
)
