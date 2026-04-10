from __future__ import annotations

from ..chem.constants import element_attributes, element_shells
from ..chem.coordinate import Coordinate, mk_angstrom
from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import Atom, AtomicSymbol, Molecule


_ATOMS_DATA = (
    (1, AtomicSymbol.C, -1.2131, -0.6884, 0.0),
    (2, AtomicSymbol.C, -1.2028, 0.7064, 0.0),
    (3, AtomicSymbol.C, -0.0103, -1.3948, 0.0),
    (4, AtomicSymbol.C, 0.0104, 1.3948, 0.0),
    (5, AtomicSymbol.C, 1.2028, -0.7063, 0.0),
    (6, AtomicSymbol.C, 1.2131, 0.6884, 0.0),
    (7, AtomicSymbol.H, -2.1577, -1.2244, 0.0),
    (8, AtomicSymbol.H, -2.1393, 1.2564, 0.0),
    (9, AtomicSymbol.H, -0.0184, -2.4809, 0.0),
    (10, AtomicSymbol.H, 0.0184, 2.4808, 0.0),
    (11, AtomicSymbol.H, 2.1394, -1.2563, 0.0),
    (12, AtomicSymbol.H, 2.1577, 1.2245, 0.0),
)


def _atom(atom_index: int, symbol: AtomicSymbol, x: float, y: float, z: float) -> Atom:
    atom_id = AtomId(atom_index)
    return Atom(
        atom_id=atom_id,
        attributes=element_attributes(symbol),
        coordinate=Coordinate(mk_angstrom(x), mk_angstrom(y), mk_angstrom(z)),
        shells=element_shells(symbol),
        formal_charge=0,
    )


def _edge_from_index_pair(atom_pair: tuple[int, int]) -> Edge:
    return Edge(AtomId(atom_pair[0]), AtomId(atom_pair[1]))


ring_carbon_ids = tuple(range(1, 7))
hydrogen_ids = tuple(range(7, 13))
ring_edges = tuple(zip(ring_carbon_ids, ring_carbon_ids[1:] + ring_carbon_ids[:1]))
sigma_edges = ring_edges + tuple(zip(ring_carbon_ids, hydrogen_ids))

benzene = Molecule(
    atoms={
        AtomId(atom_index): _atom(atom_index, symbol, x, y, z)
        for atom_index, symbol, x, y, z in _ATOMS_DATA
    },
    local_bonds=frozenset(_edge_from_index_pair(atom_pair) for atom_pair in sigma_edges),
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(len(ring_carbon_ids)),
                frozenset(_edge_from_index_pair(atom_pair) for atom_pair in ring_edges),
                "pi_ring",
            ),
        ),
    ),
)

benzene_pretty = benzene
