from __future__ import annotations

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import AtomicSymbol, Molecule
from ._literal import atom, bond


def _edge_set(atom_pairs: tuple[tuple[AtomId, AtomId], ...]) -> frozenset[Edge]:
    return frozenset(Edge(atom_a, atom_b) for atom_a, atom_b in atom_pairs)


fe = AtomId(1)
ring1_c = tuple(AtomId(index) for index in range(2, 7))
ring2_c = tuple(AtomId(index) for index in range(7, 12))
ring1_h = tuple(AtomId(index) for index in range(12, 17))
ring2_h = tuple(AtomId(index) for index in range(17, 22))


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
        atom.atom_id: atom
        for atom in (
            atom(1, AtomicSymbol.Fe, 0.000, 0.000, 0.000),
            atom(2, AtomicSymbol.C, 1.180, 0.000, 1.660),
            atom(3, AtomicSymbol.C, 0.365, 1.122, 1.660),
            atom(4, AtomicSymbol.C, -0.955, 0.694, 1.660),
            atom(5, AtomicSymbol.C, -0.955, -0.694, 1.660),
            atom(6, AtomicSymbol.C, 0.365, -1.122, 1.660),
            atom(7, AtomicSymbol.C, 0.955, 0.694, -1.660),
            atom(8, AtomicSymbol.C, -0.365, 1.122, -1.660),
            atom(9, AtomicSymbol.C, -1.180, 0.000, -1.660),
            atom(10, AtomicSymbol.C, -0.365, -1.122, -1.660),
            atom(11, AtomicSymbol.C, 0.955, -0.694, -1.660),
            atom(12, AtomicSymbol.H, 2.270, 0.000, 1.660),
            atom(13, AtomicSymbol.H, 0.702, 2.158, 1.660),
            atom(14, AtomicSymbol.H, -1.836, 1.334, 1.660),
            atom(15, AtomicSymbol.H, -1.836, -1.334, 1.660),
            atom(16, AtomicSymbol.H, 0.702, -2.158, 1.660),
            atom(17, AtomicSymbol.H, 1.836, 1.334, -1.660),
            atom(18, AtomicSymbol.H, -0.702, 2.158, -1.660),
            atom(19, AtomicSymbol.H, -2.270, 0.000, -1.660),
            atom(20, AtomicSymbol.H, -0.702, -2.158, -1.660),
            atom(21, AtomicSymbol.H, 1.836, -1.334, -1.660),
        )
    },
    local_bonds=frozenset(
        {
            bond(2, 3),
            bond(2, 6),
            bond(2, 12),
            bond(3, 4),
            bond(3, 13),
            bond(4, 5),
            bond(4, 14),
            bond(5, 6),
            bond(5, 15),
            bond(6, 16),
            bond(7, 8),
            bond(7, 11),
            bond(7, 17),
            bond(8, 9),
            bond(8, 18),
            bond(9, 10),
            bond(9, 19),
            bond(10, 11),
            bond(10, 20),
            bond(11, 21),
        }
    ),
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
