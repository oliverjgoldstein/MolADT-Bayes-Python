from __future__ import annotations

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import AtomicSymbol, Molecule
from ._literal import atom, atom_map, sigma_bonds


_RING_ATOMS = tuple(range(1, 7))
_HYDROGEN_ATOMS = tuple(range(7, 13))
_RING_EDGES = tuple(zip(_RING_ATOMS, _RING_ATOMS[1:] + _RING_ATOMS[:1]))
_CARBON_HYDROGEN_EDGES = tuple(zip(_RING_ATOMS, _HYDROGEN_ATOMS))


def _edge(atom_a: int, atom_b: int) -> Edge:
    return Edge(AtomId(atom_a), AtomId(atom_b))


benzene = Molecule(
    atoms=atom_map(
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
    ),
    local_bonds=sigma_bonds(
        *_RING_EDGES,
        *_CARBON_HYDROGEN_EDGES,
    ),
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(len(_RING_ATOMS)),
                frozenset(_edge(atom_a, atom_b) for atom_a, atom_b in _RING_EDGES),
                "pi_ring",
            ),
        ),
    ),
)

benzene_pretty = benzene
