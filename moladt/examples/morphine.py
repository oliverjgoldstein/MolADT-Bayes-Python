from __future__ import annotations

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import (
    AtomicSymbol,
    Molecule,
    SmilesAtomStereo,
    SmilesAtomStereoClass,
    SmilesStereochemistry,
)
from ._literal import atom, atom_map, sigma_bonds


MORPHINE_RING_CLOSURE_SMILES = "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"

# The atom numbering follows the non-cyclic morphine sketch in the classic
# ring-closure figure. In that sketch, the five broken edges that later become
# SMILES ring closures are ordinary sigma edges here:
#   1 -> (1, 11)
#   2 -> (2, 8)
#   3 -> (7, 18)
#   4 -> (9, 21)
#   5 -> (10, 16)
#
# The standard stereochemical boundary string used in the docs is:
#   CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5
# Its five atom-centered stereochemistry flags are preserved below as SMILES
# annotations on the explicit Dietz object.
#
_SIGMA_EDGES = (
    (1, 2),
    (1, 11),
    (2, 3),
    (2, 8),
    (3, 4),
    (3, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (7, 18),
    (8, 9),
    (8, 10),
    (9, 21),
    (10, 11),
    (10, 16),
    (11, 12),
    (12, 13),
    (12, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (19, 21),
)

_ALKENE_EDGES = (
    (5, 6),
)

_PHENYL_PI_RING_EDGES = (
    (10, 11),
    (11, 12),
    (12, 14),
    (14, 15),
    (15, 16),
    (10, 16),
)


def _edge_from_index_pair(atom_pair: tuple[int, int]) -> Edge:
    return Edge(AtomId(atom_pair[0]), AtomId(atom_pair[1]))

morphine_pretty = Molecule(
    atoms=atom_map(
        atom(1, AtomicSymbol.O, 0.000, 0.000, 0.100),
        atom(2, AtomicSymbol.C, 1.000, 0.800, 0.450),
        atom(3, AtomicSymbol.C, 2.000, 0.800, -0.100),
        atom(4, AtomicSymbol.O, 2.000, -0.400, -0.550),
        atom(5, AtomicSymbol.C, 3.000, 0.800, 0.350),
        atom(6, AtomicSymbol.C, 4.000, 0.800, 0.750),
        atom(7, AtomicSymbol.C, 5.000, 0.800, 0.200),
        atom(8, AtomicSymbol.C, 1.800, 2.000, 0.800),
        atom(9, AtomicSymbol.C, 2.800, 2.800, 1.100),
        atom(10, AtomicSymbol.C, 3.800, 2.000, 0.600),
        atom(11, AtomicSymbol.C, 0.800, 2.000, 0.150),
        atom(12, AtomicSymbol.C, 1.200, 3.200, 0.550),
        atom(13, AtomicSymbol.O, 0.400, 4.000, 0.300),
        atom(14, AtomicSymbol.C, 2.400, 3.800, 0.950),
        atom(15, AtomicSymbol.C, 3.600, 3.800, 0.700),
        atom(16, AtomicSymbol.C, 4.200, 2.800, 0.200),
        atom(17, AtomicSymbol.C, 5.400, 2.800, -0.200),
        atom(18, AtomicSymbol.C, 6.200, 1.800, -0.550),
        atom(19, AtomicSymbol.N, 7.200, 1.800, -0.850),
        atom(20, AtomicSymbol.C, 8.200, 2.400, -1.100),
        atom(21, AtomicSymbol.C, 6.000, 2.800, -0.350),
    ),
    local_bonds=sigma_bonds(*_SIGMA_EDGES),
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(2),
                frozenset(_edge_from_index_pair(atom_pair) for atom_pair in _ALKENE_EDGES),
                "alkene_bridge",
            ),
        ),
        (
            SystemId(2),
            mk_bonding_system(
                NonNegative(6),
                frozenset(_edge_from_index_pair(atom_pair) for atom_pair in _PHENYL_PI_RING_EDGES),
                "phenyl_pi_ring",
            ),
        ),
    ),
    smiles_stereochemistry=SmilesStereochemistry(
        atom_stereo=(
            SmilesAtomStereo(AtomId(2), SmilesAtomStereoClass.TETRAHEDRAL, 1, "@"),
            SmilesAtomStereo(AtomId(3), SmilesAtomStereoClass.TETRAHEDRAL, 2, "@@"),
            SmilesAtomStereo(AtomId(7), SmilesAtomStereoClass.TETRAHEDRAL, 1, "@"),
            SmilesAtomStereo(AtomId(8), SmilesAtomStereoClass.TETRAHEDRAL, 1, "@"),
            SmilesAtomStereo(AtomId(18), SmilesAtomStereoClass.TETRAHEDRAL, 1, "@"),
        ),
    ),
)
