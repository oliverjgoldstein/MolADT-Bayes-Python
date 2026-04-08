from __future__ import annotations

from ..chem.constants import element_attributes, element_shells
from ..chem.coordinate import Coordinate, mk_angstrom
from ..chem.dietz import AtomId, NonNegative, SystemId, mk_bonding_system, mk_edge
from ..chem.molecule import (
    Atom,
    AtomicSymbol,
    Molecule,
    SmilesAtomStereo,
    SmilesAtomStereoClass,
    SmilesStereochemistry,
)


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
# Coordinates are schematic rather than experimental. The point of this
# example is the explicit Dietz constitution: local sigma edges plus an alkene
# pool and a phenyl pi-ring pool.
_ATOMS_DATA = (
    (1, AtomicSymbol.O, 0.0, 0.0, 0.0),
    (2, AtomicSymbol.C, 1.0, 0.8, 0.0),
    (3, AtomicSymbol.C, 2.0, 0.8, 0.0),
    (4, AtomicSymbol.O, 2.0, -0.4, 0.0),
    (5, AtomicSymbol.C, 3.0, 0.8, 0.0),
    (6, AtomicSymbol.C, 4.0, 0.8, 0.0),
    (7, AtomicSymbol.C, 5.0, 0.8, 0.0),
    (8, AtomicSymbol.C, 1.8, 2.0, 0.0),
    (9, AtomicSymbol.C, 2.8, 2.8, 0.0),
    (10, AtomicSymbol.C, 3.8, 2.0, 0.0),
    (11, AtomicSymbol.C, 0.8, 2.0, 0.0),
    (12, AtomicSymbol.C, 1.2, 3.2, 0.0),
    (13, AtomicSymbol.O, 0.4, 4.0, 0.0),
    (14, AtomicSymbol.C, 2.4, 3.8, 0.0),
    (15, AtomicSymbol.C, 3.6, 3.8, 0.0),
    (16, AtomicSymbol.C, 4.2, 2.8, 0.0),
    (17, AtomicSymbol.C, 5.4, 2.8, 0.0),
    (18, AtomicSymbol.C, 6.2, 1.8, 0.0),
    (19, AtomicSymbol.N, 7.2, 1.8, 0.0),
    (20, AtomicSymbol.C, 8.2, 2.4, 0.0),
    (21, AtomicSymbol.C, 6.0, 2.8, 0.0),
)

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


def _atom(atom_index: int, symbol: AtomicSymbol, x: float, y: float, z: float) -> Atom:
    atom_id = AtomId(atom_index)
    return Atom(
        atom_id=atom_id,
        attributes=element_attributes(symbol),
        coordinate=Coordinate(mk_angstrom(x), mk_angstrom(y), mk_angstrom(z)),
        shells=element_shells(symbol),
        formal_charge=0,
    )


morphine_pretty = Molecule(
    atoms={
        AtomId(atom_index): _atom(atom_index, symbol, x, y, z)
        for atom_index, symbol, x, y, z in _ATOMS_DATA
    },
    local_bonds=frozenset(mk_edge(AtomId(a), AtomId(b)) for a, b in _SIGMA_EDGES),
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(2),
                frozenset(mk_edge(AtomId(a), AtomId(b)) for a, b in _ALKENE_EDGES),
                "alkene_bridge",
            ),
        ),
        (
            SystemId(2),
            mk_bonding_system(
                NonNegative(6),
                frozenset(mk_edge(AtomId(a), AtomId(b)) for a, b in _PHENYL_PI_RING_EDGES),
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
