from __future__ import annotations

from pathlib import Path

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import (
    Molecule,
    SmilesAtomStereo,
    SmilesAtomStereoClass,
    SmilesStereochemistry,
)
from ..io.sdf import read_sdf_record


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


_PROJECT_ROOT = Path(__file__).resolve().parents[2]

morphine_pretty = Molecule(
    atoms=read_sdf_record(_PROJECT_ROOT / "molecules" / "morphine.sdf").molecule.atoms,
    local_bonds=read_sdf_record(_PROJECT_ROOT / "molecules" / "morphine.sdf").molecule.local_bonds,
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
