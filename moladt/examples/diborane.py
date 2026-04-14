from __future__ import annotations

from pathlib import Path

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import Molecule
from ..io.sdf import read_sdf_record


def _edge(atom_a: AtomId, atom_b: AtomId) -> Edge:
    return Edge(atom_a, atom_b)


b1 = AtomId(1)
b2 = AtomId(2)
h3 = AtomId(3)
h4 = AtomId(4)
h5 = AtomId(5)
h6 = AtomId(6)
h7 = AtomId(7)
h8 = AtomId(8)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SDF_RECORD = read_sdf_record(_PROJECT_ROOT / "molecules" / "diborane.sdf")


diborane_pretty = Molecule(
    atoms=_SDF_RECORD.molecule.atoms,
    local_bonds=_SDF_RECORD.molecule.local_bonds,
    systems=(
        (
            SystemId(1),
            mk_bonding_system(NonNegative(2), frozenset({_edge(b1, h3), _edge(b2, h3)}), "bridge_h3_3c2e"),
        ),
        (
            SystemId(2),
            mk_bonding_system(NonNegative(2), frozenset({_edge(b1, h4), _edge(b2, h4)}), "bridge_h4_3c2e"),
        ),
    ),
)
