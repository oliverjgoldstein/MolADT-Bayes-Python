from __future__ import annotations

from pathlib import Path

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import Molecule
from ..io.sdf import read_sdf_record


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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SDF_RECORD = read_sdf_record(_PROJECT_ROOT / "molecules" / "ferrocene.sdf")


ferrocene_pretty = Molecule(
    atoms=_SDF_RECORD.molecule.atoms,
    local_bonds=_SDF_RECORD.molecule.local_bonds,
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
