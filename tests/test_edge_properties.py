from __future__ import annotations

from moladt.chem.dietz import AtomId
from moladt.chem.molecule import Molecule, add_sigma
from moladt.chem.dietz import mk_edge


def test_edge_canonicalization() -> None:
    edge = mk_edge(AtomId(9), AtomId(2))
    assert edge.a == AtomId(2)
    assert edge.b == AtomId(9)


def test_add_sigma_is_idempotent() -> None:
    molecule = Molecule(atoms={}, local_bonds=frozenset(), systems=())
    once = add_sigma(AtomId(1), AtomId(2), molecule)
    twice = add_sigma(AtomId(1), AtomId(2), once)
    assert len(once.local_bonds) == 1
    assert twice.local_bonds == once.local_bonds

