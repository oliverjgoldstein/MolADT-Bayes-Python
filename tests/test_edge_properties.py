from __future__ import annotations

from moladt.chem.dietz import (
    AtomId,
    Edge,
    NonNegative,
    SystemId,
    mk_bonding_system,
    mk_edge,
)
from moladt.chem.molecule import AtomicSymbol, Molecule, same_molecule
from moladt.chem.molecule_ops import add_sigma
from moladt.examples._literal import atom


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


def test_same_molecule_ignores_container_ordering() -> None:
    base = Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.O, 0.0, 0.0, 0.0),
            AtomId(2): atom(2, AtomicSymbol.H, 0.0, 0.8, 0.0),
            AtomId(3): atom(3, AtomicSymbol.H, 0.8, 0.0, 0.0),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(3)),
            }
        ),
        systems=(
            (
                SystemId(1),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset({Edge(AtomId(1), AtomId(2))}),
                    "oh_a",
                ),
            ),
            (
                SystemId(2),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset({Edge(AtomId(1), AtomId(3))}),
                    "oh_b",
                ),
            ),
        ),
    )
    reordered = Molecule(
        atoms={
            AtomId(3): base.atoms[AtomId(3)],
            AtomId(2): base.atoms[AtomId(2)],
            AtomId(1): base.atoms[AtomId(1)],
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(2), AtomId(1)),
                Edge(AtomId(3), AtomId(1)),
            }
        ),
        systems=(base.systems[1], base.systems[0]),
    )

    assert same_molecule(base, reordered)


def test_same_molecule_detects_structural_changes() -> None:
    base = Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.O, 0.0, 0.0, 0.0),
            AtomId(2): atom(2, AtomicSymbol.H, 0.0, 0.8, 0.0),
            AtomId(3): atom(3, AtomicSymbol.H, 0.8, 0.0, 0.0),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(3)),
            }
        ),
        systems=(),
    )
    changed = Molecule(
        atoms=base.atoms,
        local_bonds=frozenset({Edge(AtomId(1), AtomId(2))}),
        systems=base.systems,
    )

    assert not same_molecule(base, changed)
