from __future__ import annotations

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import AtomicSymbol, Molecule
from ._literal import atom


hydrogen = Molecule(
    atoms={
        AtomId(1): atom(1, AtomicSymbol.H, 0.000, 0.000, -0.370),
        AtomId(2): atom(2, AtomicSymbol.H, 0.000, 0.000, 0.370),
    },
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(2)),
                    }
                ),
                None,
            ),
        ),
    ),
)


oxygen = Molecule(
    atoms={
        AtomId(1): atom(1, AtomicSymbol.O, 0.000, 0.000, -0.605),
        AtomId(2): atom(2, AtomicSymbol.O, 0.000, 0.000, 0.605),
    },
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(4),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(2)),
                    }
                ),
                None,
            ),
        ),
    ),
)


sodium_chloride = Molecule(
    atoms={
        AtomId(1): atom(1, AtomicSymbol.Na, 0.000, 0.000, 0.000, formal_charge=1),
        AtomId(2): atom(2, AtomicSymbol.Cl, 2.360, 0.000, 0.000, formal_charge=-1),
    },
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(0),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(2)),
                    }
                ),
                "ionic",
            ),
        ),
    ),
)


water = Molecule(
    atoms={
        AtomId(1): atom(1, AtomicSymbol.H, 0.002, -0.004, 0.002),
        AtomId(2): atom(2, AtomicSymbol.O, -0.011, 0.963, 0.007),
        AtomId(3): atom(3, AtomicSymbol.H, 0.867, 1.368, 0.001),
    },
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(2)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(2),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(2), AtomId(3)),
                    }
                ),
                None,
            ),
        ),
    ),
)


methane = Molecule(
    atoms={
        AtomId(1): atom(1, AtomicSymbol.C, 0.000, 0.000, 0.000),
        AtomId(2): atom(2, AtomicSymbol.H, 0.629, 0.629, 0.629),
        AtomId(3): atom(3, AtomicSymbol.H, -0.629, -0.629, 0.629),
        AtomId(4): atom(4, AtomicSymbol.H, -0.629, 0.629, -0.629),
        AtomId(5): atom(5, AtomicSymbol.H, 0.629, -0.629, -0.629),
    },
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(2)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(2),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(3)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(3),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(4)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(4),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(5)),
                    }
                ),
                None,
            ),
        ),
    ),
)
