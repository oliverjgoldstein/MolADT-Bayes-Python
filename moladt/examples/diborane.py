from __future__ import annotations

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import AtomicSymbol, Molecule
from ._literal import atom


diborane_pretty = Molecule(
    atoms={
        AtomId(1): atom(1, AtomicSymbol.B, -0.885, 0.000, 0.000),
        AtomId(2): atom(2, AtomicSymbol.B, 0.885, 0.000, 0.000),
        AtomId(3): atom(3, AtomicSymbol.H, 0.000, 0.000, 0.993),
        AtomId(4): atom(4, AtomicSymbol.H, 0.000, 0.000, -0.993),
        AtomId(5): atom(5, AtomicSymbol.H, -0.885, 1.190, 0.000),
        AtomId(6): atom(6, AtomicSymbol.H, -0.885, -1.190, 0.000),
        AtomId(7): atom(7, AtomicSymbol.H, 0.885, 1.190, 0.000),
        AtomId(8): atom(8, AtomicSymbol.H, 0.885, -1.190, 0.000),
    },
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(3)),
                        Edge(AtomId(2), AtomId(3)),
                    }
                ),
                "bridge_h3_3c2e",
            ),
        ),
        (
            SystemId(2),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(4)),
                        Edge(AtomId(2), AtomId(4)),
                    }
                ),
                "bridge_h4_3c2e",
            ),
        ),
        (
            SystemId(3),
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
        (
            SystemId(5),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(6)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(6),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(2), AtomId(7)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(7),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(2), AtomId(8)),
                    }
                ),
                None,
            ),
        ),
    ),
)
