from __future__ import annotations

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import AtomicSymbol, Molecule
from ._literal import atom


benzene = Molecule(
    atoms={
        AtomId(1): atom(1, AtomicSymbol.C, 2.866, 1.000, 0.000),
        AtomId(2): atom(2, AtomicSymbol.C, 2.000, 0.500, 0.000),
        AtomId(3): atom(3, AtomicSymbol.C, 3.732, 0.500, 0.000),
        AtomId(4): atom(4, AtomicSymbol.C, 2.000, -0.500, 0.000),
        AtomId(5): atom(5, AtomicSymbol.C, 3.732, -0.500, 0.000),
        AtomId(6): atom(6, AtomicSymbol.C, 2.866, -1.000, 0.000),
        AtomId(7): atom(7, AtomicSymbol.H, 2.866, 1.620, 0.000),
        AtomId(8): atom(8, AtomicSymbol.H, 1.463, 0.810, 0.000),
        AtomId(9): atom(9, AtomicSymbol.H, 4.269, 0.810, 0.000),
        AtomId(10): atom(10, AtomicSymbol.H, 1.463, -0.810, 0.000),
        AtomId(11): atom(11, AtomicSymbol.H, 4.269, -0.810, 0.000),
        AtomId(12): atom(12, AtomicSymbol.H, 2.866, -1.620, 0.000),
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
                        Edge(AtomId(1), AtomId(7)),
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
                        Edge(AtomId(2), AtomId(4)),
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
                        Edge(AtomId(2), AtomId(8)),
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
                        Edge(AtomId(3), AtomId(5)),
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
                        Edge(AtomId(3), AtomId(9)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(8),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(4), AtomId(6)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(9),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(4), AtomId(10)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(10),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(5), AtomId(6)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(11),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(5), AtomId(11)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(12),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(6), AtomId(12)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(13),
            mk_bonding_system(
                NonNegative(6),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(2)),
                        Edge(AtomId(1), AtomId(3)),
                        Edge(AtomId(2), AtomId(4)),
                        Edge(AtomId(3), AtomId(5)),
                        Edge(AtomId(4), AtomId(6)),
                        Edge(AtomId(5), AtomId(6)),
                    }
                ),
                "pi_ring",
            ),
        ),
    ),
)

benzene_pretty = benzene
