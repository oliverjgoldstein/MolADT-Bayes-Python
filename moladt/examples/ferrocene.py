from __future__ import annotations

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import AtomicSymbol, Molecule
from ._literal import atom


ferrocene_pretty = Molecule(
    atoms={
        AtomId(1): atom(1, AtomicSymbol.Fe, 0.000, 0.000, 0.000, formal_charge=2),
        AtomId(2): atom(2, AtomicSymbol.C, 1.194, 0.000, 1.660, formal_charge=-1),
        AtomId(3): atom(3, AtomicSymbol.C, 0.369, 1.136, 1.660),
        AtomId(4): atom(4, AtomicSymbol.C, -0.966, 0.702, 1.660),
        AtomId(5): atom(5, AtomicSymbol.C, -0.966, -0.702, 1.660),
        AtomId(6): atom(6, AtomicSymbol.C, 0.369, -1.136, 1.660),
        AtomId(7): atom(7, AtomicSymbol.C, 0.966, 0.702, -1.660, formal_charge=-1),
        AtomId(8): atom(8, AtomicSymbol.C, -0.369, 1.136, -1.660),
        AtomId(9): atom(9, AtomicSymbol.C, -1.194, 0.000, -1.660),
        AtomId(10): atom(10, AtomicSymbol.C, -0.369, -1.136, -1.660),
        AtomId(11): atom(11, AtomicSymbol.C, 0.966, -0.702, -1.660),
        AtomId(12): atom(12, AtomicSymbol.H, 2.280, 0.000, 1.565),
        AtomId(13): atom(13, AtomicSymbol.H, 0.705, 2.168, 1.565),
        AtomId(14): atom(14, AtomicSymbol.H, -1.845, 1.340, 1.565),
        AtomId(15): atom(15, AtomicSymbol.H, -1.845, -1.340, 1.565),
        AtomId(16): atom(16, AtomicSymbol.H, 0.705, -2.168, 1.565),
        AtomId(17): atom(17, AtomicSymbol.H, 1.845, 1.340, -1.565),
        AtomId(18): atom(18, AtomicSymbol.H, -0.705, 2.168, -1.565),
        AtomId(19): atom(19, AtomicSymbol.H, -2.280, 0.000, -1.565),
        AtomId(20): atom(20, AtomicSymbol.H, -0.705, -2.168, -1.565),
        AtomId(21): atom(21, AtomicSymbol.H, 1.845, -1.340, -1.565),
    },
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(6),
                frozenset(
                    {
                        Edge(AtomId(2), AtomId(3)),
                        Edge(AtomId(2), AtomId(6)),
                        Edge(AtomId(3), AtomId(4)),
                        Edge(AtomId(4), AtomId(5)),
                        Edge(AtomId(5), AtomId(6)),
                        Edge(AtomId(1), AtomId(2)),
                        Edge(AtomId(1), AtomId(3)),
                        Edge(AtomId(1), AtomId(4)),
                        Edge(AtomId(1), AtomId(5)),
                        Edge(AtomId(1), AtomId(6)),
                    }
                ),
                "cp1_pi",
            ),
        ),
        (
            SystemId(2),
            mk_bonding_system(
                NonNegative(6),
                frozenset(
                    {
                        Edge(AtomId(7), AtomId(8)),
                        Edge(AtomId(7), AtomId(11)),
                        Edge(AtomId(8), AtomId(9)),
                        Edge(AtomId(9), AtomId(10)),
                        Edge(AtomId(10), AtomId(11)),
                        Edge(AtomId(1), AtomId(7)),
                        Edge(AtomId(1), AtomId(8)),
                        Edge(AtomId(1), AtomId(9)),
                        Edge(AtomId(1), AtomId(10)),
                        Edge(AtomId(1), AtomId(11)),
                    }
                ),
                "cp2_pi",
            ),
        ),
        (
            SystemId(3),
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
        (
            SystemId(4),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(2), AtomId(6)),
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
                        Edge(AtomId(2), AtomId(12)),
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
                        Edge(AtomId(3), AtomId(4)),
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
                        Edge(AtomId(3), AtomId(13)),
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
                        Edge(AtomId(4), AtomId(5)),
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
                        Edge(AtomId(4), AtomId(14)),
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
                        Edge(AtomId(5), AtomId(15)),
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
                        Edge(AtomId(6), AtomId(16)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(13),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(7), AtomId(8)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(14),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(7), AtomId(11)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(15),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(7), AtomId(17)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(16),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(8), AtomId(9)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(17),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(8), AtomId(18)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(18),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(9), AtomId(10)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(19),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(9), AtomId(19)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(20),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(10), AtomId(11)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(21),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(10), AtomId(20)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(22),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(11), AtomId(21)),
                    }
                ),
                None,
            ),
        ),
    ),
)
