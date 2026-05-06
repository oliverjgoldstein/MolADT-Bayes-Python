from __future__ import annotations

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import AtomicSymbol, Molecule
from ._literal import atom, single_covalent_systems


ferrocene_pretty = Molecule(
    atoms={
        AtomId(1): atom(1, AtomicSymbol.Fe, 0.000, 0.000, 0.000),
        AtomId(2): atom(2, AtomicSymbol.C, 1.194, 0.000, 1.660),
        AtomId(3): atom(3, AtomicSymbol.C, 0.368966, 1.135561, 1.660),
        AtomId(4): atom(4, AtomicSymbol.C, -0.965966, 0.701816, 1.660),
        AtomId(5): atom(5, AtomicSymbol.C, -0.965966, -0.701816, 1.660),
        AtomId(6): atom(6, AtomicSymbol.C, 0.368966, -1.135561, 1.660),
        AtomId(7): atom(7, AtomicSymbol.C, 0.965966, 0.701816, -1.660),
        AtomId(8): atom(8, AtomicSymbol.C, -0.368966, 1.135561, -1.660),
        AtomId(9): atom(9, AtomicSymbol.C, -1.194, 0.000, -1.660),
        AtomId(10): atom(10, AtomicSymbol.C, -0.368966, -1.135561, -1.660),
        AtomId(11): atom(11, AtomicSymbol.C, 0.965966, -0.701816, -1.660),
        AtomId(12): atom(12, AtomicSymbol.H, 2.280, 0.000, 1.565),
        AtomId(13): atom(13, AtomicSymbol.H, 0.704559, 2.168409, 1.565),
        AtomId(14): atom(14, AtomicSymbol.H, -1.844559, 1.340150, 1.565),
        AtomId(15): atom(15, AtomicSymbol.H, -1.844559, -1.340150, 1.565),
        AtomId(16): atom(16, AtomicSymbol.H, 0.704559, -2.168409, 1.565),
        AtomId(17): atom(17, AtomicSymbol.H, 1.844559, 1.340150, -1.565),
        AtomId(18): atom(18, AtomicSymbol.H, -0.704559, 2.168409, -1.565),
        AtomId(19): atom(19, AtomicSymbol.H, -2.280, 0.000, -1.565),
        AtomId(20): atom(20, AtomicSymbol.H, -0.704559, -2.168409, -1.565),
        AtomId(21): atom(21, AtomicSymbol.H, 1.844559, -1.340150, -1.565),
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
                    }
                ),
                "cp2_pi",
            ),
        ),
        (
            SystemId(3),
            mk_bonding_system(
                NonNegative(12),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(2)),
                        Edge(AtomId(1), AtomId(3)),
                        Edge(AtomId(1), AtomId(4)),
                        Edge(AtomId(1), AtomId(5)),
                        Edge(AtomId(1), AtomId(6)),
                        Edge(AtomId(1), AtomId(7)),
                        Edge(AtomId(1), AtomId(8)),
                        Edge(AtomId(1), AtomId(9)),
                        Edge(AtomId(1), AtomId(10)),
                        Edge(AtomId(1), AtomId(11)),
                    }
                ),
                "fe_cp_coordination",
            ),
        ),
    )
    + single_covalent_systems(
        4,
        (
            Edge(AtomId(2), AtomId(3)),
            Edge(AtomId(2), AtomId(6)),
            Edge(AtomId(2), AtomId(12)),
            Edge(AtomId(3), AtomId(4)),
            Edge(AtomId(3), AtomId(13)),
            Edge(AtomId(4), AtomId(5)),
            Edge(AtomId(4), AtomId(14)),
            Edge(AtomId(5), AtomId(6)),
            Edge(AtomId(5), AtomId(15)),
            Edge(AtomId(6), AtomId(16)),
            Edge(AtomId(7), AtomId(8)),
            Edge(AtomId(7), AtomId(11)),
            Edge(AtomId(7), AtomId(17)),
            Edge(AtomId(8), AtomId(9)),
            Edge(AtomId(8), AtomId(18)),
            Edge(AtomId(9), AtomId(10)),
            Edge(AtomId(9), AtomId(19)),
            Edge(AtomId(10), AtomId(11)),
            Edge(AtomId(10), AtomId(20)),
            Edge(AtomId(11), AtomId(21)),
        ),
    ),
)
