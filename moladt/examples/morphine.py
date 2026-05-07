from __future__ import annotations

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import (
    AtomicSymbol,
    Molecule,
    SmilesAtomStereo,
    SmilesAtomStereoClass,
    SmilesBondStereo,
    SmilesBondStereoDirection,
    SmilesStereochemistry,
)
from ._literal import atom


MORPHINE_RING_CLOSURE_SMILES = 'CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5'

morphine_pretty = Molecule(
    atoms={
        AtomId(1): atom(1, AtomicSymbol.O, 0.000, 0.000, 0.100),
        AtomId(2): atom(2, AtomicSymbol.C, 1.000, 0.800, 0.450),
        AtomId(3): atom(3, AtomicSymbol.C, 2.000, 0.800, -0.100),
        AtomId(4): atom(4, AtomicSymbol.O, 2.000, -0.400, -0.550),
        AtomId(5): atom(5, AtomicSymbol.C, 3.000, 0.800, 0.350),
        AtomId(6): atom(6, AtomicSymbol.C, 4.000, 0.800, 0.750),
        AtomId(7): atom(7, AtomicSymbol.C, 5.000, 0.800, 0.200),
        AtomId(8): atom(8, AtomicSymbol.C, 1.800, 2.000, 0.800),
        AtomId(9): atom(9, AtomicSymbol.C, 2.800, 2.800, 1.100),
        AtomId(10): atom(10, AtomicSymbol.C, 3.800, 2.000, 0.600),
        AtomId(11): atom(11, AtomicSymbol.C, 0.800, 2.000, 0.150),
        AtomId(12): atom(12, AtomicSymbol.C, 1.200, 3.200, 0.550),
        AtomId(13): atom(13, AtomicSymbol.O, 0.400, 4.000, 0.300),
        AtomId(14): atom(14, AtomicSymbol.C, 2.400, 3.800, 0.950),
        AtomId(15): atom(15, AtomicSymbol.C, 3.600, 3.800, 0.700),
        AtomId(16): atom(16, AtomicSymbol.C, 4.200, 2.800, 0.200),
        AtomId(17): atom(17, AtomicSymbol.C, 5.400, 2.800, -0.200),
        AtomId(18): atom(18, AtomicSymbol.C, 6.200, 1.800, -0.550),
        AtomId(19): atom(19, AtomicSymbol.N, 7.200, 1.800, -0.850),
        AtomId(20): atom(20, AtomicSymbol.C, 8.200, 2.400, -1.100),
        AtomId(21): atom(21, AtomicSymbol.C, 6.000, 2.800, -0.350),
    },
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(6),
                frozenset(
                    {
                        Edge(AtomId(10), AtomId(11)),
                        Edge(AtomId(10), AtomId(16)),
                        Edge(AtomId(11), AtomId(12)),
                        Edge(AtomId(12), AtomId(14)),
                        Edge(AtomId(14), AtomId(15)),
                        Edge(AtomId(15), AtomId(16)),
                    }
                ),
                "phenyl_pi_ring",
            ),
        ),
        (
            SystemId(2),
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
            SystemId(3),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(11)),
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
                        Edge(AtomId(2), AtomId(3)),
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
                        Edge(AtomId(3), AtomId(5)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(8),
            mk_bonding_system(
                NonNegative(4),
                frozenset(
                    {
                        Edge(AtomId(5), AtomId(6)),
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
                        Edge(AtomId(6), AtomId(7)),
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
                        Edge(AtomId(7), AtomId(8)),
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
                        Edge(AtomId(7), AtomId(18)),
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
                        Edge(AtomId(8), AtomId(9)),
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
                        Edge(AtomId(8), AtomId(10)),
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
                        Edge(AtomId(9), AtomId(21)),
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
                        Edge(AtomId(10), AtomId(11)),
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
                        Edge(AtomId(10), AtomId(16)),
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
                        Edge(AtomId(11), AtomId(12)),
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
                        Edge(AtomId(12), AtomId(13)),
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
                        Edge(AtomId(12), AtomId(14)),
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
                        Edge(AtomId(14), AtomId(15)),
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
                        Edge(AtomId(15), AtomId(16)),
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
                        Edge(AtomId(16), AtomId(17)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(23),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(17), AtomId(18)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(24),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(18), AtomId(19)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(25),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(19), AtomId(20)),
                    }
                ),
                None,
            ),
        ),
        (
            SystemId(26),
            mk_bonding_system(
                NonNegative(2),
                frozenset(
                    {
                        Edge(AtomId(19), AtomId(21)),
                    }
                ),
                None,
            ),
        ),
    ),
    smiles_stereochemistry=SmilesStereochemistry(
        atom_stereo=(
            SmilesAtomStereo(AtomId(2), SmilesAtomStereoClass.TETRAHEDRAL, 1, '@'),
            SmilesAtomStereo(AtomId(3), SmilesAtomStereoClass.TETRAHEDRAL, 2, '@@'),
            SmilesAtomStereo(AtomId(7), SmilesAtomStereoClass.TETRAHEDRAL, 1, '@'),
            SmilesAtomStereo(AtomId(8), SmilesAtomStereoClass.TETRAHEDRAL, 1, '@'),
            SmilesAtomStereo(AtomId(18), SmilesAtomStereoClass.TETRAHEDRAL, 1, '@'),
        ),
        bond_stereo=(
        ),
    ),
)
