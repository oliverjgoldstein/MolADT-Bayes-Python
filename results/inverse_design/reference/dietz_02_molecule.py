from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.examples._literal import atom

rank = 2
target_freesolv = -5
seed_molecule = 'freesolv-prior'
random_seed = 0
predicted_freesolv = -5.07787991629
predictive_sd = 0.864839286
target_error = 0.077879916293
bayesian_credible_score_percent = 75.5061596397
score = -0.28095594843
formula = 'C11H16O'

molecule = Molecule(
    atoms={
            AtomId(1): atom(1, AtomicSymbol.C, -1.006, 0.025, -0.434),
            AtomId(2): atom(2, AtomicSymbol.C, -2.090, 0.330, 0.622),
            AtomId(3): atom(3, AtomicSymbol.C, -1.778, 1.683, 1.295),
            AtomId(4): atom(4, AtomicSymbol.C, -3.429, 0.495, -0.134),
            AtomId(5): atom(5, AtomicSymbol.C, -2.108, -0.784, 1.681),
            AtomId(6): atom(6, AtomicSymbol.C, -3.265, -1.535, 1.886),
            AtomId(7): atom(7, AtomicSymbol.C, -3.281, -2.544, 2.849),
            AtomId(8): atom(8, AtomicSymbol.C, -2.139, -2.802, 3.607),
            AtomId(9): atom(9, AtomicSymbol.C, -0.982, -2.051, 3.403),
            AtomId(10): atom(10, AtomicSymbol.C, -0.966, -1.042, 2.440),
            AtomId(11): atom(11, AtomicSymbol.O, -2.155, -3.786, 4.547),
            AtomId(12): atom(12, AtomicSymbol.C, -4.575, 1.413, 0.330),
            AtomId(13): atom(13, AtomicSymbol.H, -0.388, -0.593, -1.052),
            AtomId(14): atom(14, AtomicSymbol.H, -0.683, 1.007, -0.710),
            AtomId(15): atom(15, AtomicSymbol.H, -0.249, 0.025, 0.323),
            AtomId(16): atom(16, AtomicSymbol.H, -1.778, 2.440, 2.052),
            AtomId(17): atom(17, AtomicSymbol.H, -1.206, 2.163, 0.529),
            AtomId(18): atom(18, AtomicSymbol.H, -2.848, 1.683, 1.295),
            AtomId(19): atom(19, AtomicSymbol.H, -3.568, -0.064, -1.036),
            AtomId(20): atom(20, AtomicSymbol.H, -4.047, -0.123, 0.484),
            AtomId(21): atom(21, AtomicSymbol.H, -4.141, -1.337, 1.304),
            AtomId(22): atom(22, AtomicSymbol.H, -4.169, -3.120, 3.006),
            AtomId(23): atom(23, AtomicSymbol.H, -0.106, -2.249, 3.985),
            AtomId(24): atom(24, AtomicSymbol.H, -0.078, -0.466, 2.283),
            AtomId(25): atom(25, AtomicSymbol.H, -2.155, -4.465, 5.226),
            AtomId(26): atom(26, AtomicSymbol.H, -5.332, 2.170, 0.330),
            AtomId(27): atom(27, AtomicSymbol.H, -4.699, 1.045, 1.327),
            AtomId(28): atom(28, AtomicSymbol.H, -3.819, 2.170, 0.330),
    },
    local_bonds=frozenset(
        {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(13)),
                Edge(AtomId(1), AtomId(14)),
                Edge(AtomId(1), AtomId(15)),
                Edge(AtomId(2), AtomId(3)),
                Edge(AtomId(2), AtomId(4)),
                Edge(AtomId(2), AtomId(5)),
                Edge(AtomId(3), AtomId(16)),
                Edge(AtomId(3), AtomId(17)),
                Edge(AtomId(3), AtomId(18)),
                Edge(AtomId(4), AtomId(12)),
                Edge(AtomId(4), AtomId(19)),
                Edge(AtomId(4), AtomId(20)),
                Edge(AtomId(5), AtomId(6)),
                Edge(AtomId(5), AtomId(10)),
                Edge(AtomId(6), AtomId(7)),
                Edge(AtomId(6), AtomId(21)),
                Edge(AtomId(7), AtomId(8)),
                Edge(AtomId(7), AtomId(22)),
                Edge(AtomId(8), AtomId(9)),
                Edge(AtomId(8), AtomId(11)),
                Edge(AtomId(9), AtomId(10)),
                Edge(AtomId(9), AtomId(23)),
                Edge(AtomId(10), AtomId(24)),
                Edge(AtomId(11), AtomId(25)),
                Edge(AtomId(12), AtomId(26)),
                Edge(AtomId(12), AtomId(27)),
                Edge(AtomId(12), AtomId(28)),
        }
    ),
    systems=(
            (
                SystemId(1),
                mk_bonding_system(
                    NonNegative(6),
                    frozenset(
                        {
                            Edge(AtomId(5), AtomId(6)),
                            Edge(AtomId(5), AtomId(10)),
                            Edge(AtomId(6), AtomId(7)),
                            Edge(AtomId(7), AtomId(8)),
                            Edge(AtomId(8), AtomId(9)),
                            Edge(AtomId(9), AtomId(10)),
                        }
                    ),
                    'pi_ring',
                ),
            ),
    ),
)

__all__ = [
    'rank',
    'target_freesolv',
    'seed_molecule',
    'random_seed',
    'predicted_freesolv',
    'predictive_sd',
    'target_error',
    'bayesian_credible_score_percent',
    'score',
    'formula',
    'molecule',
]
