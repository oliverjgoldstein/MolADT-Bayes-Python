from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.examples._literal import atom

rank = 8
target_freesolv = -5
seed_molecule = 'freesolv-prior'
random_seed = 0
predicted_freesolv = -4.87146947488
predictive_sd = 0.96483347508
target_error = 0.128530525122
bayesian_credible_score_percent = 71.6575356913
score = -0.333271863604
formula = 'C9H13N'

molecule = Molecule(
    atoms={
            AtomId(1): atom(1, AtomicSymbol.N, 1.932, -1.013, -1.603),
            AtomId(2): atom(2, AtomicSymbol.C, 1.885, -1.036, -0.112),
            AtomId(3): atom(3, AtomicSymbol.C, 2.921, -1.631, 0.608),
            AtomId(4): atom(4, AtomicSymbol.C, 2.878, -1.652, 2.002),
            AtomId(5): atom(5, AtomicSymbol.C, 1.800, -1.079, 2.676),
            AtomId(6): atom(6, AtomicSymbol.C, 0.764, -0.484, 1.955),
            AtomId(7): atom(7, AtomicSymbol.C, 0.807, -0.463, 0.561),
            AtomId(8): atom(8, AtomicSymbol.C, -0.297, 0.081, 2.597),
            AtomId(9): atom(9, AtomicSymbol.C, 1.770, -1.111, 4.168),
            AtomId(10): atom(10, AtomicSymbol.C, -0.337, 0.193, -0.235),
            AtomId(11): atom(11, AtomicSymbol.H, 1.932, -1.013, -2.613),
            AtomId(12): atom(12, AtomicSymbol.H, 2.839, -0.569, -1.585),
            AtomId(13): atom(13, AtomicSymbol.H, 3.748, -2.071, 0.091),
            AtomId(14): atom(14, AtomicSymbol.H, 3.673, -2.108, 2.555),
            AtomId(15): atom(15, AtomicSymbol.H, -1.054, 0.081, 3.354),
            AtomId(16): atom(16, AtomicSymbol.H, -0.456, 1.001, 2.074),
            AtomId(17): atom(17, AtomicSymbol.H, 0.321, 0.699, 3.215),
            AtomId(18): atom(18, AtomicSymbol.H, 1.770, -1.111, 5.238),
            AtomId(19): atom(19, AtomicSymbol.H, 1.038, -1.892, 4.152),
            AtomId(20): atom(20, AtomicSymbol.H, 1.013, -0.354, 4.168),
            AtomId(21): atom(21, AtomicSymbol.H, -0.954, 0.811, -0.853),
            AtomId(22): atom(22, AtomicSymbol.H, -1.098, -0.504, 0.045),
            AtomId(23): atom(23, AtomicSymbol.H, -0.337, 0.950, 0.522),
    },
    local_bonds=frozenset(
        {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(11)),
                Edge(AtomId(1), AtomId(12)),
                Edge(AtomId(2), AtomId(3)),
                Edge(AtomId(2), AtomId(7)),
                Edge(AtomId(3), AtomId(4)),
                Edge(AtomId(3), AtomId(13)),
                Edge(AtomId(4), AtomId(5)),
                Edge(AtomId(4), AtomId(14)),
                Edge(AtomId(5), AtomId(6)),
                Edge(AtomId(5), AtomId(9)),
                Edge(AtomId(6), AtomId(7)),
                Edge(AtomId(6), AtomId(8)),
                Edge(AtomId(7), AtomId(10)),
                Edge(AtomId(8), AtomId(15)),
                Edge(AtomId(8), AtomId(16)),
                Edge(AtomId(8), AtomId(17)),
                Edge(AtomId(9), AtomId(18)),
                Edge(AtomId(9), AtomId(19)),
                Edge(AtomId(9), AtomId(20)),
                Edge(AtomId(10), AtomId(21)),
                Edge(AtomId(10), AtomId(22)),
                Edge(AtomId(10), AtomId(23)),
        }
    ),
    systems=(
            (
                SystemId(1),
                mk_bonding_system(
                    NonNegative(6),
                    frozenset(
                        {
                            Edge(AtomId(2), AtomId(3)),
                            Edge(AtomId(2), AtomId(7)),
                            Edge(AtomId(3), AtomId(4)),
                            Edge(AtomId(4), AtomId(5)),
                            Edge(AtomId(5), AtomId(6)),
                            Edge(AtomId(6), AtomId(7)),
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
