from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.examples._literal import atom

rank = 4
target_freesolv = -5
seed_molecule = 'freesolv-prior'
random_seed = 0
predicted_freesolv = -4.8635959534
predictive_sd = 0.867919356927
target_error = 0.136404046598
bayesian_credible_score_percent = 75.1224028843
score = -0.286051364322
formula = 'C9H16O'

molecule = Molecule(
    atoms={
            AtomId(1): atom(1, AtomicSymbol.C, -6.149, -0.668, 5.183),
            AtomId(2): atom(2, AtomicSymbol.C, -5.069, -1.055, 4.207),
            AtomId(3): atom(3, AtomicSymbol.C, -4.901, -0.357, 3.068),
            AtomId(4): atom(4, AtomicSymbol.C, -3.895, -0.591, 1.976),
            AtomId(5): atom(5, AtomicSymbol.C, -2.967, 0.612, 1.761),
            AtomId(6): atom(6, AtomicSymbol.C, -1.942, 0.404, 0.665),
            AtomId(7): atom(7, AtomicSymbol.C, -1.834, 1.163, -0.442),
            AtomId(8): atom(8, AtomicSymbol.C, -0.854, 1.022, -1.563),
            AtomId(9): atom(9, AtomicSymbol.O, -0.093, 2.219, -1.653),
            AtomId(10): atom(10, AtomicSymbol.C, -1.022, -0.773, 0.865),
            AtomId(11): atom(11, AtomicSymbol.H, -6.906, -0.668, 5.940),
            AtomId(12): atom(12, AtomicSymbol.H, -6.189, 0.374, 4.943),
            AtomId(13): atom(13, AtomicSymbol.H, -5.392, -0.668, 5.940),
            AtomId(14): atom(14, AtomicSymbol.H, -4.432, -1.888, 4.420),
            AtomId(15): atom(15, AtomicSymbol.H, -5.567, 0.468, 2.922),
            AtomId(16): atom(16, AtomicSymbol.H, -3.833, -1.497, 1.410),
            AtomId(17): atom(17, AtomicSymbol.H, -4.513, 0.027, 1.358),
            AtomId(18): atom(18, AtomicSymbol.H, -3.037, 1.514, 2.332),
            AtomId(19): atom(19, AtomicSymbol.H, -3.585, 1.230, 1.143),
            AtomId(20): atom(20, AtomicSymbol.H, -2.536, 1.966, -0.531),
            AtomId(21): atom(21, AtomicSymbol.H, -0.744, 0.163, -2.192),
            AtomId(22): atom(22, AtomicSymbol.H, -1.472, 1.640, -2.181),
            AtomId(23): atom(23, AtomicSymbol.H, 0.586, 2.898, -1.653),
            AtomId(24): atom(24, AtomicSymbol.H, -0.265, -1.530, 0.865),
            AtomId(25): atom(25, AtomicSymbol.H, -1.597, -1.212, 1.654),
            AtomId(26): atom(26, AtomicSymbol.H, -0.404, -0.155, 1.483),
    },
    local_bonds=frozenset(
        {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(11)),
                Edge(AtomId(1), AtomId(12)),
                Edge(AtomId(1), AtomId(13)),
                Edge(AtomId(2), AtomId(3)),
                Edge(AtomId(2), AtomId(14)),
                Edge(AtomId(3), AtomId(4)),
                Edge(AtomId(3), AtomId(15)),
                Edge(AtomId(4), AtomId(5)),
                Edge(AtomId(4), AtomId(16)),
                Edge(AtomId(4), AtomId(17)),
                Edge(AtomId(5), AtomId(6)),
                Edge(AtomId(5), AtomId(18)),
                Edge(AtomId(5), AtomId(19)),
                Edge(AtomId(6), AtomId(7)),
                Edge(AtomId(6), AtomId(10)),
                Edge(AtomId(7), AtomId(8)),
                Edge(AtomId(7), AtomId(20)),
                Edge(AtomId(8), AtomId(9)),
                Edge(AtomId(8), AtomId(21)),
                Edge(AtomId(8), AtomId(22)),
                Edge(AtomId(9), AtomId(23)),
                Edge(AtomId(10), AtomId(24)),
                Edge(AtomId(10), AtomId(25)),
                Edge(AtomId(10), AtomId(26)),
        }
    ),
    systems=(
            (
                SystemId(1),
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
                SystemId(2),
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
