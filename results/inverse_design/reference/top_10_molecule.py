from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.examples._literal import atom

rank = 10
target_freesolv = -5
seed_molecule = 'freesolv-prior'
random_seed = 0
predicted_freesolv = -5.36119212982
predictive_sd = 0.96712892617
target_error = 0.361192129821
bayesian_credible_score_percent = 69.4998407837
score = -0.363845724303
formula = 'C4H8FNO'

molecule = Molecule(
    atoms={
            AtomId(1): atom(1, AtomicSymbol.C, -1.856, -1.850, -3.220),
            AtomId(2): atom(2, AtomicSymbol.O, -1.259, -1.696, -1.929),
            AtomId(3): atom(3, AtomicSymbol.C, -1.638, -0.584, -1.237),
            AtomId(4): atom(4, AtomicSymbol.N, -0.937, -0.555, 0.088),
            AtomId(5): atom(5, AtomicSymbol.C, -1.541, -1.309, 1.222),
            AtomId(6): atom(6, AtomicSymbol.C, -1.575, 0.189, 1.209),
            AtomId(7): atom(7, AtomicSymbol.F, -1.773, 0.519, -2.003),
            AtomId(8): atom(8, AtomicSymbol.H, -1.856, -1.850, -4.290),
            AtomId(9): atom(9, AtomicSymbol.H, -2.866, -2.111, -2.984),
            AtomId(10): atom(10, AtomicSymbol.H, -1.856, -0.780, -3.220),
            AtomId(11): atom(11, AtomicSymbol.H, -2.663, -0.618, -0.933),
            AtomId(12): atom(12, AtomicSymbol.H, -1.777, -2.238, 1.697),
            AtomId(13): atom(13, AtomicSymbol.H, -0.784, -1.309, 1.979),
            AtomId(14): atom(14, AtomicSymbol.H, -1.853, 1.114, 1.668),
            AtomId(15): atom(15, AtomicSymbol.H, -0.818, 0.189, 1.966),
    },
    local_bonds=frozenset(
        {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(8)),
                Edge(AtomId(1), AtomId(9)),
                Edge(AtomId(1), AtomId(10)),
                Edge(AtomId(2), AtomId(3)),
                Edge(AtomId(3), AtomId(4)),
                Edge(AtomId(3), AtomId(7)),
                Edge(AtomId(3), AtomId(11)),
                Edge(AtomId(4), AtomId(5)),
                Edge(AtomId(4), AtomId(6)),
                Edge(AtomId(5), AtomId(6)),
                Edge(AtomId(5), AtomId(12)),
                Edge(AtomId(5), AtomId(13)),
                Edge(AtomId(6), AtomId(14)),
                Edge(AtomId(6), AtomId(15)),
        }
    ),
    systems=(
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
