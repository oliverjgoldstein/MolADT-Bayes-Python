from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule
from moladt.examples._literal import atom

rank = 7
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -4.11708735162
predictive_sd = 1.46287523763
target_error = 0.88291264838
bayesian_credible_score_percent = 49.8455176434
score = -0.696241610505
formula = 'C3H6F2O2'

molecule = validate_molecule(
    Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.O, -0.711, -0.249, 0.107),
            AtomId(2): atom(2, AtomicSymbol.O, -1.411, -1.462, 0.207),
            AtomId(3): atom(3, AtomicSymbol.C, -0.811, -2.501, 0.407),
            AtomId(4): atom(4, AtomicSymbol.C, -0.161, -1.375, 0.507),
            AtomId(5): atom(5, AtomicSymbol.C, -0.161, -1.375, 0.507),
            AtomId(6): atom(6, AtomicSymbol.F, 0.464, -2.458, 0.707),
            AtomId(7): atom(7, AtomicSymbol.F, 0.464, -2.458, 0.707),
            AtomId(8): atom(8, AtomicSymbol.H, -1.386, 0.920, 0.307),
            AtomId(9): atom(9, AtomicSymbol.H, -2.211, -2.501, 0.407),
            AtomId(10): atom(10, AtomicSymbol.H, -0.761, -2.414, 0.607),
            AtomId(11): atom(11, AtomicSymbol.H, 0.464, -2.458, 0.707),
            AtomId(12): atom(12, AtomicSymbol.H, 1.139, -1.375, 0.507),
            AtomId(13): atom(13, AtomicSymbol.H, 0.514, -0.206, 0.607),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(8)),
                Edge(AtomId(2), AtomId(3)),
                Edge(AtomId(3), AtomId(4)),
                Edge(AtomId(3), AtomId(5)),
                Edge(AtomId(3), AtomId(9)),
                Edge(AtomId(4), AtomId(10)),
                Edge(AtomId(4), AtomId(11)),
                Edge(AtomId(4), AtomId(12)),
                Edge(AtomId(5), AtomId(6)),
                Edge(AtomId(5), AtomId(7)),
                Edge(AtomId(5), AtomId(13)),
            }
        ),
        systems=(
        ),
    )
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
