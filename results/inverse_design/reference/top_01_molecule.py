from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule
from moladt.examples._literal import atom

rank = 1
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -4.52719272006
predictive_sd = 0.794246843091
target_error = 0.472807279945
bayesian_credible_score_percent = 73.1190140829
score = -0.313081742482
formula = 'CH5N'

molecule = validate_molecule(
    Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.N, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.C, 0.557, 0.395, -1.091),
            AtomId(3): atom(3, AtomicSymbol.H, -0.011, 1.677, 0.721),
            AtomId(4): atom(4, AtomicSymbol.H, -0.825, 0.398, 0.202),
            AtomId(5): atom(5, AtomicSymbol.H, 1.175, -0.223, -1.708),
            AtomId(6): atom(6, AtomicSymbol.H, 0.026, 0.926, -1.853),
            AtomId(7): atom(7, AtomicSymbol.H, 1.313, 1.152, -1.091),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(3)),
                Edge(AtomId(1), AtomId(4)),
                Edge(AtomId(2), AtomId(5)),
                Edge(AtomId(2), AtomId(6)),
                Edge(AtomId(2), AtomId(7)),
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
