from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule
from moladt.examples._literal import atom

rank = 9
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -4.67781970415
predictive_sd = 0.991194978955
target_error = 0.32218029585
bayesian_credible_score_percent = 69.1874520002
score = -0.368350669284
formula = 'H2FN'

molecule = validate_molecule(
    Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.N, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.F, 0.557, 0.395, -1.091),
            AtomId(3): atom(3, AtomicSymbol.H, -0.011, 1.677, 0.721),
            AtomId(4): atom(4, AtomicSymbol.H, -0.825, 0.398, 0.202),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(3)),
                Edge(AtomId(1), AtomId(4)),
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
