from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule
from moladt.examples._literal import atom

rank = 3
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -4.55218157431
predictive_sd = 0.861924664469
target_error = 0.44781842569
bayesian_credible_score_percent = 71.511656589
score = -0.335309720359
formula = 'CH2Cl2O'

molecule = validate_molecule(
    Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.C, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.O, -1.491, 0.963, 0.007),
            AtomId(3): atom(3, AtomicSymbol.Cl, 1.011, -0.059, -1.015),
            AtomId(4): atom(4, AtomicSymbol.Cl, 0.540, 2.637, -0.162),
            AtomId(5): atom(5, AtomicSymbol.H, 0.143, 0.455, 0.936),
            AtomId(6): atom(6, AtomicSymbol.H, -2.451, 0.963, 0.007),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(3)),
                Edge(AtomId(1), AtomId(4)),
                Edge(AtomId(1), AtomId(5)),
                Edge(AtomId(2), AtomId(6)),
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
