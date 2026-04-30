from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule
from moladt.examples._literal import atom

rank = 8
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -4.50121898382
predictive_sd = 0.890627743953
target_error = 0.498781016182
bayesian_credible_score_percent = 69.6718763964
score = -0.361373444677
formula = 'CH3FO'

molecule = validate_molecule(
    Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.C, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.F, -1.491, 0.963, 0.007),
            AtomId(3): atom(3, AtomicSymbol.O, 1.679, 0.963, 0.007),
            AtomId(4): atom(4, AtomicSymbol.H, -0.011, 2.033, 0.007),
            AtomId(5): atom(5, AtomicSymbol.H, -0.011, 0.963, 1.077),
            AtomId(6): atom(6, AtomicSymbol.H, 2.639, 0.963, 0.007),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(3)),
                Edge(AtomId(1), AtomId(4)),
                Edge(AtomId(1), AtomId(5)),
                Edge(AtomId(3), AtomId(6)),
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
