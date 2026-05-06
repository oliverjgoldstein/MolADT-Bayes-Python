from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.examples._literal import atom

rank = 10
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -4.46304698579
predictive_sd = 0.957140156383
target_error = 0.536953014207
bayesian_credible_score_percent = 67.0061659855
score = -0.400385541197
formula = 'ClFO'

molecule = Molecule(
    atoms={
            AtomId(1): atom(1, AtomicSymbol.O, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.F, -1.491, 0.963, 0.007),
            AtomId(3): atom(3, AtomicSymbol.Cl, 1.419, 0.963, 0.007),
    },
    local_bonds=frozenset(
        {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(3)),
        }
    ),
    systems=(
            (
                SystemId(1),
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
                SystemId(2),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset(
                        {
                            Edge(AtomId(1), AtomId(3)),
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
