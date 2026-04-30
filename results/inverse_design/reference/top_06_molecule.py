from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule
from moladt.examples._literal import atom

rank = 6
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -4.73877650207
predictive_sd = 1.64938300643
target_error = 0.26122349793
bayesian_credible_score_percent = 51.3710794846
score = -0.666094827818
formula = 'HCl2NO2'

molecule = validate_molecule(
    Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.Cl, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.O, -0.711, -0.249, 0.107),
            AtomId(3): atom(3, AtomicSymbol.O, -1.411, -1.462, 0.207),
            AtomId(4): atom(4, AtomicSymbol.N, -0.811, -2.501, 0.407),
            AtomId(5): atom(5, AtomicSymbol.Cl, -0.161, -1.375, 0.507),
            AtomId(6): atom(6, AtomicSymbol.H, -1.486, -1.332, 0.607),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(2), AtomId(3)),
                Edge(AtomId(3), AtomId(4)),
                Edge(AtomId(4), AtomId(5)),
                Edge(AtomId(4), AtomId(6)),
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
