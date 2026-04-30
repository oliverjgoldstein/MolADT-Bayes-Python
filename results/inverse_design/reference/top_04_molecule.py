from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule
from moladt.examples._literal import atom

rank = 4
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -4.00660834221
predictive_sd = 1.07144998084
target_error = 0.99339165779
bayesian_credible_score_percent = 54.2277424461
score = -0.611977555246
formula = 'HClO'

molecule = validate_molecule(
    Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.Cl, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.O, -0.711, -0.249, 0.107),
            AtomId(3): atom(3, AtomicSymbol.H, -2.061, -0.249, 0.107),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(2), AtomId(3)),
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
