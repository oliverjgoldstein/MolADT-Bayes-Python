from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule
from moladt.examples._literal import atom

rank = 8
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -3.84226286921
predictive_sd = 1.27832229212
target_error = 1.15773713079
bayesian_credible_score_percent = 47.773682142
score = -0.738695280921
formula = 'CH3ClO'

molecule = validate_molecule(
    Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.O, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.Cl, -0.711, -0.249, 0.107),
            AtomId(3): atom(3, AtomicSymbol.C, -0.711, -0.249, 0.107),
            AtomId(4): atom(4, AtomicSymbol.H, -0.111, -1.289, 0.307),
            AtomId(5): atom(5, AtomicSymbol.H, 0.539, -0.249, 0.107),
            AtomId(6): atom(6, AtomicSymbol.H, -0.061, 0.876, 0.207),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(3)),
                Edge(AtomId(3), AtomId(4)),
                Edge(AtomId(3), AtomId(5)),
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
