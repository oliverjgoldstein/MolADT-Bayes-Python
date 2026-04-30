from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule
from moladt.examples._literal import atom

rank = 1
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -5.11643619349
predictive_sd = 1.35411457323
target_error = 0.11643619349
bayesian_credible_score_percent = 59.263839007
score = -0.523170863508
formula = 'CH2Cl2FNO'

molecule = validate_molecule(
    Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.O, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.Cl, -0.711, -0.249, 0.107),
            AtomId(3): atom(3, AtomicSymbol.C, -0.711, -0.249, 0.107),
            AtomId(4): atom(4, AtomicSymbol.F, -0.061, 0.876, 0.207),
            AtomId(5): atom(5, AtomicSymbol.N, -0.061, 0.876, 0.207),
            AtomId(6): atom(6, AtomicSymbol.Cl, -1.461, 0.876, 0.207),
            AtomId(7): atom(7, AtomicSymbol.H, -1.311, -1.289, 0.207),
            AtomId(8): atom(8, AtomicSymbol.H, 0.564, -0.206, 0.407),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(3)),
                Edge(AtomId(3), AtomId(4)),
                Edge(AtomId(3), AtomId(5)),
                Edge(AtomId(3), AtomId(7)),
                Edge(AtomId(5), AtomId(6)),
                Edge(AtomId(5), AtomId(8)),
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
