from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule
from moladt.examples._literal import atom

rank = 6
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -5.16189399675
predictive_sd = 0.997720570694
target_error = 0.161893996746
bayesian_credible_score_percent = 70.3279245722
score = -0.352001245954
formula = 'CH3Cl2NO'

molecule = validate_molecule(
    Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.O, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.N, 1.419, 0.963, 0.007),
            AtomId(3): atom(3, AtomicSymbol.Cl, -0.837, 1.789, -0.819),
            AtomId(4): atom(4, AtomicSymbol.Cl, 2.147, 0.117, -0.839),
            AtomId(5): atom(5, AtomicSymbol.C, 2.139, 1.869, 0.913),
            AtomId(6): atom(6, AtomicSymbol.H, 2.757, 2.487, 1.531),
            AtomId(7): atom(7, AtomicSymbol.H, 1.235, 2.273, 1.317),
            AtomId(8): atom(8, AtomicSymbol.H, 2.139, 1.112, 1.670),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(3)),
                Edge(AtomId(2), AtomId(4)),
                Edge(AtomId(2), AtomId(5)),
                Edge(AtomId(5), AtomId(6)),
                Edge(AtomId(5), AtomId(7)),
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
