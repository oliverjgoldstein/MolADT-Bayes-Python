from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule
from moladt.examples._literal import atom

rank = 2
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -5.08596746952
predictive_sd = 0.964645867596
target_error = 0.0859674695248
bayesian_credible_score_percent = 71.8338493726
score = -0.330814381297
formula = 'C2H6FNO'

molecule = validate_molecule(
    Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.O, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.N, 1.419, 0.963, 0.007),
            AtomId(3): atom(3, AtomicSymbol.C, 2.308, 0.074, -0.882),
            AtomId(4): atom(4, AtomicSymbol.F, 2.095, 1.886, 0.930),
            AtomId(5): atom(5, AtomicSymbol.C, -0.837, 1.789, -0.819),
            AtomId(6): atom(6, AtomicSymbol.H, 2.926, -0.544, -1.500),
            AtomId(7): atom(7, AtomicSymbol.H, 3.065, 0.830, -0.882),
            AtomId(8): atom(8, AtomicSymbol.H, 2.308, -0.683, -0.126),
            AtomId(9): atom(9, AtomicSymbol.H, -1.454, 2.406, -1.436),
            AtomId(10): atom(10, AtomicSymbol.H, -0.080, 2.545, -0.819),
            AtomId(11): atom(11, AtomicSymbol.H, -1.593, 1.789, -0.062),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(5)),
                Edge(AtomId(2), AtomId(3)),
                Edge(AtomId(2), AtomId(4)),
                Edge(AtomId(3), AtomId(6)),
                Edge(AtomId(3), AtomId(7)),
                Edge(AtomId(3), AtomId(8)),
                Edge(AtomId(5), AtomId(9)),
                Edge(AtomId(5), AtomId(10)),
                Edge(AtomId(5), AtomId(11)),
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
