from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule
from moladt.examples._literal import atom

rank = 5
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -4.53695900714
predictive_sd = 0.861279907207
target_error = 0.463040992857
bayesian_credible_score_percent = 71.2476869348
score = -0.339007831439
formula = 'CH2F2O'

molecule = validate_molecule(
    Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.O, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.C, 1.419, 0.963, 0.007),
            AtomId(3): atom(3, AtomicSymbol.F, 2.308, 0.074, -0.882),
            AtomId(4): atom(4, AtomicSymbol.F, 1.625, 0.284, 1.248),
            AtomId(5): atom(5, AtomicSymbol.H, -0.971, 0.963, 0.007),
            AtomId(6): atom(6, AtomicSymbol.H, 1.684, 1.962, -0.269),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(5)),
                Edge(AtomId(2), AtomId(3)),
                Edge(AtomId(2), AtomId(4)),
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
