from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule
from moladt.examples._literal import atom

rank = 9
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -5.79533497234
predictive_sd = 1.77258781748
target_error = 0.79533497234
bayesian_credible_score_percent = 45.5228738752
score = -0.786955263822
formula = 'C2H4ClFO2'

molecule = validate_molecule(
    Molecule(
        atoms={
            AtomId(1): atom(1, AtomicSymbol.O, -0.711, -0.249, 0.107),
            AtomId(2): atom(2, AtomicSymbol.O, -1.411, -1.462, 0.207),
            AtomId(3): atom(3, AtomicSymbol.C, -0.811, -2.501, 0.407),
            AtomId(4): atom(4, AtomicSymbol.Cl, -0.161, -1.375, 0.507),
            AtomId(5): atom(5, AtomicSymbol.C, -0.161, -1.375, 0.507),
            AtomId(6): atom(6, AtomicSymbol.F, 0.464, -2.458, 0.707),
            AtomId(7): atom(7, AtomicSymbol.H, 0.589, -0.249, 0.107),
            AtomId(8): atom(8, AtomicSymbol.H, -0.136, -1.332, 0.507),
            AtomId(9): atom(9, AtomicSymbol.H, -0.861, -0.163, 0.707),
            AtomId(10): atom(10, AtomicSymbol.H, -1.361, -1.375, 0.507),
        },
        local_bonds=frozenset(
            {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(7)),
                Edge(AtomId(2), AtomId(3)),
                Edge(AtomId(3), AtomId(4)),
                Edge(AtomId(3), AtomId(5)),
                Edge(AtomId(3), AtomId(8)),
                Edge(AtomId(5), AtomId(6)),
                Edge(AtomId(5), AtomId(9)),
                Edge(AtomId(5), AtomId(10)),
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
