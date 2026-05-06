from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.examples._literal import atom

rank = 8
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -4.39584457402
predictive_sd = 0.85830354021
target_error = 0.604155425977
bayesian_credible_score_percent = 68.3126376335
score = -0.381075405291
formula = 'CH4ClN'

molecule = Molecule(
    atoms={
            AtomId(1): atom(1, AtomicSymbol.N, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.C, -0.625, 1.577, -1.179),
            AtomId(3): atom(3, AtomicSymbol.Cl, -0.570, -0.441, 0.890),
            AtomId(4): atom(4, AtomicSymbol.H, 0.830, 1.402, 0.353),
            AtomId(5): atom(5, AtomicSymbol.H, -1.242, 2.194, -1.797),
            AtomId(6): atom(6, AtomicSymbol.H, -0.094, 1.046, -1.942),
            AtomId(7): atom(7, AtomicSymbol.H, 0.132, 2.333, -1.179),
    },
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
            (
                SystemId(3),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset(
                        {
                            Edge(AtomId(1), AtomId(4)),
                        }
                    ),
                    None,
                ),
            ),
            (
                SystemId(4),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset(
                        {
                            Edge(AtomId(2), AtomId(5)),
                        }
                    ),
                    None,
                ),
            ),
            (
                SystemId(5),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset(
                        {
                            Edge(AtomId(2), AtomId(6)),
                        }
                    ),
                    None,
                ),
            ),
            (
                SystemId(6),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset(
                        {
                            Edge(AtomId(2), AtomId(7)),
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
