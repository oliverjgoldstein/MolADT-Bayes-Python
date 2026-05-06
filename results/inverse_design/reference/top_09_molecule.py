from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.examples._literal import atom

rank = 9
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -5.1536185208
predictive_sd = 1.07004192021
target_error = 0.153618520796
bayesian_credible_score_percent = 67.9044745143
score = -0.387068255004
formula = 'CH3ClFNO'

molecule = Molecule(
    atoms={
            AtomId(1): atom(1, AtomicSymbol.O, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.F, -1.491, 0.963, 0.007),
            AtomId(3): atom(3, AtomicSymbol.N, 1.389, 0.963, 0.007),
            AtomId(4): atom(4, AtomicSymbol.Cl, 2.351, 0.963, -0.955),
            AtomId(5): atom(5, AtomicSymbol.C, 1.952, 0.963, 1.365),
            AtomId(6): atom(6, AtomicSymbol.H, 1.952, 0.963, 2.435),
            AtomId(7): atom(7, AtomicSymbol.H, 3.001, 0.963, 1.156),
            AtomId(8): atom(8, AtomicSymbol.H, 1.952, 2.033, 1.365),
    },
    local_bonds=frozenset(
        {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(3)),
                Edge(AtomId(3), AtomId(4)),
                Edge(AtomId(3), AtomId(5)),
                Edge(AtomId(5), AtomId(6)),
                Edge(AtomId(5), AtomId(7)),
                Edge(AtomId(5), AtomId(8)),
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
                    'single',
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
                    'single',
                ),
            ),
            (
                SystemId(3),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset(
                        {
                            Edge(AtomId(3), AtomId(4)),
                        }
                    ),
                    'single',
                ),
            ),
            (
                SystemId(4),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset(
                        {
                            Edge(AtomId(3), AtomId(5)),
                        }
                    ),
                    'single',
                ),
            ),
            (
                SystemId(5),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset(
                        {
                            Edge(AtomId(5), AtomId(6)),
                        }
                    ),
                    'single',
                ),
            ),
            (
                SystemId(6),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset(
                        {
                            Edge(AtomId(5), AtomId(7)),
                        }
                    ),
                    'single',
                ),
            ),
            (
                SystemId(7),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset(
                        {
                            Edge(AtomId(5), AtomId(8)),
                        }
                    ),
                    'single',
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
