from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.examples._literal import atom

rank = 2
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -4.59261091704
predictive_sd = 0.740672529435
target_error = 0.407389082956
bayesian_credible_score_percent = 76.1656042961
score = -0.272260212392
formula = 'C2H5ClO'

molecule = Molecule(
    atoms={
            AtomId(1): atom(1, AtomicSymbol.C, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.C, -0.625, 1.577, -1.179),
            AtomId(3): atom(3, AtomicSymbol.Cl, -0.570, -0.441, 0.890),
            AtomId(4): atom(4, AtomicSymbol.O, 1.155, 1.572, 0.486),
            AtomId(5): atom(5, AtomicSymbol.H, 0.607, 0.345, -0.611),
            AtomId(6): atom(6, AtomicSymbol.H, -1.242, 2.194, -1.797),
            AtomId(7): atom(7, AtomicSymbol.H, -0.094, 1.046, -1.942),
            AtomId(8): atom(8, AtomicSymbol.H, 0.132, 2.333, -1.179),
            AtomId(9): atom(9, AtomicSymbol.H, 1.709, 2.126, 1.040),
    },
    local_bonds=frozenset(
        {
                Edge(AtomId(1), AtomId(2)),
                Edge(AtomId(1), AtomId(3)),
                Edge(AtomId(1), AtomId(4)),
                Edge(AtomId(1), AtomId(5)),
                Edge(AtomId(2), AtomId(6)),
                Edge(AtomId(2), AtomId(7)),
                Edge(AtomId(2), AtomId(8)),
                Edge(AtomId(4), AtomId(9)),
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
                            Edge(AtomId(1), AtomId(4)),
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
                            Edge(AtomId(1), AtomId(5)),
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
                            Edge(AtomId(2), AtomId(6)),
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
                            Edge(AtomId(2), AtomId(7)),
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
                            Edge(AtomId(2), AtomId(8)),
                        }
                    ),
                    'single',
                ),
            ),
            (
                SystemId(8),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset(
                        {
                            Edge(AtomId(4), AtomId(9)),
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
