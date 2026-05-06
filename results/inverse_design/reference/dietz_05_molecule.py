from __future__ import annotations

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.examples._literal import atom

rank = 5
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -4.89587967076
predictive_sd = 0.956240616296
target_error = 0.104120329245
bayesian_credible_score_percent = 72.0699819966
score = -0.327532566856
formula = 'C2H4Cl2O2'

molecule = Molecule(
    atoms={
            AtomId(1): atom(1, AtomicSymbol.C, -0.011, 0.963, 0.007),
            AtomId(2): atom(2, AtomicSymbol.O, -0.625, 1.577, -1.179),
            AtomId(3): atom(3, AtomicSymbol.Cl, -0.570, -0.441, 0.890),
            AtomId(4): atom(4, AtomicSymbol.O, 1.155, 1.572, 0.486),
            AtomId(5): atom(5, AtomicSymbol.Cl, 1.011, -0.059, -1.015),
            AtomId(6): atom(6, AtomicSymbol.C, -2.150, 1.771, -1.268),
            AtomId(7): atom(7, AtomicSymbol.H, 1.709, 2.126, 1.040),
            AtomId(8): atom(8, AtomicSymbol.H, -3.220, 1.771, -1.268),
            AtomId(9): atom(9, AtomicSymbol.H, -2.075, 2.742, -1.712),
            AtomId(10): atom(10, AtomicSymbol.H, -2.150, 1.771, -0.198),
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
                            Edge(AtomId(1), AtomId(5)),
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
                            Edge(AtomId(4), AtomId(7)),
                        }
                    ),
                    None,
                ),
            ),
            (
                SystemId(7),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset(
                        {
                            Edge(AtomId(6), AtomId(8)),
                        }
                    ),
                    None,
                ),
            ),
            (
                SystemId(8),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset(
                        {
                            Edge(AtomId(6), AtomId(9)),
                        }
                    ),
                    None,
                ),
            ),
            (
                SystemId(9),
                mk_bonding_system(
                    NonNegative(2),
                    frozenset(
                        {
                            Edge(AtomId(6), AtomId(10)),
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
