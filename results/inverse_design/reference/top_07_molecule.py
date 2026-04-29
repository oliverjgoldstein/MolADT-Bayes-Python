from __future__ import annotations

from moladt.chem.constants import element_attributes, element_shells
from moladt.chem.coordinate import Coordinate, mk_angstrom
from moladt.chem.dietz import AtomId, NonNegative, SystemId, mk_bonding_system, mk_edge
from moladt.chem.molecule import Atom, AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule

rank = 7
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -4.11708735162
predictive_sd = 1.46287523763
target_error = 0.882912648384
bayesian_credible_score_percent = 49.8455176434
score = -0.696241610505
formula = 'C3H6F2O2'

atoms = {
    AtomId(1): Atom(
        atom_id=AtomId(1),
        attributes=element_attributes(AtomicSymbol.O),
        coordinate=Coordinate(
            mk_angstrom(-0.711),
            mk_angstrom(-0.249435565298),
            mk_angstrom(0.107),
        ),
        shells=element_shells(AtomicSymbol.O),
        formal_charge=0,
    ),
    AtomId(2): Atom(
        atom_id=AtomId(2),
        attributes=element_attributes(AtomicSymbol.O),
        coordinate=Coordinate(
            mk_angstrom(-1.411),
            mk_angstrom(-1.4618711306),
            mk_angstrom(0.207),
        ),
        shells=element_shells(AtomicSymbol.O),
        formal_charge=0,
    ),
    AtomId(3): Atom(
        atom_id=AtomId(3),
        attributes=element_attributes(AtomicSymbol.C),
        coordinate=Coordinate(
            mk_angstrom(-0.811),
            mk_angstrom(-2.50110161514),
            mk_angstrom(0.407),
        ),
        shells=element_shells(AtomicSymbol.C),
        formal_charge=0,
    ),
    AtomId(4): Atom(
        atom_id=AtomId(4),
        attributes=element_attributes(AtomicSymbol.C),
        coordinate=Coordinate(
            mk_angstrom(-0.161),
            mk_angstrom(-1.37526859022),
            mk_angstrom(0.507),
        ),
        shells=element_shells(AtomicSymbol.C),
        formal_charge=0,
    ),
    AtomId(5): Atom(
        atom_id=AtomId(5),
        attributes=element_attributes(AtomicSymbol.C),
        coordinate=Coordinate(
            mk_angstrom(-0.161),
            mk_angstrom(-1.37526859022),
            mk_angstrom(0.507),
        ),
        shells=element_shells(AtomicSymbol.C),
        formal_charge=0,
    ),
    AtomId(6): Atom(
        atom_id=AtomId(6),
        attributes=element_attributes(AtomicSymbol.F),
        coordinate=Coordinate(
            mk_angstrom(0.464),
            mk_angstrom(-2.45780034495),
            mk_angstrom(0.707),
        ),
        shells=element_shells(AtomicSymbol.F),
        formal_charge=0,
    ),
    AtomId(7): Atom(
        atom_id=AtomId(7),
        attributes=element_attributes(AtomicSymbol.F),
        coordinate=Coordinate(
            mk_angstrom(0.464),
            mk_angstrom(-2.45780034495),
            mk_angstrom(0.707),
        ),
        shells=element_shells(AtomicSymbol.F),
        formal_charge=0,
    ),
    AtomId(8): Atom(
        atom_id=AtomId(8),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(-1.386),
            mk_angstrom(0.919698729811),
            mk_angstrom(0.307),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(9): Atom(
        atom_id=AtomId(9),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(-2.211),
            mk_angstrom(-2.50110161514),
            mk_angstrom(0.407),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(10): Atom(
        atom_id=AtomId(10),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(-0.761),
            mk_angstrom(-2.41449907476),
            mk_angstrom(0.607),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(11): Atom(
        atom_id=AtomId(11),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(0.464),
            mk_angstrom(-2.45780034495),
            mk_angstrom(0.707),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(12): Atom(
        atom_id=AtomId(12),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(1.139),
            mk_angstrom(-1.37526859022),
            mk_angstrom(0.507),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(13): Atom(
        atom_id=AtomId(13),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(0.514),
            mk_angstrom(-0.206134295109),
            mk_angstrom(0.607),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
}

local_bonds = frozenset({
    mk_edge(AtomId(1), AtomId(2)),
    mk_edge(AtomId(1), AtomId(8)),
    mk_edge(AtomId(2), AtomId(3)),
    mk_edge(AtomId(3), AtomId(4)),
    mk_edge(AtomId(3), AtomId(5)),
    mk_edge(AtomId(3), AtomId(9)),
    mk_edge(AtomId(4), AtomId(10)),
    mk_edge(AtomId(4), AtomId(11)),
    mk_edge(AtomId(4), AtomId(12)),
    mk_edge(AtomId(5), AtomId(6)),
    mk_edge(AtomId(5), AtomId(7)),
    mk_edge(AtomId(5), AtomId(13)),
})

systems = (
)

molecule = validate_molecule(
    Molecule(
        atoms=atoms,
        local_bonds=local_bonds,
        systems=systems,
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
