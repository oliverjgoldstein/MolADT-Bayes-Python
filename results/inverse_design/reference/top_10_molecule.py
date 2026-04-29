from __future__ import annotations

from moladt.chem.constants import element_attributes, element_shells
from moladt.chem.coordinate import Coordinate, mk_angstrom
from moladt.chem.dietz import AtomId, NonNegative, SystemId, mk_bonding_system, mk_edge
from moladt.chem.molecule import Atom, AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule

rank = 10
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -3.76139155159
predictive_sd = 1.50415799052
target_error = 1.23860844841
bayesian_credible_score_percent = 43.763717176
score = -0.826365086874
formula = 'C2H3ClF2O2'

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
        attributes=element_attributes(AtomicSymbol.Cl),
        coordinate=Coordinate(
            mk_angstrom(-0.161),
            mk_angstrom(-1.37526859022),
            mk_angstrom(0.507),
        ),
        shells=element_shells(AtomicSymbol.Cl),
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
            mk_angstrom(0.589),
            mk_angstrom(-0.249435565298),
            mk_angstrom(0.107),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(9): Atom(
        atom_id=AtomId(9),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(-0.136),
            mk_angstrom(-1.33196732003),
            mk_angstrom(0.507),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(10): Atom(
        atom_id=AtomId(10),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(-0.861),
            mk_angstrom(-0.16283302492),
            mk_angstrom(0.707),
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
    mk_edge(AtomId(5), AtomId(6)),
    mk_edge(AtomId(5), AtomId(7)),
    mk_edge(AtomId(5), AtomId(10)),
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
