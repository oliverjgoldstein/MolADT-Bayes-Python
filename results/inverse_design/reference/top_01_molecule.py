from __future__ import annotations

from moladt.chem.constants import element_attributes, element_shells
from moladt.chem.coordinate import Coordinate, mk_angstrom
from moladt.chem.dietz import AtomId, NonNegative, SystemId, mk_bonding_system, mk_edge
from moladt.chem.molecule import Atom, AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule

rank = 1
target_freesolv = -5
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -5.01628547231
predictive_sd = 6.15640095498
target_error = 0.016285472315
score = -0.016285472315
formula = 'CH3FO'

atoms = {
    AtomId(1): Atom(
        atom_id=AtomId(1),
        attributes=element_attributes(AtomicSymbol.C),
        coordinate=Coordinate(
            mk_angstrom(-0.711),
            mk_angstrom(-0.249435565298),
            mk_angstrom(0.107),
        ),
        shells=element_shells(AtomicSymbol.C),
        formal_charge=0,
    ),
    AtomId(2): Atom(
        atom_id=AtomId(2),
        attributes=element_attributes(AtomicSymbol.F),
        coordinate=Coordinate(
            mk_angstrom(-1.411),
            mk_angstrom(-1.4618711306),
            mk_angstrom(0.207),
        ),
        shells=element_shells(AtomicSymbol.F),
        formal_charge=0,
    ),
    AtomId(3): Atom(
        atom_id=AtomId(3),
        attributes=element_attributes(AtomicSymbol.O),
        coordinate=Coordinate(
            mk_angstrom(-1.411),
            mk_angstrom(-1.4618711306),
            mk_angstrom(0.207),
        ),
        shells=element_shells(AtomicSymbol.O),
        formal_charge=0,
    ),
    AtomId(4): Atom(
        atom_id=AtomId(4),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(-1.411),
            mk_angstrom(-1.4618711306),
            mk_angstrom(0.207),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(5): Atom(
        atom_id=AtomId(5),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(-0.111),
            mk_angstrom(-1.28866604984),
            mk_angstrom(0.307),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(6): Atom(
        atom_id=AtomId(6),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(-0.161),
            mk_angstrom(-1.4618711306),
            mk_angstrom(0.207),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
}

local_bonds = frozenset({
    mk_edge(AtomId(1), AtomId(2)),
    mk_edge(AtomId(1), AtomId(3)),
    mk_edge(AtomId(1), AtomId(4)),
    mk_edge(AtomId(1), AtomId(5)),
    mk_edge(AtomId(3), AtomId(6)),
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
    'score',
    'formula',
    'molecule',
]
