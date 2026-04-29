from __future__ import annotations

from moladt.chem.constants import element_attributes, element_shells
from moladt.chem.coordinate import Coordinate, mk_angstrom
from moladt.chem.dietz import AtomId, NonNegative, SystemId, mk_bonding_system, mk_edge
from moladt.chem.molecule import Atom, AtomicSymbol, Molecule
from moladt.chem.validate import validate_molecule

rank = 6
target_freesolv = -20
seed_molecule = 'water'
random_seed = 0
predicted_freesolv = -8.88687824158
predictive_sd = 5.39916236972
target_error = 11.1131217584
score = -3.75116079668
formula = 'C3H8ClN3O3'

atoms = {
    AtomId(1): Atom(
        atom_id=AtomId(1),
        attributes=element_attributes(AtomicSymbol.O),
        coordinate=Coordinate(
            mk_angstrom(-0.011),
            mk_angstrom(0.963),
            mk_angstrom(0.007),
        ),
        shells=element_shells(AtomicSymbol.O),
        formal_charge=0,
    ),
    AtomId(2): Atom(
        atom_id=AtomId(2),
        attributes=element_attributes(AtomicSymbol.C),
        coordinate=Coordinate(
            mk_angstrom(-0.711),
            mk_angstrom(-0.249435565298),
            mk_angstrom(0.107),
        ),
        shells=element_shells(AtomicSymbol.C),
        formal_charge=0,
    ),
    AtomId(3): Atom(
        atom_id=AtomId(3),
        attributes=element_attributes(AtomicSymbol.N),
        coordinate=Coordinate(
            mk_angstrom(-0.111),
            mk_angstrom(-1.28866604984),
            mk_angstrom(0.307),
        ),
        shells=element_shells(AtomicSymbol.N),
        formal_charge=0,
    ),
    AtomId(4): Atom(
        atom_id=AtomId(4),
        attributes=element_attributes(AtomicSymbol.C),
        coordinate=Coordinate(
            mk_angstrom(0.639),
            mk_angstrom(2.08883302492),
            mk_angstrom(0.107),
        ),
        shells=element_shells(AtomicSymbol.C),
        formal_charge=0,
    ),
    AtomId(5): Atom(
        atom_id=AtomId(5),
        attributes=element_attributes(AtomicSymbol.N),
        coordinate=Coordinate(
            mk_angstrom(-0.711),
            mk_angstrom(-2.32789653438),
            mk_angstrom(0.407),
        ),
        shells=element_shells(AtomicSymbol.N),
        formal_charge=0,
    ),
    AtomId(6): Atom(
        atom_id=AtomId(6),
        attributes=element_attributes(AtomicSymbol.C),
        coordinate=Coordinate(
            mk_angstrom(1.939),
            mk_angstrom(2.08883302492),
            mk_angstrom(0.107),
        ),
        shells=element_shells(AtomicSymbol.C),
        formal_charge=0,
    ),
    AtomId(7): Atom(
        atom_id=AtomId(7),
        attributes=element_attributes(AtomicSymbol.O),
        coordinate=Coordinate(
            mk_angstrom(2.614),
            mk_angstrom(3.25796732003),
            mk_angstrom(0.207),
        ),
        shells=element_shells(AtomicSymbol.O),
        formal_charge=0,
    ),
    AtomId(8): Atom(
        atom_id=AtomId(8),
        attributes=element_attributes(AtomicSymbol.Cl),
        coordinate=Coordinate(
            mk_angstrom(-1.411),
            mk_angstrom(-1.11546096908),
            mk_angstrom(0.607),
        ),
        shells=element_shells(AtomicSymbol.Cl),
        formal_charge=0,
    ),
    AtomId(9): Atom(
        atom_id=AtomId(9),
        attributes=element_attributes(AtomicSymbol.O),
        coordinate=Coordinate(
            mk_angstrom(-0.736),
            mk_angstrom(-2.37119780457),
            mk_angstrom(0.407),
        ),
        shells=element_shells(AtomicSymbol.O),
        formal_charge=0,
    ),
    AtomId(10): Atom(
        atom_id=AtomId(10),
        attributes=element_attributes(AtomicSymbol.N),
        coordinate=Coordinate(
            mk_angstrom(-0.061),
            mk_angstrom(-1.37526859022),
            mk_angstrom(0.307),
        ),
        shells=element_shells(AtomicSymbol.N),
        formal_charge=0,
    ),
    AtomId(11): Atom(
        atom_id=AtomId(11),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(0.589),
            mk_angstrom(-0.249435565298),
            mk_angstrom(0.107),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(12): Atom(
        atom_id=AtomId(12),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(1.314),
            mk_angstrom(3.25796732003),
            mk_angstrom(0.207),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(13): Atom(
        atom_id=AtomId(13),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(-0.061),
            mk_angstrom(3.30126859022),
            mk_angstrom(0.307),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(14): Atom(
        atom_id=AtomId(14),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(0.739),
            mk_angstrom(2.08883302492),
            mk_angstrom(0.107),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(15): Atom(
        atom_id=AtomId(15),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(1.989),
            mk_angstrom(2.1754355653),
            mk_angstrom(0.307),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(16): Atom(
        atom_id=AtomId(16),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(-0.086),
            mk_angstrom(-3.49703082949),
            mk_angstrom(0.607),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(17): Atom(
        atom_id=AtomId(17),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(1.289),
            mk_angstrom(-1.37526859022),
            mk_angstrom(0.307),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
    AtomId(18): Atom(
        atom_id=AtomId(18),
        attributes=element_attributes(AtomicSymbol.H),
        coordinate=Coordinate(
            mk_angstrom(0.639),
            mk_angstrom(-0.16283302492),
            mk_angstrom(0.407),
        ),
        shells=element_shells(AtomicSymbol.H),
        formal_charge=0,
    ),
}

local_bonds = frozenset({
    mk_edge(AtomId(1), AtomId(2)),
    mk_edge(AtomId(1), AtomId(4)),
    mk_edge(AtomId(2), AtomId(3)),
    mk_edge(AtomId(2), AtomId(10)),
    mk_edge(AtomId(2), AtomId(11)),
    mk_edge(AtomId(3), AtomId(5)),
    mk_edge(AtomId(3), AtomId(9)),
    mk_edge(AtomId(4), AtomId(6)),
    mk_edge(AtomId(4), AtomId(12)),
    mk_edge(AtomId(4), AtomId(13)),
    mk_edge(AtomId(5), AtomId(6)),
    mk_edge(AtomId(5), AtomId(8)),
    mk_edge(AtomId(6), AtomId(7)),
    mk_edge(AtomId(6), AtomId(14)),
    mk_edge(AtomId(7), AtomId(15)),
    mk_edge(AtomId(9), AtomId(16)),
    mk_edge(AtomId(10), AtomId(17)),
    mk_edge(AtomId(10), AtomId(18)),
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
