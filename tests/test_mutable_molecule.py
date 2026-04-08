from dataclasses import replace

from moladt.chem.coordinate import Coordinate, mk_angstrom
from moladt.chem.dietz import AtomId
from moladt.chem.molecule import MutableMolecule
from moladt.examples.benzene import benzene
from moladt.examples.sample_molecules import water
from moladt.io.smiles import parse_smiles


def test_molecule_to_mutable_round_trips_back_to_same_immutable_value() -> None:
    molecule = parse_smiles("F[C@](Cl)(Br)I")

    mutable = molecule.to_mutable()
    frozen = mutable.freeze()

    assert frozen.atoms == molecule.atoms
    assert frozen.local_bonds == molecule.local_bonds
    assert frozen.systems == molecule.systems
    assert frozen.smiles_stereochemistry == molecule.smiles_stereochemistry


def test_mutable_molecule_uses_mutable_collections_for_proposal_edits() -> None:
    mutable = water.to_mutable()

    assert isinstance(mutable, MutableMolecule)
    assert isinstance(mutable.atoms, dict)
    assert isinstance(mutable.local_bonds, set)
    assert isinstance(mutable.systems, list)

    mutable.local_bonds.clear()
    mutable.atoms[AtomId(1)] = replace(
        mutable.atoms[AtomId(1)],
        coordinate=Coordinate(mk_angstrom(1.0), mk_angstrom(2.0), mk_angstrom(3.0)),
        formal_charge=1,
    )

    assert len(water.local_bonds) == 2
    assert water.atoms[AtomId(1)].formal_charge == 0
    assert water.atoms[AtomId(1)].coordinate != mutable.atoms[AtomId(1)].coordinate

    proposal = mutable.freeze()
    assert len(proposal.local_bonds) == 0
    assert proposal.atoms[AtomId(1)].formal_charge == 1


def test_mutable_molecule_system_edits_do_not_touch_original() -> None:
    mutable = benzene.to_mutable()

    mutable.systems.clear()

    assert len(benzene.systems) == 1
    assert mutable.freeze().systems == ()
