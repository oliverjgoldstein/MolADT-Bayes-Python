from dataclasses import replace

from moladt.chem.coordinate import Coordinate, mk_angstrom
from moladt.chem.dietz import AtomId
from moladt.chem.molecule import molecule_edges
from moladt.chem.mutable import MutableMolecule
from moladt.examples import ferrocene_pretty
from moladt.examples.sample_molecules import water
from moladt.io.smiles import parse_smiles


def test_molecule_to_mutable_round_trips_back_to_same_immutable_value() -> None:
    molecule = parse_smiles("F[C@](Cl)(Br)I")

    mutable = MutableMolecule.from_molecule(molecule)
    frozen = mutable.freeze()

    assert frozen.atoms == molecule.atoms
    assert molecule_edges(frozen) == molecule_edges(molecule)
    assert frozen.systems == molecule.systems
    assert frozen.smiles_stereochemistry == molecule.smiles_stereochemistry


def test_mutable_molecule_uses_mutable_collections_for_proposal_edits() -> None:
    mutable = MutableMolecule.from_molecule(water)

    assert isinstance(mutable, MutableMolecule)
    assert isinstance(mutable.atoms, dict)
    assert isinstance(mutable.edges, set)
    assert isinstance(mutable.systems, list)

    mutable.systems.clear()
    mutable.atoms[AtomId(1)] = replace(
        mutable.atoms[AtomId(1)],
        coordinate=Coordinate(mk_angstrom(1.0), mk_angstrom(2.0), mk_angstrom(3.0)),
        formal_charge=1,
    )

    assert len(molecule_edges(water)) == 2
    assert water.atoms[AtomId(1)].formal_charge == 0
    assert water.atoms[AtomId(1)].coordinate != mutable.atoms[AtomId(1)].coordinate

    proposal = mutable.freeze()
    assert len(molecule_edges(proposal)) == 0
    assert proposal.atoms[AtomId(1)].formal_charge == 1


def test_mutable_molecule_system_edits_do_not_touch_original() -> None:
    mutable = MutableMolecule.from_molecule(ferrocene_pretty)

    mutable.systems.clear()

    assert [system.tag for _, system in ferrocene_pretty.systems if system.tag] == [
        "cp1_pi",
        "cp2_pi",
        "fe_cp_coordination",
    ]
    assert mutable.freeze().systems == ()
