from moladt.chem import (
    molecule_atoms,
    molecule_edges,
    molecule_fields,
    molecule_smiles_stereochemistry,
    molecule_systems,
)
from moladt.examples.benzene import benzene
from moladt.examples.sample_molecules import water
from moladt.io.smiles import parse_smiles


def test_molecule_fields_follow_adt_field_order() -> None:
    atoms, systems, smiles_stereochemistry = molecule_fields(water)

    assert atoms is water.atoms
    assert systems is water.systems
    assert smiles_stereochemistry is water.smiles_stereochemistry


def test_molecule_accessors_return_the_underlying_fields() -> None:
    assert molecule_atoms(water) is water.atoms
    assert molecule_edges(water) == frozenset(edge for _, system in water.systems for edge in system.member_edges)
    assert molecule_systems(water) is water.systems
    assert molecule_smiles_stereochemistry(water) is water.smiles_stereochemistry


def test_molecule_fields_remain_individually_iterable() -> None:
    atom_id, atom = next(iter(water.atoms.items()))
    edge = next(iter(molecule_edges(water)))
    _, system = next((item for item in benzene.systems if item[1].tag == "pi_ring"))
    stereo_molecule = parse_smiles("F[C@](Cl)(Br)I")
    stereo = next(iter((stereo_molecule.smiles_stereochemistry.atom_stereo, stereo_molecule.smiles_stereochemistry.bond_stereo)))

    assert atom_id == atom.atom_id
    assert edge in molecule_edges(water)
    assert system.tag == "pi_ring"
    assert len(stereo) == 1
