from __future__ import annotations

from scripts.features import compute_moladt_featurized_descriptors

from moladt.chem.validate import validate_molecule
from moladt.examples import diborane_pretty, ferrocene_pretty, morphine_pretty, psilocybin_pretty
from moladt.examples.sample_molecules import hydrogen, methane, oxygen, water
from moladt.io.sdf import read_sdf, read_sdf_record


def test_diborane_constructs_a_valid_molecule() -> None:
    assert validate_molecule(diborane_pretty) == diborane_pretty


def test_ferrocene_constructs_a_valid_molecule() -> None:
    assert validate_molecule(ferrocene_pretty) == ferrocene_pretty


def test_ferrocene_typed_descriptors_use_canonical_dietz_edges() -> None:
    descriptors = compute_moladt_featurized_descriptors(ferrocene_pretty)

    assert descriptors["system_shared_electrons_sum"] == 18.0
    assert descriptors["system_member_edges_max"] == 10.0


def test_morphine_constructs_a_valid_molecule() -> None:
    assert validate_molecule(morphine_pretty) == morphine_pretty


def test_morphine_example_keeps_the_documented_stereochemistry_flags() -> None:
    assert [
        (item.center.value, item.stereo_class.value, item.configuration, item.token)
        for item in morphine_pretty.smiles_stereochemistry.atom_stereo
    ] == [
        (2, "TH", 1, "@"),
        (3, "TH", 2, "@@"),
        (7, "TH", 1, "@"),
        (8, "TH", 1, "@"),
        (18, "TH", 1, "@"),
    ]


def test_psilocybin_constructs_a_valid_molecule() -> None:
    assert validate_molecule(psilocybin_pretty) == psilocybin_pretty


def test_psilocybin_typed_descriptors_keep_indole_and_phosphoryl_systems() -> None:
    descriptors = compute_moladt_featurized_descriptors(psilocybin_pretty)

    assert descriptors["system_shared_electrons_sum"] == 12.0
    assert descriptors["system_member_edges_max"] == 10.0


def test_manuscript_examples_keep_sdf_backed_atoms_and_sigma_edges() -> None:
    diborane_record = read_sdf_record("molecules/diborane.sdf")
    ferrocene_record = read_sdf_record("molecules/ferrocene.sdf")
    morphine_record = read_sdf_record("molecules/morphine.sdf")
    psilocybin_record = read_sdf_record("molecules/psilocybin.sdf")

    assert diborane_pretty.atoms == diborane_record.molecule.atoms
    assert diborane_pretty.local_bonds == diborane_record.molecule.local_bonds
    assert ferrocene_pretty.atoms == ferrocene_record.molecule.atoms
    assert ferrocene_pretty.local_bonds == ferrocene_record.molecule.local_bonds
    assert morphine_pretty.atoms == morphine_record.molecule.atoms
    assert morphine_pretty.local_bonds == morphine_record.molecule.local_bonds
    assert psilocybin_pretty.atoms == psilocybin_record.molecule.atoms
    assert psilocybin_pretty.local_bonds == psilocybin_record.molecule.local_bonds


def test_small_example_molecules_are_sdf_backed() -> None:
    assert hydrogen == read_sdf("molecules/hydrogen.sdf")
    assert oxygen == read_sdf("molecules/oxygen.sdf")
    assert water == read_sdf("molecules/water.sdf")
    assert methane == read_sdf("molecules/methane.sdf")
