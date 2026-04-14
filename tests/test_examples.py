from __future__ import annotations

from scripts.features import compute_moladt_featurized_descriptors

from moladt.chem.validate import validate_molecule
from moladt.examples import diborane_pretty, ferrocene_pretty, morphine_pretty, psilocybin_pretty
from moladt.examples.sample_molecules import hydrogen, methane, oxygen, water
from moladt.io.sdf import read_sdf, read_sdf_record


def _rounded_coordinates(molecule):
    return {
        atom_id.value: (
            atom.attributes.symbol.value,
            round(atom.coordinate.x.value, 3),
            round(atom.coordinate.y.value, 3),
            round(atom.coordinate.z.value, 3),
        )
        for atom_id, atom in molecule.atoms.items()
    }


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


def test_all_examples_include_orbital_shells() -> None:
    examples = (
        diborane_pretty,
        ferrocene_pretty,
        morphine_pretty,
        psilocybin_pretty,
        hydrogen,
        oxygen,
        water,
        methane,
    )

    for molecule in examples:
        assert all(atom.shells for atom in molecule.atoms.values())


def test_manuscript_examples_keep_sdf_reference_geometry_and_sigma_edges() -> None:
    diborane_record = read_sdf_record("molecules/diborane.sdf")
    ferrocene_record = read_sdf_record("molecules/ferrocene.sdf")
    morphine_record = read_sdf_record("molecules/morphine.sdf")
    psilocybin_record = read_sdf_record("molecules/psilocybin.sdf")

    assert _rounded_coordinates(diborane_pretty) == _rounded_coordinates(diborane_record.molecule)
    assert diborane_pretty.local_bonds == diborane_record.molecule.local_bonds
    assert _rounded_coordinates(ferrocene_pretty) == _rounded_coordinates(ferrocene_record.molecule)
    assert ferrocene_pretty.local_bonds == ferrocene_record.molecule.local_bonds
    assert _rounded_coordinates(morphine_pretty) == _rounded_coordinates(morphine_record.molecule)
    assert morphine_pretty.local_bonds == morphine_record.molecule.local_bonds
    assert _rounded_coordinates(psilocybin_pretty) == _rounded_coordinates(psilocybin_record.molecule)
    assert psilocybin_pretty.local_bonds == psilocybin_record.molecule.local_bonds


def test_small_example_molecules_match_sdf_geometry_and_sigma_edges() -> None:
    hydrogen_sdf = read_sdf("molecules/hydrogen.sdf")
    oxygen_sdf = read_sdf("molecules/oxygen.sdf")
    water_sdf = read_sdf("molecules/water.sdf")
    methane_sdf = read_sdf("molecules/methane.sdf")

    assert _rounded_coordinates(hydrogen) == _rounded_coordinates(hydrogen_sdf)
    assert hydrogen.local_bonds == hydrogen_sdf.local_bonds
    assert _rounded_coordinates(oxygen) == _rounded_coordinates(oxygen_sdf)
    assert oxygen.local_bonds == oxygen_sdf.local_bonds
    assert _rounded_coordinates(water) == _rounded_coordinates(water_sdf)
    assert water.local_bonds == water_sdf.local_bonds
    assert _rounded_coordinates(methane) == _rounded_coordinates(methane_sdf)
    assert methane.local_bonds == methane_sdf.local_bonds
