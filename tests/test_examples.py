from __future__ import annotations

from scripts.features import compute_moladt_featurized_descriptors

from moladt.chem.validate import validate_molecule
from moladt.examples import diborane_pretty, ferrocene_pretty, morphine_pretty


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
