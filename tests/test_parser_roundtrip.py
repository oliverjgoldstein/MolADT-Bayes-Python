from __future__ import annotations

from collections import Counter
from pathlib import Path

from moladt.examples.morphine import MORPHINE_RING_CLOSURE_SMILES
from moladt.examples.sample_molecules import methane, water
from moladt.io.sdf import molecule_to_sdf, parse_sdf, read_sdf, read_sdf_record
from moladt.io.smiles import molecule_to_smiles, parse_smiles


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_benzene_round_trip_preserves_atom_count_and_sigma_bonds() -> None:
    molecule = read_sdf(PROJECT_ROOT / "molecules" / "benzene.sdf")
    round_tripped = parse_sdf(molecule_to_sdf(molecule))
    assert len(round_tripped.atoms) == len(molecule.atoms)
    assert round_tripped.local_bonds == molecule.local_bonds


def test_benzene_detects_one_pi_ring() -> None:
    record = read_sdf_record(PROJECT_ROOT / "molecules" / "benzene.sdf")
    assert len(record.molecule.systems) == 1
    assert record.molecule.systems[0][1].tag == "pi_ring"


def test_water_smoke_parse() -> None:
    molecule = read_sdf(PROJECT_ROOT / "molecules" / "water.sdf")
    assert len(molecule.atoms) == 3
    assert len(molecule.local_bonds) == 2


def test_smiles_parse_recovers_benzene_pi_ring() -> None:
    molecule = parse_smiles("c1ccccc1")
    counts = Counter(atom.attributes.symbol.value for atom in molecule.atoms.values())
    assert counts == Counter({"C": 6, "H": 6})
    assert len(molecule.local_bonds) == 12
    assert len(molecule.systems) == 1
    assert molecule.systems[0][1].tag == "pi_ring"


def test_morphine_ring_closure_smiles_parses_as_a_boundary_format_example() -> None:
    molecule = parse_smiles(MORPHINE_RING_CLOSURE_SMILES)
    counts = Counter(atom.attributes.symbol.value for atom in molecule.atoms.values())
    assert counts == Counter({"C": 17, "H": 19, "N": 1, "O": 3})
    assert len(molecule.local_bonds) == 44
    assert [system.tag for _, system in molecule.systems] == [None, "pi_ring"]
    assert molecule.smiles_stereochemistry.atom_stereo == ()


def test_bare_atoms_infer_terminal_hydrogens() -> None:
    methane = parse_smiles("C")
    water_from_smiles = parse_smiles("O")

    methane_counts = Counter(atom.attributes.symbol.value for atom in methane.atoms.values())
    water_counts = Counter(atom.attributes.symbol.value for atom in water_from_smiles.atoms.values())

    assert methane_counts == Counter({"C": 1, "H": 4})
    assert len(methane.local_bonds) == 4
    assert water_counts == Counter({"O": 1, "H": 2})
    assert len(water_from_smiles.local_bonds) == 2


def test_bracket_atoms_do_not_gain_extra_implicit_hydrogens() -> None:
    radical_oxygen = parse_smiles("[O]")
    assert Counter(atom.attributes.symbol.value for atom in radical_oxygen.atoms.values()) == Counter({"O": 1})
    assert len(radical_oxygen.local_bonds) == 0


def test_water_smiles_round_trip_preserves_counts() -> None:
    smiles = molecule_to_smiles(water)
    round_tripped = parse_smiles(smiles)
    assert smiles == "[OH2]"
    assert len(round_tripped.atoms) == 3
    assert len(round_tripped.local_bonds) == 2


def test_methane_smiles_round_trip_preserves_counts() -> None:
    smiles = molecule_to_smiles(methane)
    round_tripped = parse_smiles(smiles)
    assert smiles == "[CH4]"
    assert len(round_tripped.atoms) == 5
    assert len(round_tripped.local_bonds) == 4


def test_benzene_kekule_smiles_round_trip_preserves_localized_double_bonds() -> None:
    molecule = read_sdf(PROJECT_ROOT / "molecules" / "benzene.sdf")
    smiles = molecule_to_smiles(molecule)
    round_tripped = parse_smiles(smiles)
    counts = Counter(atom.attributes.symbol.value for atom in round_tripped.atoms.values())
    assert smiles == "[CH]1=[CH][CH]=[CH][CH]=[CH]1"
    assert counts == Counter({"C": 6, "H": 6})
    assert len(round_tripped.local_bonds) == 12
    assert len(round_tripped.systems) == 3
    assert [system.tag for _, system in round_tripped.systems] == [None, None, None]


def test_smiles_parse_preserves_atom_centered_stereochemistry_annotations() -> None:
    molecule = parse_smiles("N[C@](Br)(O)C")

    assert len(molecule.smiles_stereochemistry.atom_stereo) == 1
    stereo = molecule.smiles_stereochemistry.atom_stereo[0]
    assert stereo.stereo_class.value == "TH"
    assert stereo.configuration == 1
    assert stereo.token == "@"


def test_smiles_parse_preserves_directional_bond_annotations() -> None:
    molecule = parse_smiles("F/C=C\\F")

    directions = {(item.start_atom.value, item.end_atom.value, item.direction.value) for item in molecule.smiles_stereochemistry.bond_stereo}
    assert directions == {(1, 2, "/"), (3, 4, "\\")}


def test_smiles_stereochemistry_survives_json_round_trip() -> None:
    molecule = parse_smiles("C[C@@H](O)F")
    restored = molecule.from_json(molecule.to_json_bytes())

    assert restored.smiles_stereochemistry == molecule.smiles_stereochemistry
