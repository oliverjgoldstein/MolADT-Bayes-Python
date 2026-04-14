from __future__ import annotations

from collections import Counter
from pathlib import Path

from moladt.chem.dietz import AtomId
from moladt.examples.morphine import MORPHINE_RING_CLOSURE_SMILES
from moladt.examples.sample_molecules import methane, water
from moladt.io.molecule_json import molecule_from_json, molecule_to_json_bytes
from moladt.io.sdf import molecule_to_sdf, parse_sdf, parse_sdf_record, read_sdf, read_sdf_record
from moladt.io.smiles import molecule_to_smiles, parse_smiles


PROJECT_ROOT = Path(__file__).resolve().parent.parent
V3000_WATER = """water
MolADT
generated
  0  0  0  0  0  0            999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 3 2 0 0 0
M  V30 BEGIN ATOM
M  V30 1 O 0.0000 0.0000 0.0000 0
M  V30 2 H 0.9572 0.0000 0.0000 0
M  V30 3 H -0.2390 0.9270 0.0000 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 1 2
M  V30 2 1 1 3
M  V30 END BOND
M  V30 END CTAB
M  END
> <source>
v3000

$$$$
"""

V3000_AMMONIUM = """ammonium
MolADT
generated
  0  0  0  0  0  0            999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 5 4 0 0 0
M  V30 BEGIN ATOM
M  V30 1 N 0.0000 0.0000 0.0000 0 CHG=1
M  V30 2 H 0.9000 0.0000 0.0000 0
M  V30 3 H -0.3000 0.8500 0.0000 0
M  V30 4 H -0.3000 -0.4000 0.8000 0
M  V30 5 H -0.3000 -0.4000 -0.8000 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 1 2
M  V30 2 1 1 3
M  V30 3 1 1 4
M  V30 4 1 1 5
M  V30 END BOND
M  V30 END CTAB
M  END
$$$$
"""


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


def test_small_sample_sdfs_smoke_parse() -> None:
    hydrogen = read_sdf(PROJECT_ROOT / "molecules" / "hydrogen.sdf")
    oxygen = read_sdf(PROJECT_ROOT / "molecules" / "oxygen.sdf")
    methane = read_sdf(PROJECT_ROOT / "molecules" / "methane.sdf")

    assert len(hydrogen.atoms) == 2
    assert len(hydrogen.local_bonds) == 1
    assert len(oxygen.atoms) == 2
    assert len(oxygen.local_bonds) == 1
    assert len(methane.atoms) == 5
    assert len(methane.local_bonds) == 4


def test_psilocybin_sdf_smoke_parse() -> None:
    record = read_sdf_record(PROJECT_ROOT / "molecules" / "psilocybin.sdf")

    assert record.title == "psilocybin"
    assert record.property("pubchem_url") == "https://pubchem.ncbi.nlm.nih.gov/compound/10624#section=Structures"
    assert len(record.molecule.atoms) == 19
    assert len(record.molecule.local_bonds) == 20


def test_manuscript_example_sdfs_smoke_parse() -> None:
    diborane = read_sdf(PROJECT_ROOT / "molecules" / "diborane.sdf")
    ferrocene = read_sdf(PROJECT_ROOT / "molecules" / "ferrocene.sdf")
    morphine = read_sdf(PROJECT_ROOT / "molecules" / "morphine.sdf")

    assert len(diborane.atoms) == 8
    assert len(diborane.local_bonds) == 5
    assert len(ferrocene.atoms) == 21
    assert len(ferrocene.local_bonds) == 20
    assert len(morphine.atoms) == 21
    assert len(morphine.local_bonds) == 25


def test_v3000_water_record_parse_preserves_properties() -> None:
    record = parse_sdf_record(V3000_WATER)

    assert record.title == "water"
    assert record.property("source") == "v3000"
    assert len(record.molecule.atoms) == 3
    assert len(record.molecule.local_bonds) == 2


def test_v3000_atom_charge_is_read_from_atom_tokens() -> None:
    molecule = parse_sdf(V3000_AMMONIUM)

    assert molecule.atoms[AtomId(1)].formal_charge == 1


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
    assert [system.tag for _, system in molecule.systems] == [None, None, None, None]
    assert [
        (item.center.value, item.stereo_class.value, item.configuration, item.token)
        for item in molecule.smiles_stereochemistry.atom_stereo
    ] == [
        (5, "TH", 1, "@"),
        (14, "TH", 1, "@"),
        (16, "TH", 2, "@@"),
        (21, "TH", 1, "@"),
        (23, "TH", 1, "@"),
    ]


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
    restored = molecule_from_json(molecule_to_json_bytes(molecule))

    assert restored.smiles_stereochemistry == molecule.smiles_stereochemistry
