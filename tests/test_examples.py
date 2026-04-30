from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.features import compute_moladt_featurized_descriptors

from moladt.chem.dietz import AtomId
from moladt.chem.validate import validate_molecule
from moladt.cli import DEFAULT_VIEW_EXAMPLES, EXAMPLE_VIEWER_MOLECULES
from moladt.examples import benzene_pretty, diborane_pretty, ferrocene_pretty, morphine_pretty
from moladt.examples.sample_molecules import hydrogen, methane, oxygen, water


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def _edge_pairs(molecule):
    return sorted((edge.a.value, edge.b.value) for edge in molecule.local_bonds)


def _distance(molecule, atom_a: int, atom_b: int) -> float:
    left = molecule.atoms[AtomId(atom_a)].coordinate
    right = molecule.atoms[AtomId(atom_b)].coordinate
    return (
        (left.x.value - right.x.value) ** 2
        + (left.y.value - right.y.value) ** 2
        + (left.z.value - right.z.value) ** 2
    ) ** 0.5


def test_diborane_constructs_a_valid_molecule() -> None:
    assert validate_molecule(diborane_pretty) == diborane_pretty


def test_diborane_example_has_two_3c2e_bridge_systems() -> None:
    assert [system.tag for _, system in diborane_pretty.systems] == ["bridge_h3_3c2e", "bridge_h4_3c2e"]
    assert [system.shared_electrons.value for _, system in diborane_pretty.systems] == [2, 2]
    assert [len(system.member_edges) for _, system in diborane_pretty.systems] == [2, 2]


def test_ferrocene_constructs_a_valid_molecule() -> None:
    assert validate_molecule(ferrocene_pretty) == ferrocene_pretty


def test_ferrocene_example_has_pi_and_backdonation_systems() -> None:
    assert [system.tag for _, system in ferrocene_pretty.systems] == ["cp1_pi", "cp2_pi", "fe_backdonation"]
    assert [system.shared_electrons.value for _, system in ferrocene_pretty.systems] == [6, 6, 6]
    assert [len(system.member_edges) for _, system in ferrocene_pretty.systems] == [10, 10, 10]


def test_ferrocene_typed_descriptors_use_canonical_dietz_edges() -> None:
    descriptors = compute_moladt_featurized_descriptors(ferrocene_pretty)

    assert descriptors["system_shared_electrons_sum"] == 18.0
    assert descriptors["system_member_edges_max"] == 10.0


def test_morphine_constructs_a_valid_molecule() -> None:
    assert validate_molecule(morphine_pretty) == morphine_pretty


def test_benzene_and_morphine_examples_have_documented_pi_systems() -> None:
    assert [system.tag for _, system in benzene_pretty.systems] == ["pi_ring"]
    assert [system.tag for _, system in morphine_pretty.systems] == ["alkene_bridge", "phenyl_pi_ring"]


def test_default_view_examples_are_valid_adt_examples_with_systems_preserved() -> None:
    entries = [EXAMPLE_VIEWER_MOLECULES[name] for name in DEFAULT_VIEW_EXAMPLES]
    tags_by_title = {
        title: [system.tag for _, system in validate_molecule(molecule).systems]
        for title, molecule in entries
    }

    assert tags_by_title["Benzene"] == ["pi_ring"]
    assert tags_by_title["Diborane (B2H6)"] == ["bridge_h3_3c2e", "bridge_h4_3c2e"]
    assert tags_by_title["Ferrocene (Fe(C5H5)2)"] == ["cp1_pi", "cp2_pi", "fe_backdonation"]
    assert tags_by_title["Morphine"] == ["alkene_bridge", "phenyl_pi_ring"]


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


def test_all_examples_include_orbital_shells() -> None:
    examples = (
        diborane_pretty,
        ferrocene_pretty,
        morphine_pretty,
        hydrogen,
        oxygen,
        water,
        methane,
    )

    for molecule in examples:
        assert all(atom.shells for atom in molecule.atoms.values())


def test_example_sources_are_explicit_literals_without_generation_loops() -> None:
    forbidden_calls = ("range(", "zip(", "atom_map(", "sigma_bonds(")
    generated_nodes = (ast.For, ast.AsyncFor, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    for path in sorted((PROJECT_ROOT / "moladt" / "examples").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            assert not isinstance(node, generated_nodes), f"{path.name} contains generated example syntax"
        for pattern in forbidden_calls:
            assert pattern not in source, f"{path.name} contains generated example pattern {pattern!r}"


def test_explicit_examples_keep_expected_sigma_edges() -> None:
    assert _edge_pairs(diborane_pretty) == [(1, 2), (1, 5), (1, 6), (2, 7), (2, 8)]
    assert _edge_pairs(ferrocene_pretty) == [
        (2, 3),
        (2, 6),
        (2, 12),
        (3, 4),
        (3, 13),
        (4, 5),
        (4, 14),
        (5, 6),
        (5, 15),
        (6, 16),
        (7, 8),
        (7, 11),
        (7, 17),
        (8, 9),
        (8, 18),
        (9, 10),
        (9, 19),
        (10, 11),
        (10, 20),
        (11, 21),
    ]
    assert _edge_pairs(morphine_pretty) == [
        (1, 2),
        (1, 11),
        (2, 3),
        (2, 8),
        (3, 4),
        (3, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (7, 18),
        (8, 9),
        (8, 10),
        (9, 21),
        (10, 11),
        (10, 16),
        (11, 12),
        (12, 13),
        (12, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (19, 21),
    ]


def test_ferrocene_geometry_matches_sandwich_structure_distances() -> None:
    for carbon_id in range(2, 12):
        assert _distance(ferrocene_pretty, 1, carbon_id) == pytest.approx(2.046, abs=0.002)

    for atom_a, atom_b in ((2, 3), (3, 4), (4, 5), (5, 6), (2, 6), (7, 8), (8, 9), (9, 10), (10, 11), (7, 11)):
        assert _distance(ferrocene_pretty, atom_a, atom_b) == pytest.approx(1.404, abs=0.002)

    for atom_a, atom_b in ((2, 12), (3, 13), (4, 14), (5, 15), (6, 16), (7, 17), (8, 18), (9, 19), (10, 20), (11, 21)):
        assert _distance(ferrocene_pretty, atom_a, atom_b) == pytest.approx(1.090, abs=0.002)


def test_small_example_molecules_are_explicit_adt_values() -> None:
    assert _rounded_coordinates(hydrogen) == {1: ("H", 0.0, 0.0, -0.37), 2: ("H", 0.0, 0.0, 0.37)}
    assert _edge_pairs(hydrogen) == [(1, 2)]
    assert _rounded_coordinates(oxygen) == {1: ("O", 0.0, 0.0, -0.605), 2: ("O", 0.0, 0.0, 0.605)}
    assert _edge_pairs(oxygen) == [(1, 2)]
    assert _rounded_coordinates(water) == {
        1: ("H", 0.002, -0.004, 0.002),
        2: ("O", -0.011, 0.963, 0.007),
        3: ("H", 0.867, 1.368, 0.001),
    }
    assert _edge_pairs(water) == [(1, 2), (2, 3)]
    assert _rounded_coordinates(methane) == {
        1: ("C", 0.0, 0.0, 0.0),
        2: ("H", 0.629, 0.629, 0.629),
        3: ("H", -0.629, -0.629, 0.629),
        4: ("H", -0.629, 0.629, -0.629),
        5: ("H", 0.629, -0.629, -0.629),
    }
    assert _edge_pairs(methane) == [(1, 2), (1, 3), (1, 4), (1, 5)]
