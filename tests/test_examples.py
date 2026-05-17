from __future__ import annotations

import ast
from pathlib import Path

import pytest

from benchmarking.features import compute_moladt_featurized_descriptors

from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule, molecule_edges
from moladt.chem.validate import validate_molecule
from moladt.cli import DEFAULT_VIEW_EXAMPLES, EXAMPLE_VIEWER_MOLECULES
from moladt.examples import benzene_pretty, diborane_pretty, ferrocene_pretty, morphine_pretty
from moladt.examples._literal import atom
from moladt.examples.sample_molecules import hydrogen, methane, oxygen, sodium_chloride, water
from moladt.io import molecule_to_python_literal


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
    return sorted((edge.a.value, edge.b.value) for edge in molecule_edges(molecule))


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
    bridges = [system for _, system in diborane_pretty.systems if system.tag and system.tag.startswith("bridge_")]
    assert [system.tag for system in bridges] == ["bridge_h3_3c2e", "bridge_h4_3c2e"]
    assert [system.shared_electrons.value for system in bridges] == [2, 2]
    assert [len(system.member_edges) for system in bridges] == [2, 2]
    assert _count_unnamed_edge_systems(diborane_pretty, 2) == 4


def test_ferrocene_constructs_a_valid_molecule() -> None:
    assert validate_molecule(ferrocene_pretty) == ferrocene_pretty


def test_ferrocene_example_has_pi_and_coordination_systems() -> None:
    named = [system for _, system in ferrocene_pretty.systems if system.tag]
    assert [system.tag for system in named] == [
        "cp1_pi",
        "cp2_pi",
    ]
    assert [system.shared_electrons.value for system in named] == [6, 6]
    assert [len(system.member_edges) for system in named] == [10, 10]
    assert AtomId(1) in named[0].member_atoms
    assert AtomId(1) in named[1].member_atoms
    assert _count_unnamed_edge_systems(ferrocene_pretty, 2) == 20
    assert ferrocene_pretty.atoms[AtomId(1)].formal_charge == 2
    assert ferrocene_pretty.atoms[AtomId(2)].formal_charge == -1
    assert ferrocene_pretty.atoms[AtomId(7)].formal_charge == -1
    assert sum(atom.formal_charge for atom in ferrocene_pretty.atoms.values()) == 0


def test_ferrocene_typed_descriptors_use_canonical_dietz_edges() -> None:
    descriptors = compute_moladt_featurized_descriptors(ferrocene_pretty)

    assert descriptors["bonding_system_count"] == 2.0
    assert descriptors["system_shared_electrons_sum"] == 12.0
    assert descriptors["system_member_edges_max"] == 10.0


def test_morphine_constructs_a_valid_molecule() -> None:
    assert validate_molecule(morphine_pretty) == morphine_pretty


def test_benzene_and_morphine_examples_have_documented_pi_systems() -> None:
    assert _special_tags(benzene_pretty) == ["pi_ring"]
    assert _special_tags(morphine_pretty) == ["phenyl_pi_ring"]
    assert _count_unnamed_edge_systems(morphine_pretty, 2) == 24
    assert _count_unnamed_edge_systems(morphine_pretty, 4) == 1


def test_benchmark_descriptors_do_not_count_conventional_singletons_as_legacy_systems() -> None:
    descriptors = compute_moladt_featurized_descriptors(benzene_pretty)

    assert descriptors["bonding_system_count"] == 1.0
    assert descriptors["system_shared_electrons_sum"] == 6.0
    assert descriptors["sigma_edge_count"] == 12.0


def test_default_view_examples_are_valid_adt_examples_with_systems_preserved() -> None:
    entries = [EXAMPLE_VIEWER_MOLECULES[name] for name in DEFAULT_VIEW_EXAMPLES]
    tags_by_title = {
        title: _special_tags(validate_molecule(molecule))
        for title, molecule in entries
    }

    assert tags_by_title["Benzene"] == ["pi_ring"]
    assert tags_by_title["Diborane (B2H6)"] == ["bridge_h3_3c2e", "bridge_h4_3c2e"]
    assert tags_by_title["Ferrocene (Fe(C5H5)2)"] == [
        "cp1_pi",
        "cp2_pi",
    ]
    assert tags_by_title["Morphine"] == ["phenyl_pi_ring"]


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
        sodium_chloride,
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


def test_checked_examples_are_canonical_expanded_molecules() -> None:
    examples = (
        benzene_pretty,
        diborane_pretty,
        ferrocene_pretty,
        morphine_pretty,
        hydrogen,
        oxygen,
        sodium_chloride,
        water,
        methane,
    )

    for molecule in examples:
        assert [atom_id.value for atom_id in molecule.atoms] == sorted(
            atom_id.value for atom_id in molecule.atoms
        )
        assert all(edge.a.value < edge.b.value for edge in molecule_edges(molecule))
        assert [system_id.value for system_id, _ in molecule.systems] == sorted(
            system_id.value for system_id, _ in molecule.systems
        )
        for _, system in molecule.systems:
            assert all(edge.a.value < edge.b.value for edge in system.member_edges)


def test_python_literal_export_is_the_canonical_normal_form() -> None:
    molecule = Molecule(
        atoms={
            AtomId(2): atom(2, AtomicSymbol.H, 0.0, 0.0, 0.9),
            AtomId(1): atom(1, AtomicSymbol.O, 0.0, 0.0, 0.0),
        },
        systems=(
            (
                SystemId(2),
                mk_bonding_system(NonNegative(2), frozenset({Edge(AtomId(2), AtomId(1))}), "later"),
            ),
            (
                SystemId(1),
                mk_bonding_system(NonNegative(2), frozenset({Edge(AtomId(2), AtomId(1))}), "earlier"),
            ),
        ),
    )

    source = molecule_to_python_literal(molecule, variable_name="canonical")

    assert source.index("AtomId(1): atom") < source.index("AtomId(2): atom")
    assert "Edge(AtomId(1), AtomId(2))" in source
    assert "Edge(AtomId(2), AtomId(1))" not in source
    assert source.index("SystemId(1)") < source.index("SystemId(2)")


def test_explicit_examples_keep_expected_sigma_edges() -> None:
    assert _edge_pairs(benzene_pretty) == [
        (1, 2),
        (1, 3),
        (1, 7),
        (2, 4),
        (2, 8),
        (3, 5),
        (3, 9),
        (4, 6),
        (4, 10),
        (5, 6),
        (5, 11),
        (6, 12),
    ]
    assert _edge_pairs(diborane_pretty) == [
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 3),
        (2, 4),
        (2, 7),
        (2, 8),
    ]
    assert _edge_pairs(ferrocene_pretty) == [
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 10),
        (1, 11),
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
    assert _rounded_coordinates(sodium_chloride) == {1: ("Na", 0.0, 0.0, 0.0), 2: ("Cl", 2.36, 0.0, 0.0)}
    assert _edge_pairs(sodium_chloride) == [(1, 2)]
    assert sodium_chloride.atoms[AtomId(1)].formal_charge == 1
    assert sodium_chloride.atoms[AtomId(2)].formal_charge == -1
    assert [
        (system_id.value, system.shared_electrons.value, system.tag)
        for system_id, system in sodium_chloride.systems
    ] == [(1, 0, "ionic")]
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


def _special_tags(molecule: Molecule) -> list[str | None]:
    return [system.tag for _, system in molecule.systems if system.tag]


def _count_unnamed_edge_systems(molecule: Molecule, electrons: int) -> int:
    return sum(
        1
        for _, system in molecule.systems
        if system.tag is None
        and len(system.member_edges) == 1
        and system.shared_electrons.value == electrons
    )
