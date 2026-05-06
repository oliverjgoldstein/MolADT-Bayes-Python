from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from moladt.cli import main
import moladt.cli as cli
from moladt.examples import benzene_pretty, diborane_pretty, ferrocene_pretty, sodium_chloride
from moladt.io import molecule_to_json, parse_smiles
from moladt.viewer import (
    molecule_viewer_collection_html,
    molecule_viewer_collection_payload,
    molecule_viewer_html,
    molecule_viewer_payload,
    molecule_viewer_uri,
    write_molecule_viewer_html,
)


def _payload_atom(payload: dict, atom_id: int) -> dict:
    return next(atom for atom in payload["atoms"] if atom["id"] == atom_id)


def _payload_distance(payload: dict, atom_a: int, atom_b: int) -> float:
    left = _payload_atom(payload, atom_a)
    right = _payload_atom(payload, atom_b)
    return math.dist((left["x"], left["y"], left["z"]), (right["x"], right["y"], right["z"]))


def _payload_angle(payload: dict, atom_a: int, center: int, atom_b: int) -> float:
    left = _payload_atom(payload, atom_a)
    middle = _payload_atom(payload, center)
    right = _payload_atom(payload, atom_b)
    vector_a = (
        left["x"] - middle["x"],
        left["y"] - middle["y"],
        left["z"] - middle["z"],
    )
    vector_b = (
        right["x"] - middle["x"],
        right["y"] - middle["y"],
        right["z"] - middle["z"],
    )
    length_a = math.sqrt(sum(component * component for component in vector_a))
    length_b = math.sqrt(sum(component * component for component in vector_b))
    cosine = sum(
        component_a * component_b
        for component_a, component_b in zip(vector_a, vector_b, strict=True)
    ) / (length_a * length_b)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def test_molecule_viewer_payload_includes_bonding_system_annotations() -> None:
    payload = molecule_viewer_payload(diborane_pretty, title="Diborane")

    assert payload["format"] == "moladt-viewer-v1"
    assert payload["title"] == "Diborane"
    assert len(payload["atoms"]) == 8
    assert len(payload["bonds"]) == 8
    bridge_systems = [system for system in payload["systems"] if system["tag"] and system["tag"].startswith("bridge_")]
    assert [system["tag"] for system in bridge_systems] == ["bridge_h3_3c2e", "bridge_h4_3c2e"]
    assert [(edge["a"], edge["b"]) for edge in bridge_systems[0]["edges"]] == [(1, 3), (2, 3)]
    assert all("length" in edge for edge in bridge_systems[0]["edges"])


def test_molecule_viewer_payload_marks_ionic_edges_and_draws_charge_field() -> None:
    payload = molecule_viewer_payload(sodium_chloride, title="Sodium chloride")

    assert payload["atoms"][0]["charge"] == 1
    assert payload["atoms"][1]["charge"] == -1
    assert len(payload["bonds"]) == 1
    bond = payload["bonds"][0]
    assert (bond["a"], bond["b"]) == (1, 2)
    assert bond["order"] == 0.0
    assert bond["kind"] == "ionic"
    assert bond["length"] == pytest.approx(2.36)
    html = molecule_viewer_html(sodium_chloride, title="Sodium chloride")
    assert "POSITIVE_CHARGE_COLOR" in html
    assert "NEGATIVE_CHARGE_COLOR" in html
    assert "drawChargeField" in html
    assert "drawBondLines" in html
    assert "chargeGradientForEdge" in html
    assert '"kind":"ionic"' in html
    assert "setLineDash" not in html


def test_molecule_viewer_payload_lengths_come_from_3d_coordinates() -> None:
    payload = molecule_viewer_payload(ferrocene_pretty, title="Ferrocene")
    sigma_edges = {tuple(sorted((edge["a"], edge["b"]))): edge for edge in payload["bonds"]}
    coordination = next(system for system in payload["systems"] if system["tag"] == "fe_cp_coordination")
    coordination_edges = {
        tuple(sorted((edge["a"], edge["b"]))): edge
        for edge in coordination["edges"]
    }

    assert sigma_edges[(2, 3)]["length"] == pytest.approx(_payload_distance(payload, 2, 3))
    assert sigma_edges[(2, 3)]["length"] == pytest.approx(1.404, abs=0.002)
    assert coordination_edges[(1, 2)]["length"] == pytest.approx(_payload_distance(payload, 1, 2))
    assert coordination_edges[(1, 2)]["length"] == pytest.approx(2.046, abs=0.002)


def test_molecule_viewer_payload_angles_come_from_3d_coordinates() -> None:
    payload = molecule_viewer_payload(ferrocene_pretty, title="Ferrocene")
    angles = {
        (item["a"], item["center"], item["b"]): item["angle"]
        for item in payload["angles"]
    }

    assert angles[(3, 2, 6)] == pytest.approx(_payload_angle(payload, 3, 2, 6))
    assert angles[(3, 2, 6)] == pytest.approx(108.0, abs=0.2)
    assert angles[(2, 1, 7)] == pytest.approx(_payload_angle(payload, 2, 1, 7))


def test_molecule_viewer_payload_keeps_overlapping_system_colours_distinct() -> None:
    payload = molecule_viewer_payload(ferrocene_pretty, title="Ferrocene")

    edge_to_colours: dict[tuple[int, int], set[str]] = {}
    for system in payload["systems"]:
        for edge in system["edges"]:
            key = tuple(sorted((edge["a"], edge["b"])))
            edge_to_colours.setdefault(key, set()).add(system["color"])

    cp2_colour = next(system["color"] for system in payload["systems"] if system["tag"] == "cp2_pi")
    single_colour = next(
        system["color"]
        for system in payload["systems"]
        if system["tag"] is None
        and system["sharedElectrons"] == 2
        and any(tuple(sorted((edge["a"], edge["b"]))) == (7, 11) for edge in system["edges"])
    )

    assert edge_to_colours[(7, 11)] == {cp2_colour, single_colour}
    assert cp2_colour != single_colour


def test_molecule_viewer_payload_uses_grey_covalent_and_coloured_nonstandard_systems() -> None:
    covalent_colour = "#374151"
    single_payload = molecule_viewer_payload(parse_smiles("CC"), title="Ethane")
    double_payload = molecule_viewer_payload(parse_smiles("C=C"), title="Ethene")
    triple_payload = molecule_viewer_payload(parse_smiles("C#N"), title="Hydrogen cyanide")
    quadruple_payload = molecule_viewer_payload(parse_smiles("C$C"), title="Quadruple covalent")
    ionic_payload = molecule_viewer_payload(sodium_chloride, title="Sodium chloride")
    benzene_payload = molecule_viewer_payload(benzene_pretty, title="Benzene")

    assert {system["color"] for system in single_payload["systems"] if "single covalent" in system["label"]} == {
        covalent_colour
    }
    assert next(system["color"] for system in double_payload["systems"] if "double covalent" in system["label"]) == covalent_colour
    assert next(system["color"] for system in triple_payload["systems"] if "triple covalent" in system["label"]) == covalent_colour
    assert next(system["color"] for system in quadruple_payload["systems"] if "quadruple covalent" in system["label"]) == covalent_colour
    assert next(system["color"] for system in ionic_payload["systems"] if system["label"].endswith("ionic")) == "#0f766e"

    pi_ring_colour = next(system["color"] for system in benzene_payload["systems"] if system["tag"] == "pi_ring")
    assert pi_ring_colour not in {covalent_colour, "#0f766e"}


def test_molecule_viewer_payload_includes_atom_orbitals() -> None:
    payload = molecule_viewer_payload(ferrocene_pretty, title="Ferrocene")
    carbon = next(atom for atom in payload["atoms"] if atom["symbol"] == "C")

    assert carbon["shells"][1]["n"] == 2
    p_orbitals = [orbital for orbital in carbon["shells"][1]["orbitals"] if orbital["kind"] == "p"]
    assert [orbital["orbital"] for orbital in p_orbitals] == ["px", "py", "pz"]
    assert [orbital["electrons"] for orbital in p_orbitals] == [1, 1, 0]
    assert p_orbitals[0]["orientation"] == {"x": 1.0, "y": 0.0, "z": 0.0}


def test_molecule_viewer_collection_payload_lists_multiple_molecules() -> None:
    payload = molecule_viewer_collection_payload(
        (
            ("Diborane", diborane_pretty),
            ("Ferrocene", ferrocene_pretty),
        ),
        title="Two molecules",
    )

    assert payload["format"] == "moladt-viewer-collection-v1"
    assert payload["title"] == "Two molecules"
    assert [item["title"] for item in payload["molecules"]] == ["Diborane", "Ferrocene"]


def test_molecule_viewer_html_offsets_overlapping_system_edges() -> None:
    html = molecule_viewer_html(ferrocene_pretty, title="Ferrocene")

    assert "function systemEdgeLaneMap" in html
    assert "offset: laneOffset" in html
    assert "alpha: active ? 1 : 0.18" in html


def test_molecule_viewer_html_can_draw_selected_atom_orbitals() -> None:
    html = molecule_viewer_html(ferrocene_pretty, title="Ferrocene")

    assert "function drawSelectedAtomOrbitals" in html
    assert "function drawDirectionalOrbital" in html
    assert "function drawOrbitalLobe" in html
    assert "orbitalSummary(atom)" in html
    assert "state.selectedAtom" in html


def test_molecule_viewer_html_draws_coordinate_axes() -> None:
    html = molecule_viewer_html(ferrocene_pretty, title="Ferrocene")

    assert 'id="toggle-axes"' in html
    assert "function drawCoordinateAxes" in html
    assert "axis.label + \" \" + tick.value + \"A\"" in html


def test_molecule_viewer_html_displays_3d_lengths_and_angles() -> None:
    html = molecule_viewer_html(ferrocene_pretty, title="Ferrocene")

    assert "function geometryEdgesForAtom" in html
    assert "function bondAnglesForAtom" in html
    assert "function normalizeAngles" in html
    assert "length: numericOrNull" in html
    assert "molecule.angles" in html
    assert "Edge Lengths From 3D Coordinates" in html
    assert "Bond Angles" in html


def test_molecule_viewer_html_can_switch_between_multiple_molecules() -> None:
    html = molecule_viewer_collection_html(
        (
            ("Diborane", diborane_pretty),
            ("Ferrocene", ferrocene_pretty),
        ),
        title="Two molecules",
    )

    assert "moladt-viewer-collection-v1" in html
    assert "function switchMolecule" in html
    assert 'id="molecule-list"' in html
    assert "Diborane" in html
    assert "Ferrocene" in html


def test_molecule_viewer_html_is_interactive_visual_document() -> None:
    html = molecule_viewer_html(diborane_pretty, title="Diborane")

    assert 'data-moladt-viewer' in html
    assert '<canvas id="molecule-canvas">' in html
    assert "window.loadMolADT" in html
    assert "bridge_h3_3c2e" in html
    assert "Drop MolADT JSON" in html
    assert "Molecule Report" not in html


def test_write_molecule_viewer_html_writes_standalone_file(tmp_path: Path) -> None:
    output_path = write_molecule_viewer_html(diborane_pretty, tmp_path / "diborane.html", title="Diborane")
    html = output_path.read_text(encoding="utf-8")

    assert output_path.exists()
    assert "<!doctype html>" in html
    assert json.loads(html.split('<script id="moladt-payload" type="application/json">')[1].split("</script>")[0])[
        "title"
    ] == "Diborane"


def test_view_html_cli_accepts_moladt_json(tmp_path: Path, capsys) -> None:
    json_path = tmp_path / "diborane.moladt.json"
    html_path = tmp_path / "diborane.viewer.html"
    json_path.write_text(molecule_to_json(diborane_pretty), encoding="utf-8")

    result = main(["view-html", str(json_path), "--format", "json", "--output", str(html_path)])
    output = capsys.readouterr().out

    assert result == 0
    assert str(html_path) in output
    assert f"Viewer URL: {molecule_viewer_uri(html_path)}" in output
    assert "bridge_h4_3c2e" in html_path.read_text(encoding="utf-8")


def test_view_html_cli_accepts_multiple_inputs_in_one_viewer(tmp_path: Path, capsys) -> None:
    diborane_json = tmp_path / "diborane.moladt.json"
    ferrocene_json = tmp_path / "ferrocene.moladt.json"
    html_path = tmp_path / "collection.viewer.html"
    diborane_json.write_text(molecule_to_json(diborane_pretty), encoding="utf-8")
    ferrocene_json.write_text(molecule_to_json(ferrocene_pretty), encoding="utf-8")

    result = main(
        [
            "view-html",
            str(diborane_json),
            str(ferrocene_json),
            "--output",
            str(html_path),
        ]
    )
    output = capsys.readouterr().out
    html = html_path.read_text(encoding="utf-8")

    assert result == 0
    assert str(html_path) in output
    assert f"Viewer URL: {molecule_viewer_uri(html_path)}" in output
    assert "moladt-viewer-collection-v1" in html
    assert "diborane" in html
    assert "ferrocene" in html


def test_view_html_cli_rejects_sdf_format() -> None:
    try:
        main(["view-html", "molecules/diborane.sdf", "--format", "sdf"])
    except SystemExit as error:
        assert error.code != 0
    else:
        raise AssertionError("view-html should reject SDF input")


def test_view_html_cli_rejects_sdf_path_without_format() -> None:
    try:
        main(["view-html", "molecules/diborane.sdf"])
    except ValueError as error:
        assert "MolADT JSON" in str(error)
    else:
        raise AssertionError("view-html should reject SDF input")


def test_view_examples_cli_uses_builtin_adt_examples_with_bonding_systems(tmp_path: Path, capsys) -> None:
    html_path = tmp_path / "examples.viewer.html"

    result = main(["view-examples", "--output", str(html_path)])
    output = capsys.readouterr().out
    html = html_path.read_text(encoding="utf-8")

    assert result == 0
    assert str(html_path) in output
    assert f"Viewer URL: {molecule_viewer_uri(html_path)}" in output
    assert "moladt-viewer-collection-v1" in html
    assert "bridge_h3_3c2e" in html
    assert "fe_cp_coordination" in html
    assert "phenyl_pi_ring" in html


def test_view_html_cli_can_open_written_viewer(tmp_path: Path, capsys, monkeypatch) -> None:
    opened: list[Path] = []
    json_path = tmp_path / "diborane.moladt.json"
    html_path = tmp_path / "diborane.viewer.html"
    json_path.write_text(molecule_to_json(diborane_pretty), encoding="utf-8")

    monkeypatch.setattr(cli, "open_molecule_viewer", lambda path: opened.append(Path(path)) or True)

    result = main(
        [
            "view-html",
            str(json_path),
            "--output",
            str(html_path),
            "--open-viewer",
        ]
    )
    output = capsys.readouterr().out

    assert result == 0
    assert opened == [html_path]
    assert "Opened viewer:" in output
    assert molecule_viewer_uri(html_path) in output


def test_view_html_cli_reports_url_when_auto_open_fails(tmp_path: Path, capsys, monkeypatch) -> None:
    json_path = tmp_path / "diborane.moladt.json"
    html_path = tmp_path / "diborane viewer.html"
    json_path.write_text(molecule_to_json(diborane_pretty), encoding="utf-8")

    monkeypatch.setattr(cli, "open_molecule_viewer", lambda path: False)

    result = main(
        [
            "view-html",
            str(json_path),
            "--output",
            str(html_path),
            "--open-viewer",
        ]
    )
    output = capsys.readouterr().out

    assert result == 0
    assert "Viewer open request failed" in output
    assert "diborane%20viewer.html" in output
    assert molecule_viewer_uri(html_path) in output


def test_pretty_example_cli_can_write_and_open_viewer(tmp_path: Path, capsys, monkeypatch) -> None:
    opened: list[Path] = []
    html_path = tmp_path / "ferrocene.viewer.html"

    monkeypatch.setattr(cli, "open_molecule_viewer", lambda path: opened.append(Path(path)) or True)

    result = main(
        [
            "pretty-example",
            "ferrocene",
            "--viewer-output",
            str(html_path),
            "--open-viewer",
        ]
    )
    output = capsys.readouterr().out

    assert result == 0
    assert "Ferrocene (Fe(C5H5)2)" in output
    assert "Viewer:" in output
    assert f"Viewer URL: {molecule_viewer_uri(html_path)}" in output
    assert opened == [html_path]
    assert "fe_cp_coordination" in html_path.read_text(encoding="utf-8")
