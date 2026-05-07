from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from moladt.cli import main
import moladt.cli as cli
from moladt.examples import benzene_pretty, diborane_pretty, ferrocene_pretty, morphine_pretty, sodium_chloride
from moladt.io import molecule_to_json, parse_smiles
from moladt.viewer import (
    molecule_viewer_collection_html,
    molecule_viewer_collection_payload,
    molecule_viewer_html,
    molecule_viewer_payload,
    molecule_viewer_uri,
    open_molecule_viewer,
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


def _assert_viewer_script_executes(html_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for viewer script execution smoke checks")
    supports_modern_js = subprocess.run(
        [node, "-e", "new Function('return null ?? 1')()"],
        capture_output=True,
        text=True,
    )
    if supports_modern_js.returncode != 0:
        pytest.skip("node is too old for the viewer script syntax")
    script = r"""
const fs = require("fs");
const html = fs.readFileSync(process.argv[1], "utf8");
const payloadMatch = html.match(/<script id="moladt-payload" type="application\/json">([\s\S]*?)<\/script>/);
const scripts = [...html.matchAll(/<script(?![^>]*application\/json)[^>]*>([\s\S]*?)<\/script>/g)].map((match) => match[1]);
if (!payloadMatch || !scripts.length) throw new Error("missing viewer payload or script");

const elements = new Map();
function classList() {
  return { add() {}, remove() {}, toggle() {} };
}
function makeElement(id) {
  return {
    id,
    style: {},
    children: [],
    classList: classList(),
    _queries: {},
    textContent: "",
    innerHTML: "",
    type: "",
    className: "",
    append(...nodes) { this.children.push(...nodes); },
    appendChild(node) { this.children.push(node); },
    setAttribute() {},
    addEventListener() {},
    setPointerCapture() {},
    releasePointerCapture() {},
    getBoundingClientRect() { return { width: 900, height: 640, left: 0, top: 0 }; },
    get clientWidth() { return 900; },
    get clientHeight() { return 640; },
    querySelector(selector) {
      if (!this._queries[selector]) this._queries[selector] = makeElement(`${id}:${selector}`);
      return this._queries[selector];
    }
  };
}
const ctx = new Proxy({}, {
  get(_target, prop) {
    if (prop === "createLinearGradient" || prop === "createRadialGradient") return () => ({ addColorStop() {} });
    if (prop === "measureText") return (text) => ({ width: String(text || "").length * 7 });
    return () => {};
  },
  set() { return true; }
});
const canvas = makeElement("molecule-canvas");
canvas.getContext = () => ctx;
global.document = {
  body: makeElement("body"),
  createElement: (tag) => makeElement(tag),
  getElementById(id) {
    if (id === "moladt-payload") return { textContent: payloadMatch[1] };
    if (id === "molecule-canvas") return canvas;
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  }
};
global.window = { devicePixelRatio: 1, addEventListener() {} };
new Function(scripts[scripts.length - 1])();
const systemList = elements.get("system-list");
const labels = (systemList ? systemList.children : [])
  .map((child) => child._queries && child._queries[".row-title"] ? child._queries[".row-title"].textContent : child.textContent)
  .filter(Boolean);
if (!labels.some((label) => /single covalent/.test(label))) {
  throw new Error(`ordinary covalent systems should be labelled in the side panel: ${labels.join(", ")}`);
}
if (!labels.some((label) => /delocalised bonding/.test(label))) {
  throw new Error(`delocalised systems should be labelled in the side panel: ${labels.join(", ")}`);
}
"""
    subprocess.run([node, "-e", script, str(html_path)], check=True, capture_output=True, text=True)


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
    assert "function ionicAtomIdSet" in html
    assert "ionicAtomIds.has(point.atom.id)" in html
    assert "const strength = Math.min(5, Math.abs(charge));" in html
    assert "const radiusFactor = ionic ? 10.5 + strength * 3.0 : 5.2 + strength * 2.2;" in html
    assert "Math.max(120, point.radius * radiusFactor)" in html
    assert "hexToRgba(color, centerAlpha)" in html
    assert "drawBondLines" in html
    assert "chargeGradientForEdge" in html
    assert '"kind":"ionic"' in html
    assert "ctx.setLineDash(options.dash)" in html


def test_molecule_viewer_script_executes_for_ferrocene_and_sodium_chloride(tmp_path: Path) -> None:
    html_path = tmp_path / "viewer.html"
    html_path.write_text(
        molecule_viewer_collection_html(
            (
                ("Ferrocene", ferrocene_pretty),
                ("Sodium chloride", sodium_chloride),
            ),
            title="Viewer smoke",
        ),
        encoding="utf-8",
    )

    _assert_viewer_script_executes(html_path)


def test_molecule_viewer_projects_atom_radius_before_charge_field() -> None:
    html = molecule_viewer_html(ferrocene_pretty, title="Ferrocene")

    assert "const radius = Math.max(8, Number(atom.radius || 0.82) * 13 * perspective * state.zoom);" in html
    assert "radius,\n        atom" in html
    assert "const radius = point.radius;" in html


def test_molecule_viewer_selection_keeps_system_lookup_inside_function() -> None:
    html = molecule_viewer_html(ferrocene_pretty, title="Ferrocene")

    assert "return;\n      }\n      const systems = displaySystems" in html
    assert "return;\n      }\n      }\n      const systems" not in html


def test_molecule_viewer_payload_lengths_come_from_3d_coordinates() -> None:
    payload = molecule_viewer_payload(ferrocene_pretty, title="Ferrocene")
    sigma_edges = {tuple(sorted((edge["a"], edge["b"]))): edge for edge in payload["bonds"]}
    cp1 = next(system for system in payload["systems"] if system["tag"] == "cp1_pi")
    cp1_edges = {
        tuple(sorted((edge["a"], edge["b"]))): edge
        for edge in cp1["edges"]
    }

    assert sigma_edges[(2, 3)]["length"] == pytest.approx(_payload_distance(payload, 2, 3))
    assert sigma_edges[(2, 3)]["length"] == pytest.approx(1.404, abs=0.002)
    assert cp1_edges[(1, 2)]["length"] == pytest.approx(_payload_distance(payload, 1, 2))
    assert cp1_edges[(1, 2)]["length"] == pytest.approx(2.046, abs=0.002)


def test_molecule_viewer_payload_expands_ferrocene_cp_systems_through_iron() -> None:
    payload = molecule_viewer_payload(ferrocene_pretty, title="Ferrocene")
    cp1 = next(system for system in payload["systems"] if system["tag"] == "cp1_pi")
    cp2 = next(system for system in payload["systems"] if system["tag"] == "cp2_pi")

    assert cp1["label"] == "#1 delocalised bonding"
    assert cp2["label"] == "#2 delocalised bonding"
    assert cp1["kind"] == "delocalised"
    assert cp2["kind"] == "delocalised"
    assert 1 in cp1["atoms"]
    assert 1 in cp2["atoms"]
    assert {tuple(sorted((edge["a"], edge["b"]))) for edge in cp1["edges"]} >= {
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
    }
    assert {tuple(sorted((edge["a"], edge["b"]))) for edge in cp2["edges"]} >= {
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 10),
        (1, 11),
    }


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


def test_molecule_viewer_payload_shows_morphine_alkene_as_double_covalent() -> None:
    payload = molecule_viewer_payload(morphine_pretty, title="Morphine")

    alkene_system = next(system for system in payload["systems"] if system["label"].endswith("double covalent"))
    alkene_bond = next(bond for bond in payload["bonds"] if {bond["a"], bond["b"]} == {5, 6})

    assert alkene_system["id"] == 8
    assert alkene_system["kind"] == "covalent"
    assert alkene_system["sharedElectrons"] == 4
    assert [tuple(sorted((edge["a"], edge["b"]))) for edge in alkene_system["edges"]] == [(5, 6)]
    assert alkene_bond["order"] == 2.0


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
    assert "const lanes = systemEdgeLaneMap(molecule.systems);" in html
    assert "if (!selected && !overlapping) return;" in html
    assert "offset: laneOffset" in html
    assert "alpha: active ? 1 : 0.18" in html
    assert "dash: overlapping ? [7, 6] : null" in html
    assert "dash: active ? [10, 7] : [8, 8]" in html
    assert "forEach((system) => drawSystemLabel(system, points))" not in html


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


def test_molecule_viewer_uri_uses_windows_file_url_for_wsl_drive_paths() -> None:
    uri = molecule_viewer_uri(
        "/mnt/c/Users/samma/Documents/MolADT/MolADT-Bayes-Python/results/viewer/examples viewer.html"
    )

    assert (
        uri
        == "file:///C:/Users/samma/Documents/MolADT/MolADT-Bayes-Python/results/viewer/examples%20viewer.html"
    )


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
    assert "cp1_pi" in html
    assert "delocalised bonding" in html
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
    assert "\nOpen this URL manually: " in output
    assert "diborane%20viewer.html" in output
    assert molecule_viewer_uri(html_path) in output


def test_open_molecule_viewer_returns_false_for_unavailable_override(tmp_path: Path, monkeypatch) -> None:
    html_path = tmp_path / "viewer with space.html"
    html_path.write_text("<!doctype html>", encoding="utf-8")
    monkeypatch.setenv("MOLADT_VIEWER_OPENER", "definitely-not-a-viewer-opener")

    assert open_molecule_viewer(html_path) is False


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
    html = html_path.read_text(encoding="utf-8")
    assert "cp1_pi" in html
    assert "delocalised bonding" in html
