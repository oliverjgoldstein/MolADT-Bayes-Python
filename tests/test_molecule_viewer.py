from __future__ import annotations

import json
from pathlib import Path

from moladt.cli import main
import moladt.cli as cli
from moladt.examples import diborane_pretty, ferrocene_pretty
from moladt.io import molecule_to_json
from moladt.viewer import molecule_viewer_html, molecule_viewer_payload, write_molecule_viewer_html


def test_molecule_viewer_payload_includes_bonding_system_annotations() -> None:
    payload = molecule_viewer_payload(diborane_pretty, title="Diborane")

    assert payload["format"] == "moladt-viewer-v1"
    assert payload["title"] == "Diborane"
    assert len(payload["atoms"]) == 8
    assert len(payload["bonds"]) == 5
    assert [system["tag"] for system in payload["systems"]] == ["bridge_h3_3c2e", "bridge_h4_3c2e"]
    assert payload["systems"][0]["edges"] == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_molecule_viewer_payload_keeps_overlapping_system_colours_distinct() -> None:
    payload = molecule_viewer_payload(ferrocene_pretty, title="Ferrocene")

    edge_to_colours: dict[tuple[int, int], set[str]] = {}
    for system in payload["systems"]:
        for edge in system["edges"]:
            key = tuple(sorted((edge["a"], edge["b"])))
            edge_to_colours.setdefault(key, set()).add(system["color"])

    assert edge_to_colours[(1, 7)] == {payload["systems"][1]["color"], payload["systems"][2]["color"]}
    assert payload["systems"][1]["color"] != payload["systems"][2]["color"]


def test_molecule_viewer_html_offsets_overlapping_system_edges() -> None:
    html = molecule_viewer_html(ferrocene_pretty, title="Ferrocene")

    assert "function systemEdgeLaneMap" in html
    assert "offset: laneOffset" in html
    assert "alpha: active ? 1 : 0.18" in html


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
    assert "bridge_h4_3c2e" in html_path.read_text(encoding="utf-8")


def test_view_html_cli_can_open_written_viewer(tmp_path: Path, capsys, monkeypatch) -> None:
    opened: list[Path] = []
    html_path = tmp_path / "diborane.viewer.html"

    monkeypatch.setattr(cli, "open_molecule_viewer", lambda path: opened.append(Path(path)) or True)

    result = main(
        [
            "view-html",
            "molecules/diborane.sdf",
            "--output",
            str(html_path),
            "--open-viewer",
        ]
    )
    output = capsys.readouterr().out

    assert result == 0
    assert opened == [html_path]
    assert "Opened viewer:" in output


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
    assert opened == [html_path]
    assert "fe_backdonation" in html_path.read_text(encoding="utf-8")
