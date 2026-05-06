from __future__ import annotations

import json
import math
import webbrowser
from html import escape
from pathlib import Path
from typing import Any, Sequence

from .chem.dietz import AtomId, BondingSystem, Edge
from .chem.molecule import Molecule, molecule_edges
from .chem.molecule_ops import effective_order


_COVALENT_BOND_COLOR = "#374151"
_IONIC_SYSTEM_COLOR = "#0f766e"

_EXTRA_SYSTEM_COLORS = (
    "#0891b2",
    "#db2777",
    "#ca8a04",
    "#475569",
    "#0d9488",
    "#4f46e5",
    "#a16207",
    "#be185d",
)

_ELEMENT_STYLES: dict[str, dict[str, object]] = {
    "H": {"color": "#f8fafc", "edge": "#9aa6b2", "radius": 0.31},
    "C": {"color": "#303640", "edge": "#12151a", "radius": 0.76},
    "N": {"color": "#3d6fd8", "edge": "#244998", "radius": 0.71},
    "O": {"color": "#d94a42", "edge": "#9e2722", "radius": 0.66},
    "F": {"color": "#3aa66a", "edge": "#247143", "radius": 0.57},
    "Cl": {"color": "#79b84a", "edge": "#4a7d27", "radius": 1.02},
    "Br": {"color": "#a85d35", "edge": "#73391f", "radius": 1.20},
    "I": {"color": "#7d4fa3", "edge": "#53306f", "radius": 1.39},
    "B": {"color": "#f0a46d", "edge": "#b76b35", "radius": 0.84},
    "S": {"color": "#e1ba2f", "edge": "#9a7c18", "radius": 1.05},
    "P": {"color": "#d67f30", "edge": "#96541a", "radius": 1.07},
    "Si": {"color": "#9c7fbd", "edge": "#69528b", "radius": 1.11},
    "Fe": {"color": "#d27845", "edge": "#944521", "radius": 1.32},
    "Na": {"color": "#7b8de8", "edge": "#4d5ca8", "radius": 1.66},
}

_DEFAULT_ELEMENT_STYLE: dict[str, object] = {"color": "#7b8794", "edge": "#485260", "radius": 0.82}


def molecule_viewer_payload(molecule: Molecule, *, title: str = "MolADT 3D Viewer") -> dict[str, Any]:
    """Return the compact JSON payload used by the browser molecule viewer."""

    ionic_edges = _ionic_edge_keys(molecule)
    atoms = [
        {
            "id": atom_id.value,
            "symbol": atom.attributes.symbol.value,
            "label": f"{atom.attributes.symbol.value}#{atom_id.value}",
            "x": atom.coordinate.x.value,
            "y": atom.coordinate.y.value,
            "z": atom.coordinate.z.value,
            "charge": atom.formal_charge,
            "shells": _shells_payload(atom.shells or ()),
            **_element_style(atom.attributes.symbol.value),
        }
        for atom_id, atom in sorted(molecule.atoms.items(), key=lambda item: item[0].value)
    ]
    bonds = [
        {
            "a": edge.a.value,
            "b": edge.b.value,
            "order": round(effective_order(molecule, edge), 3),
            "kind": "ionic" if _edge_key(edge) in ionic_edges else "sigma",
            "length": _edge_length(molecule, edge),
        }
        for edge in sorted(molecule_edges(molecule))
    ]
    systems = [
        {
            "id": system_id.value,
            "label": _system_label(system_id.value, system),
            "tag": system.tag,
            "sharedElectrons": system.shared_electrons.value,
            "color": _system_color(system_id.value, system),
            "atoms": [atom_id.value for atom_id in sorted(system.member_atoms)],
            "edges": [_edge_payload(edge, molecule=molecule) for edge in sorted(system.member_edges)],
        }
        for system_id, system in molecule.systems
    ]
    return {
        "format": "moladt-viewer-v1",
        "title": title,
        "atoms": atoms,
        "bonds": bonds,
        "systems": systems,
        "angles": _angle_payloads(molecule),
    }


def _system_label(system_id: int, system: BondingSystem) -> str:
    display = _system_display_label(system)
    return f"#{system_id} {display}" if display else f"#{system_id}"


def _system_display_label(system: BondingSystem) -> str | None:
    if system.tag == "ionic" and len(system.member_edges) == 1 and system.shared_electrons.value == 0:
        return "ionic"
    if system.tag not in {None, "single", "double", "triple", "quadruple"}:
        return system.tag
    if len(system.member_edges) == 1:
        if system.shared_electrons.value == 2:
            return "single covalent"
        if system.shared_electrons.value == 4:
            return "double covalent"
        if system.shared_electrons.value == 6:
            return "triple covalent"
        if system.shared_electrons.value == 8:
            return "quadruple covalent"
    return system.tag


def _system_color(system_id: int, system: BondingSystem) -> str:
    standard_key = _standard_system_key(system)
    if standard_key == "ionic":
        return _IONIC_SYSTEM_COLOR
    if standard_key is not None:
        return _COVALENT_BOND_COLOR
    return _EXTRA_SYSTEM_COLORS[(system_id - 1) % len(_EXTRA_SYSTEM_COLORS)]


def _standard_system_key(system: BondingSystem) -> str | None:
    if _is_ionic_system(system):
        return "ionic"
    if system.tag not in {None, "single", "double", "triple", "quadruple"}:
        return None
    if len(system.member_edges) != 1:
        return None
    shared_electrons = system.shared_electrons.value
    if shared_electrons == 2:
        return "single"
    if shared_electrons == 4:
        return "double"
    if shared_electrons == 6:
        return "triple"
    if shared_electrons == 8:
        return "quadruple"
    return None


def _ionic_edge_keys(molecule: Molecule) -> frozenset[tuple[int, int]]:
    return frozenset(
        _edge_key(edge)
        for _, system in molecule.systems
        if _is_ionic_system(system)
        for edge in system.member_edges
    )


def _is_ionic_system(system: BondingSystem) -> bool:
    return system.tag == "ionic" and len(system.member_edges) == 1 and system.shared_electrons.value == 0


def _edge_key(edge: Edge) -> tuple[int, int]:
    left = edge.a.value
    right = edge.b.value
    return (left, right) if left <= right else (right, left)


def molecule_viewer_html(molecule: Molecule, *, title: str = "MolADT 3D Viewer") -> str:
    """Render a standalone interactive 3D HTML viewer for a MolADT molecule."""

    payload = molecule_viewer_payload(molecule, title=title)
    return _viewer_html(payload, title=title)


def molecule_viewer_collection_payload(
    molecules: Sequence[tuple[str, Molecule]],
    *,
    title: str = "MolADT Molecule Viewer",
) -> dict[str, Any]:
    """Return a compact JSON payload for a single viewer containing many molecules."""

    return {
        "format": "moladt-viewer-collection-v1",
        "title": title,
        "molecules": [
            molecule_viewer_payload(molecule, title=molecule_title)
            for molecule_title, molecule in molecules
        ],
    }


def molecule_viewer_collection_html(
    molecules: Sequence[tuple[str, Molecule]],
    *,
    title: str = "MolADT Molecule Viewer",
) -> str:
    """Render a standalone viewer page that can switch between multiple molecules."""

    payload = molecule_viewer_collection_payload(molecules, title=title)
    return _viewer_html(payload, title=title)


def _viewer_html(payload: dict[str, Any], *, title: str) -> str:
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).replace("</", "<\\/")
    return (
        _HTML_TEMPLATE.replace("__DOCUMENT_TITLE__", escape(title))
        .replace("__PAYLOAD_JSON__", payload_json)
    )


def write_molecule_viewer_html(
    molecule: Molecule,
    path: str | Path,
    *,
    title: str = "MolADT 3D Viewer",
) -> Path:
    """Write a standalone interactive 3D HTML viewer for a MolADT molecule."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(molecule_viewer_html(molecule, title=title), encoding="utf-8")
    return output_path


def write_molecule_viewer_collection_html(
    molecules: Sequence[tuple[str, Molecule]],
    path: str | Path,
    *,
    title: str = "MolADT Molecule Viewer",
) -> Path:
    """Write one standalone HTML viewer containing multiple molecules."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(molecule_viewer_collection_html(molecules, title=title), encoding="utf-8")
    return output_path


def open_molecule_viewer(path: str | Path) -> bool:
    """Open a written molecule viewer HTML file in the default browser."""

    return webbrowser.open(molecule_viewer_uri(path))


def molecule_viewer_uri(path: str | Path) -> str:
    """Return the portable file URI for a written molecule viewer HTML file."""

    return Path(path).resolve().as_uri()


def _edge_payload(edge: Edge, *, molecule: Molecule | None = None) -> dict[str, int | float]:
    payload: dict[str, int | float] = {"a": edge.a.value, "b": edge.b.value}
    if molecule is not None:
        payload["length"] = _edge_length(molecule, edge)
    return payload


def _edge_length(molecule: Molecule, edge: Edge) -> float:
    return round(_atom_distance(molecule, edge.a, edge.b), 6)


def _atom_distance(molecule: Molecule, atom_a: AtomId, atom_b: AtomId) -> float:
    left = molecule.atoms[atom_a].coordinate
    right = molecule.atoms[atom_b].coordinate
    return math.dist(
        (left.x.value, left.y.value, left.z.value),
        (right.x.value, right.y.value, right.z.value),
    )


def _angle_payloads(molecule: Molecule) -> list[dict[str, int | float]]:
    geometry_edges = set(molecule_edges(molecule))
    neighbors: dict[AtomId, set[AtomId]] = {atom_id: set() for atom_id in molecule.atoms}
    for edge in geometry_edges:
        if edge.a in neighbors and edge.b in neighbors:
            neighbors[edge.a].add(edge.b)
            neighbors[edge.b].add(edge.a)

    payloads: list[dict[str, int | float]] = []
    for center in sorted(neighbors):
        adjacent = sorted(neighbors[center])
        for left_index, left in enumerate(adjacent):
            for right in adjacent[left_index + 1 :]:
                angle = _atom_angle(molecule, left, center, right)
                if angle is not None:
                    payloads.append(
                        {
                            "a": left.value,
                            "center": center.value,
                            "b": right.value,
                            "angle": round(angle, 6),
                        }
                    )
    return payloads


def _atom_angle(molecule: Molecule, atom_a: AtomId, center: AtomId, atom_b: AtomId) -> float | None:
    left = molecule.atoms[atom_a].coordinate
    middle = molecule.atoms[center].coordinate
    right = molecule.atoms[atom_b].coordinate
    vector_a = (
        left.x.value - middle.x.value,
        left.y.value - middle.y.value,
        left.z.value - middle.z.value,
    )
    vector_b = (
        right.x.value - middle.x.value,
        right.y.value - middle.y.value,
        right.z.value - middle.z.value,
    )
    length_a = math.sqrt(sum(component * component for component in vector_a))
    length_b = math.sqrt(sum(component * component for component in vector_b))
    if length_a <= 1e-12 or length_b <= 1e-12:
        return None
    dot = sum(component_a * component_b for component_a, component_b in zip(vector_a, vector_b, strict=True))
    cosine = max(-1.0, min(1.0, dot / (length_a * length_b)))
    return math.degrees(math.acos(cosine))


def _shells_payload(shells: Any) -> list[dict[str, Any]]:
    return [_shell_payload(shell) for shell in shells]


def _shell_payload(shell: Any) -> dict[str, Any]:
    orbitals: list[dict[str, Any]] = []
    for kind, subshell in (
        ("s", shell.s_subshell),
        ("p", shell.p_subshell),
        ("d", shell.d_subshell),
        ("f", shell.f_subshell),
    ):
        if subshell is not None:
            orbitals.extend(_orbital_payload(kind, orbital) for orbital in subshell.orbitals)
    return {
        "n": shell.principal_quantum_number,
        "orbitals": orbitals,
    }


def _orbital_payload(kind: str, orbital: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "orbital": orbital.orbital_type.value,
        "electrons": orbital.electron_count,
        "orientation": None if orbital.orientation is None else _coordinate_payload(orbital.orientation),
        "hybridComponents": None
        if orbital.hybrid_components is None
        else [
            {
                "weight": weight,
                "orbital": getattr(pure.orbital, "value", str(pure.orbital)),
            }
            for weight, pure in orbital.hybrid_components
        ],
    }


def _coordinate_payload(coordinate: Any) -> dict[str, float]:
    return {
        "x": coordinate.x.value,
        "y": coordinate.y.value,
        "z": coordinate.z.value,
    }


def _element_style(symbol: str) -> dict[str, object]:
    return dict(_ELEMENT_STYLES.get(symbol, _DEFAULT_ELEMENT_STYLE))


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__DOCUMENT_TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f6f5;
      --ink: #1f2933;
      --muted: #687482;
      --panel: #ffffff;
      --line: #d8dfdf;
      --accent: #008f87;
      --accent-2: #f05a3f;
      --shadow: 0 18px 44px rgba(31, 41, 51, 0.14);
    }

    * {
      box-sizing: border-box;
    }

    html,
    body {
      margin: 0;
      min-height: 100%;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    body {
      min-height: 100vh;
      overflow: hidden;
    }

    .viewer-shell {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: 100vh;
      width: 100vw;
    }

    .stage {
      position: relative;
      min-width: 0;
      min-height: 100vh;
      background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(232, 245, 242, 0.72)),
        #f8faf8;
    }

    #molecule-canvas {
      display: block;
      width: 100%;
      height: 100vh;
      cursor: grab;
      touch-action: none;
    }

    #molecule-canvas:active {
      cursor: grabbing;
    }

    .toolbar {
      position: absolute;
      top: 18px;
      left: 18px;
      display: flex;
      gap: 8px;
      align-items: center;
      padding: 8px;
      border: 1px solid rgba(216, 223, 223, 0.86);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.84);
      box-shadow: 0 10px 28px rgba(31, 41, 51, 0.10);
      backdrop-filter: blur(14px);
    }

    button {
      min-height: 34px;
      border: 1px solid #cdd6d6;
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      font: inherit;
      font-weight: 700;
      padding: 6px 10px;
      cursor: pointer;
    }

    button:hover,
    button.active {
      border-color: var(--accent);
      color: #006b65;
    }

    .drop-overlay {
      position: absolute;
      inset: 18px;
      display: none;
      place-items: center;
      border: 2px dashed var(--accent);
      border-radius: 8px;
      background: rgba(243, 250, 248, 0.88);
      color: #006b65;
      font-size: clamp(18px, 3vw, 36px);
      font-weight: 800;
      pointer-events: none;
    }

    body.dragging .drop-overlay {
      display: grid;
    }

    .inspector {
      min-width: 0;
      max-height: 100vh;
      overflow: auto;
      border-left: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .inspector-inner {
      display: grid;
      gap: 20px;
      padding: 24px;
    }

    h1,
    h2,
    p {
      margin: 0;
    }

    h1 {
      font-size: 22px;
      line-height: 1.12;
      letter-spacing: 0;
    }

    h2 {
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0;
      color: var(--muted);
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }

    .metric {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfc;
    }

    .metric strong,
    .metric span {
      display: block;
    }

    .metric strong {
      font-size: 18px;
      line-height: 1.05;
    }

    .metric span {
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
    }

    .molecule-list,
    .system-list,
    .atom-list {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }

    .molecule-list {
      max-height: 210px;
      overflow-y: auto;
      padding-right: 3px;
    }

    .molecule-row,
    .system-row {
      display: grid;
      grid-template-columns: 10px minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      padding: 10px;
      text-align: left;
    }

    .molecule-row {
      grid-template-columns: 24px minmax(0, 1fr) auto;
    }

    .molecule-row.active,
    .system-row.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(0, 143, 135, 0.13);
    }

    .index-badge {
      display: inline-grid;
      place-items: center;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: #eef6f4;
      color: #006b65;
      font-size: 12px;
      font-weight: 900;
    }

    .swatch {
      width: 10px;
      height: 34px;
      border-radius: 999px;
    }

    .row-title {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 800;
    }

    .row-meta {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
    }

    .pill {
      border-radius: 999px;
      background: #eef6f4;
      color: #006b65;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 800;
    }

    .atom-row {
      display: grid;
      grid-template-columns: 12px minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      border-bottom: 1px solid #edf0f0;
      padding: 7px 0;
      color: var(--muted);
      cursor: pointer;
    }

    .atom-row.active {
      color: var(--ink);
    }

    .atom-dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      border: 1px solid rgba(31, 41, 51, 0.28);
    }

    .selection {
      min-height: 68px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfc;
      padding: 12px;
      color: var(--muted);
    }

    .selection strong,
    .geometry-block strong {
      color: var(--ink);
    }

    .geometry-block {
      display: grid;
      gap: 4px;
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid #edf0f0;
    }

    .geometry-list {
      display: grid;
      gap: 2px;
      font-size: 12px;
      line-height: 1.35;
    }

    .tooltip {
      position: fixed;
      z-index: 4;
      display: none;
      pointer-events: none;
      border: 1px solid rgba(31, 41, 51, 0.18);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.94);
      box-shadow: 0 12px 30px rgba(31, 41, 51, 0.16);
      padding: 8px 10px;
      color: var(--ink);
      font-size: 12px;
      font-weight: 800;
    }

    @media (max-width: 860px) {
      body {
        overflow: auto;
      }

      .viewer-shell {
        grid-template-columns: 1fr;
      }

      .stage,
      #molecule-canvas {
        min-height: 66vh;
        height: 66vh;
      }

      .inspector {
        max-height: none;
        border-left: 0;
        border-top: 1px solid var(--line);
      }

      .toolbar {
        top: 10px;
        left: 10px;
      }
    }
  </style>
</head>
<body>
  <main class="viewer-shell" data-moladt-viewer>
    <section class="stage" aria-label="Molecule viewer">
      <canvas id="molecule-canvas"></canvas>
      <div class="toolbar" aria-label="Viewer controls">
        <button id="reset-view" type="button" title="Reset view">Reset</button>
        <button id="toggle-labels" type="button" title="Toggle atom labels">Labels</button>
        <button id="toggle-systems" type="button" title="Toggle bonding systems">Systems</button>
        <button id="toggle-axes" type="button" title="Toggle coordinate axes">Axes</button>
      </div>
      <div class="drop-overlay">Drop MolADT JSON</div>
    </section>
    <aside class="inspector" aria-label="Molecule details">
      <div class="inspector-inner">
        <header>
          <h1 id="molecule-title"></h1>
        </header>
        <section class="metric-grid" aria-label="Molecule summary">
          <div class="metric"><strong id="atom-count">0</strong><span>atoms</span></div>
          <div class="metric"><strong id="bond-count">0</strong><span>edges</span></div>
          <div class="metric"><strong id="system-count">0</strong><span>systems</span></div>
        </section>
        <section id="molecule-section">
          <h2>Molecules</h2>
          <div id="molecule-list" class="molecule-list"></div>
        </section>
        <section>
          <h2>Bonding Systems</h2>
          <div id="system-list" class="system-list"></div>
        </section>
        <section>
          <h2>Selected</h2>
          <div id="selection" class="selection">None</div>
        </section>
        <section>
          <h2>Atoms</h2>
          <div id="atom-list" class="atom-list"></div>
        </section>
      </div>
    </aside>
  </main>
  <div id="tooltip" class="tooltip"></div>
  <script id="moladt-payload" type="application/json">__PAYLOAD_JSON__</script>
  <script>
    const ELEMENT_STYLES = {
      H: { color: "#f8fafc", edge: "#9aa6b2", radius: 0.31 },
      C: { color: "#303640", edge: "#12151a", radius: 0.76 },
      N: { color: "#3d6fd8", edge: "#244998", radius: 0.71 },
      O: { color: "#d94a42", edge: "#9e2722", radius: 0.66 },
      F: { color: "#3aa66a", edge: "#247143", radius: 0.57 },
      Cl: { color: "#79b84a", edge: "#4a7d27", radius: 1.02 },
      Br: { color: "#a85d35", edge: "#73391f", radius: 1.20 },
      I: { color: "#7d4fa3", edge: "#53306f", radius: 1.39 },
      B: { color: "#f0a46d", edge: "#b76b35", radius: 0.84 },
      S: { color: "#e1ba2f", edge: "#9a7c18", radius: 1.05 },
      P: { color: "#d67f30", edge: "#96541a", radius: 1.07 },
      Si: { color: "#9c7fbd", edge: "#69528b", radius: 1.11 },
      Fe: { color: "#d27845", edge: "#944521", radius: 1.32 },
      Na: { color: "#7b8de8", edge: "#4d5ca8", radius: 1.66 },
      default: { color: "#7b8794", edge: "#485260", radius: 0.82 }
    };

    const COVALENT_BOND_COLOR = "#374151";
    const IONIC_SYSTEM_COLOR = "#0f766e";
    const EXTRA_SYSTEM_COLORS = ["#0891b2", "#db2777", "#ca8a04", "#475569", "#0d9488", "#4f46e5", "#a16207", "#be185d"];
    const POSITIVE_CHARGE_COLOR = "#2563eb";
    const NEGATIVE_CHARGE_COLOR = "#dc2626";
    const canvas = document.getElementById("molecule-canvas");
    const ctx = canvas.getContext("2d");
    const tooltip = document.getElementById("tooltip");
    const state = {
      rx: -0.62,
      ry: 0.74,
      zoom: 1,
      labels: true,
      systems: true,
      axes: true,
      activeSystem: null,
      selectedAtom: null,
      pointerDown: false,
      pointerMoved: false,
      lastX: 0,
      lastY: 0,
      pointerStartX: 0,
      pointerStartY: 0,
      hoverAtom: null,
      screenAtoms: []
    };

    let viewer = normalizeViewerPayload(JSON.parse(document.getElementById("moladt-payload").textContent));
    let molecule = viewer.molecules[viewer.activeIndex] || emptyMolecule();

    function emptyMolecule() {
      return {
        title: "No molecule",
        atoms: [],
        atomMap: new Map(),
        bonds: [],
        systems: [],
        center: { x: 0, y: 0, z: 0 },
        scale: 1,
        axes: buildAxes({ x: 0, y: 0, z: 0 }, { minX: -1, maxX: 1, minY: -1, maxY: 1, minZ: -1, maxZ: 1 }, 1)
      };
    }

    function edgeKey(edge) {
      return [Number(edge.a), Number(edge.b)].sort((a, b) => a - b).join("-");
    }

    function atomIdValue(value) {
      return typeof value === "object" && value !== null ? Number(value.value) : Number(value);
    }

    function edgeFromMoladt(edge) {
      return { a: atomIdValue(edge.a), b: atomIdValue(edge.b) };
    }

    function coordinateValue(value) {
      return typeof value === "object" && value !== null ? Number(value.value) : Number(value || 0);
    }

    function coordinateFromMoladt(coordinate) {
      if (!coordinate) {
        return null;
      }
      return {
        x: coordinateValue(coordinate.x),
        y: coordinateValue(coordinate.y),
        z: coordinateValue(coordinate.z)
      };
    }

    function orbitalFromMoladt(kind, orbital) {
      return {
        kind,
        orbital: String(orbital.orbital_type || orbital.orbital || kind),
        electrons: Number(orbital.electron_count ?? orbital.electrons ?? 0),
        orientation: coordinateFromMoladt(orbital.orientation),
        hybridComponents: orbital.hybrid_components || orbital.hybridComponents || null
      };
    }

    function shellsFromMoladt(shells) {
      return (shells || []).map((shell) => {
        const orbitals = [];
        [
          ["s", shell.s_subshell],
          ["p", shell.p_subshell],
          ["d", shell.d_subshell],
          ["f", shell.f_subshell]
        ].forEach(([kind, subshell]) => {
          if (subshell && subshell.orbitals) {
            subshell.orbitals.forEach((orbital) => orbitals.push(orbitalFromMoladt(kind, orbital)));
          }
        });
        return {
          n: Number(shell.principal_quantum_number || shell.n || 0),
          orbitals
        };
      });
    }

    function normalizeShells(shells) {
      return (shells || []).map((shell) => ({
        n: Number(shell.n || shell.principal_quantum_number || 0),
        orbitals: (shell.orbitals || []).map((orbital) => ({
          kind: String(orbital.kind || "s"),
          orbital: String(orbital.orbital || orbital.orbital_type || orbital.kind || "s"),
          electrons: Number(orbital.electrons ?? orbital.electron_count ?? 0),
          orientation: orbital.orientation ? coordinateFromMoladt(orbital.orientation) : null,
          hybridComponents: orbital.hybridComponents || orbital.hybrid_components || null
        }))
      }));
    }

    function roundedValue(value) {
      return Math.round(Number(value) * 10) / 10;
    }

    function uniqueTicks(values) {
      return Array.from(new Set(values.map(roundedValue))).sort((a, b) => a - b);
    }

    function axisTicks(min, max) {
      const ticks = [min, max];
      if (min < 0 && max > 0) {
        ticks.push(0);
      }
      return uniqueTicks(ticks);
    }

    function axisRange(min, max, fallbackCenter, fallbackSpan) {
      if (!Number.isFinite(min) || !Number.isFinite(max)) {
        return { min: fallbackCenter - fallbackSpan, max: fallbackCenter + fallbackSpan };
      }
      const span = Math.max(0.6, max - min, fallbackSpan * 0.5);
      const pad = Math.max(0.35, span * 0.16);
      if (max - min < 0.6) {
        return { min: fallbackCenter - span / 2 - pad, max: fallbackCenter + span / 2 + pad };
      }
      return { min: min - pad, max: max + pad };
    }

    function buildAxes(center, bounds, maxSpan) {
      const span = Math.max(1, maxSpan);
      const xRange = axisRange(bounds.minX, bounds.maxX, center.x, span);
      const yRange = axisRange(bounds.minY, bounds.maxY, center.y, span);
      const zRange = axisRange(bounds.minZ, bounds.maxZ, center.z, span);
      return [
        {
          key: "x",
          label: "x",
          color: "#ef4444",
          start: { x: xRange.min, y: center.y, z: center.z },
          end: { x: xRange.max, y: center.y, z: center.z },
          ticks: axisTicks(xRange.min, xRange.max).map((value) => ({ value, point: { x: value, y: center.y, z: center.z } }))
        },
        {
          key: "y",
          label: "y",
          color: "#059669",
          start: { x: center.x, y: yRange.min, z: center.z },
          end: { x: center.x, y: yRange.max, z: center.z },
          ticks: axisTicks(yRange.min, yRange.max).map((value) => ({ value, point: { x: center.x, y: value, z: center.z } }))
        },
        {
          key: "z",
          label: "z",
          color: "#2563eb",
          start: { x: center.x, y: center.y, z: zRange.min },
          end: { x: center.x, y: center.y, z: zRange.max },
          ticks: axisTicks(zRange.min, zRange.max).map((value) => ({ value, point: { x: center.x, y: center.y, z: value } }))
        }
      ];
    }

    function fromMoladtJson(raw) {
      const systems = (raw.systems || []).map((item, index) => {
        const system = item.bonding_system;
        const id = atomIdValue(item.system_id);
        const sharedElectrons = atomIdValue(system.shared_electrons);
        const edges = (system.member_edges || []).map(edgeFromMoladt);
        return {
          id,
          label: systemLabel(id, system.tag || null, sharedElectrons, edges),
          tag: system.tag || null,
          sharedElectrons,
          color: systemColor(id, system.tag || null, sharedElectrons, edges, index),
          atoms: (system.member_atoms || []).map(atomIdValue),
          edges
        };
      });
      const atoms = (raw.atoms || []).map((item) => {
        const atom = item.atom;
        const id = atomIdValue(item.atom_id);
        const symbol = atom.attributes.symbol;
        const style = ELEMENT_STYLES[symbol] || ELEMENT_STYLES.default;
        return {
          id,
          symbol,
          label: symbol + "#" + id,
          x: Number(atom.coordinate.x.value),
          y: Number(atom.coordinate.y.value),
          z: Number(atom.coordinate.z.value),
          charge: Number(atom.formal_charge || 0),
          shells: shellsFromMoladt(atom.shells || []),
          color: style.color,
          edge: style.edge,
          radius: style.radius
        };
      });
      const systemContrib = new Map();
      systems.forEach((system) => {
        const perEdge = system.edges.length ? system.sharedElectrons / (2 * system.edges.length) : 0;
        system.edges.forEach((edge) => {
          systemContrib.set(edgeKey(edge), (systemContrib.get(edgeKey(edge)) || 0) + perEdge);
        });
      });
      const edgeMap = new Map();
      systems.forEach((system) => {
        system.edges.forEach((edge) => edgeMap.set(edgeKey(edge), edge));
      });
      const ionicEdgeKeys = ionicEdgeKeySet(systems);
      const bonds = Array.from(edgeMap.values()).map((edge) => ({
        ...edge,
        order: systemContrib.get(edgeKey(edge)) || 0,
        kind: ionicEdgeKeys.has(edgeKey(edge)) ? "ionic" : "sigma",
        length: null
      }));
      return {
        format: "moladt-viewer-v1",
        title: raw.title || "Dropped MolADT",
        atoms,
        bonds,
        systems,
        angles: []
      };
    }

    function normalizeViewerPayload(raw) {
      if (raw && raw.format === "moladt-viewer-collection-v1") {
        const molecules = (raw.molecules || []).map(normalizePayload);
        return {
          title: raw.title || "MolADT Molecule Viewer",
          activeIndex: 0,
          molecules
        };
      }
      const single = normalizePayload(raw);
      return {
        title: single.title,
        activeIndex: 0,
        molecules: [single]
      };
    }

      function normalizePayload(raw) {
      const payload = raw && raw.format === "moladt-viewer-v1" ? raw : fromMoladtJson(raw);
      const atoms = payload.atoms.map((atom) => {
        const style = ELEMENT_STYLES[atom.symbol] || ELEMENT_STYLES.default;
        return {
          ...atom,
          id: Number(atom.id),
          x: Number(atom.x),
          y: Number(atom.y),
          z: Number(atom.z),
          charge: Number(atom.charge || 0),
          shells: normalizeShells(atom.shells || []),
          radius: Number(atom.radius || style.radius),
          color: atom.color || style.color,
          edge: atom.edge || style.edge
        };
      }).sort((a, b) => a.id - b.id);
      const center = atoms.reduce((acc, atom) => {
        acc.x += atom.x;
        acc.y += atom.y;
        acc.z += atom.z;
        return acc;
      }, { x: 0, y: 0, z: 0 });
      if (atoms.length) {
        center.x /= atoms.length;
        center.y /= atoms.length;
        center.z /= atoms.length;
      }
      const bounds = atoms.reduce((acc, atom) => ({
        minX: Math.min(acc.minX, atom.x),
        maxX: Math.max(acc.maxX, atom.x),
        minY: Math.min(acc.minY, atom.y),
        maxY: Math.max(acc.maxY, atom.y),
        minZ: Math.min(acc.minZ, atom.z),
        maxZ: Math.max(acc.maxZ, atom.z)
      }), { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: Infinity, maxZ: -Infinity });
      const maxSpan = Math.max(1, ...atoms.map((atom) => Math.hypot(atom.x - center.x, atom.y - center.y, atom.z - center.z)));
      const scale = 2.9 / maxSpan;
      atoms.forEach((atom) => {
        atom.vx = (atom.x - center.x) * scale;
        atom.vy = (atom.y - center.y) * scale;
        atom.vz = (atom.z - center.z) * scale;
      });
      const atomMap = new Map(atoms.map((atom) => [atom.id, atom]));
      const rawBonds = (payload.bonds || []).map((bond) => ({
        a: Number(bond.a),
        b: Number(bond.b),
        order: Number(bond.order || 1),
        kind: bond.kind || "sigma",
        length: numericOrNull(bond.length)
      })).filter((bond) => atomMap.has(bond.a) && atomMap.has(bond.b)).map((bond) => ({
        ...bond,
        length: bond.length ?? distanceBetweenAtoms(atomMap.get(bond.a), atomMap.get(bond.b))
      }));
      const systems = (payload.systems || []).map((system, index) => ({
        ...system,
        id: Number(system.id),
        label: system.label || systemLabel(Number(system.id), system.tag || null, Number(system.sharedElectrons || 0), system.edges || []),
        sharedElectrons: Number(system.sharedElectrons || 0),
        color: system.color || systemColor(Number(system.id), system.tag || null, Number(system.sharedElectrons || 0), system.edges || [], index),
        atoms: (system.atoms || []).map(Number),
        edges: (system.edges || []).map((edge) => ({
          a: Number(edge.a),
          b: Number(edge.b),
          length: numericOrNull(edge.length)
        })).filter((edge) => atomMap.has(edge.a) && atomMap.has(edge.b)).map((edge) => ({
          ...edge,
          length: edge.length ?? distanceBetweenAtoms(atomMap.get(edge.a), atomMap.get(edge.b))
        }))
      }));
      const ionicEdgeKeys = ionicEdgeKeySet(systems);
      const bonds = rawBonds.map((bond) => ({
        ...bond,
        kind: ionicEdgeKeys.has(edgeKey(bond)) ? "ionic" : bond.kind
      }));
      const payloadAngles = normalizeAngles(payload.angles || [], atomMap);
      const angles = payloadAngles.length ? payloadAngles : buildAnglePayloads(atomMap, bonds, systems);
      return {
        title: payload.title || "MolADT 3D Viewer",
        atoms,
        atomMap,
        bonds,
        systems,
        angles,
        center,
        scale,
          axes: buildAxes(center, bounds, maxSpan)
        };
      }

      function systemLabel(id, tag, sharedElectrons, edges) {
      const covalent = covalentLabel(tag, sharedElectrons, edges);
        const display = covalent || tag;
        return display ? "#" + id + " " + display : "#" + id;
      }

    function covalentLabel(tag, sharedElectrons, edges) {
      if (tag === "ionic" && edges && edges.length === 1 && sharedElectrons === 0) return "ionic";
      if (tag && tag !== "single" && tag !== "double" && tag !== "triple" && tag !== "quadruple") return null;
      if (!edges || edges.length !== 1) return null;
        if (sharedElectrons === 2) return "single covalent";
        if (sharedElectrons === 4) return "double covalent";
        if (sharedElectrons === 6) return "triple covalent";
        if (sharedElectrons === 8) return "quadruple covalent";
      return null;
    }

    function standardSystemKey(tag, sharedElectrons, edges) {
      if (tag === "ionic" && edges && edges.length === 1 && sharedElectrons === 0) return "ionic";
      if (tag && tag !== "single" && tag !== "double" && tag !== "triple" && tag !== "quadruple") return null;
      if (!edges || edges.length !== 1) return null;
      if (sharedElectrons === 2) return "single";
      if (sharedElectrons === 4) return "double";
      if (sharedElectrons === 6) return "triple";
      if (sharedElectrons === 8) return "quadruple";
      return null;
    }

    function systemColor(id, tag, sharedElectrons, edges, index) {
      const key = standardSystemKey(tag, sharedElectrons, edges);
      if (key === "ionic") return IONIC_SYSTEM_COLOR;
      if (key) return COVALENT_BOND_COLOR;
      const offset = Number.isFinite(index) ? index : Number(id || 1) - 1;
      return EXTRA_SYSTEM_COLORS[((offset % EXTRA_SYSTEM_COLORS.length) + EXTRA_SYSTEM_COLORS.length) % EXTRA_SYSTEM_COLORS.length];
    }

    function isStandardCovalentSystem(system) {
      return !!standardCovalentLineCount(system);
    }

    function displaySystems(systems) {
      return (systems || []).filter((system) => !isStandardCovalentSystem(system));
    }

    function standardCovalentLineCount(system) {
      if (!system) return 0;
      const key = standardSystemKey(system.tag || null, Number(system.sharedElectrons || 0), system.edges || []);
      if (key === "single") return 1;
      if (key === "double") return 2;
      if (key === "triple") return 3;
      if (key === "quadruple") return 4;
      return 0;
    }

    function standardSystemByEdge(systems) {
      const byEdge = new Map();
      (systems || []).forEach((system) => {
        const lineCount = standardCovalentLineCount(system);
        if (lineCount) {
          (system.edges || []).forEach((edge) => byEdge.set(edgeKey(edge), system));
        }
      });
      return byEdge;
    }

    function lineCountFromBond(bond, standardByEdge) {
      const standard = standardByEdge.get(edgeKey(bond));
      return standardCovalentLineCount(standard) || Math.max(1, Math.min(4, Math.round(Number(bond.order || 1))));
    }

    function isIonicSystem(system) {
      return system && system.tag === "ionic" && system.sharedElectrons === 0 && system.edges && system.edges.length === 1;
    }

    function ionicEdgeKeySet(systems) {
      const keys = new Set();
      (systems || []).forEach((system) => {
        if (isIonicSystem(system)) {
          system.edges.forEach((edge) => keys.add(edgeKey(edge)));
        }
      });
      return keys;
    }

    function chargeColor(charge) {
      if (Number(charge) > 0) return POSITIVE_CHARGE_COLOR;
      if (Number(charge) < 0) return NEGATIVE_CHARGE_COLOR;
      return "#535b63";
    }

    function chargeGradientForEdge(atomMap, edge) {
      const atomA = atomMap.get(Number(edge.a));
      const atomB = atomMap.get(Number(edge.b));
      return {
        colorA: chargeColor(atomA ? atomA.charge : 0),
        colorB: chargeColor(atomB ? atomB.charge : 0)
      };
    }

      function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function rotatePoint(atom) {
      const cx = Math.cos(state.rx);
      const sx = Math.sin(state.rx);
      const cy = Math.cos(state.ry);
      const sy = Math.sin(state.ry);
      const y1 = atom.vy * cx - atom.vz * sx;
      const z1 = atom.vy * sx + atom.vz * cx;
      const x2 = atom.vx * cy + z1 * sy;
      const z2 = -atom.vx * sy + z1 * cy;
      return { x: x2, y: y1, z: z2 };
    }

    function project(atom) {
      const rect = canvas.getBoundingClientRect();
      const rotated = rotatePoint(atom);
      const distance = 8.5;
      const perspective = distance / (distance - rotated.z);
      const unit = Math.min(rect.width, rect.height) / 8.2;
      const radius = Math.max(8, Number(atom.radius || 0.82) * 13 * perspective * state.zoom);
      return {
        x: rect.width / 2 + rotated.x * unit * state.zoom * perspective,
        y: rect.height / 2 - rotated.y * unit * state.zoom * perspective,
        z: rotated.z,
        p: perspective,
        radius,
        atom
      };
    }

    function projectWorld(point) {
      return project({
        vx: (point.x - molecule.center.x) * molecule.scale,
        vy: (point.y - molecule.center.y) * molecule.scale,
        vz: (point.z - molecule.center.z) * molecule.scale
      });
    }

    function drawCoordinateAxes() {
      if (!state.axes || !molecule.axes || !molecule.axes.length) {
        return;
      }
      ctx.save();
      ctx.font = "800 11px ui-sans-serif, system-ui, sans-serif";
      ctx.textBaseline = "middle";
      molecule.axes.forEach((axis) => {
        const start = projectWorld(axis.start);
        const end = projectWorld(axis.end);
        drawLine(start, end, {
          width: 1.8,
          color: axis.color,
          alpha: 0.66
        });
        ctx.fillStyle = axis.color;
        ctx.textAlign = "center";
        ctx.fillText(axis.label, end.x + 12, end.y - 12);
        (axis.ticks || []).forEach((tick) => {
          const point = projectWorld(tick.point);
          ctx.beginPath();
          ctx.arc(point.x, point.y, 3.2, 0, Math.PI * 2);
          ctx.fillStyle = axis.color;
          ctx.globalAlpha = 0.72;
          ctx.fill();
          ctx.globalAlpha = 1;
          ctx.textAlign = "left";
          ctx.fillStyle = "rgba(31, 41, 51, 0.72)";
          ctx.fillText(axis.label + " " + tick.value + "A", point.x + 6, point.y + 10);
        });
      });
      ctx.restore();
    }

    function drawLine(a, b, options) {
      ctx.save();
      ctx.globalAlpha = options.alpha ?? 1;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = options.width;
      const offset = Number(options.offset || 0);
      let ax = a.x;
      let ay = a.y;
      let bx = b.x;
      let by = b.y;
      if (offset !== 0) {
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const length = Math.max(1, Math.hypot(dx, dy));
        const nx = -dy / length;
        const ny = dx / length;
        ax += nx * offset;
        ay += ny * offset;
        bx += nx * offset;
        by += ny * offset;
      }
      const gradient = ctx.createLinearGradient(ax, ay, bx, by);
      gradient.addColorStop(0, options.colorA || options.color);
      gradient.addColorStop(1, options.colorB || options.color);
      ctx.strokeStyle = gradient;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
      ctx.restore();
    }

    function bondLineOffsets(lineCount, spacing) {
      if (lineCount <= 1) return [0];
      const center = (lineCount - 1) / 2;
      return Array.from({ length: lineCount }, (_, index) => (index - center) * spacing);
    }

    function drawBondLines(a, b, options) {
      const lineCount = Math.max(1, Math.min(4, Math.round(Number(options.lineCount || 1))));
      const spacing = Number(options.spacing || 5);
      bondLineOffsets(lineCount, spacing).forEach((offset) => {
        drawLine(a, b, {
          width: options.width,
          color: options.color,
          colorA: options.colorA || options.color,
          colorB: options.colorB || options.color,
          alpha: options.alpha,
          offset: Number(options.offset || 0) + offset
        });
      });
    }

    function drawChargeField(points) {
      const charged = Array.from(points.values())
        .filter((point) => point.atom && Number(point.atom.charge || 0) !== 0)
        .sort((left, right) => left.z - right.z);
      if (!charged.length) {
        return;
      }
      ctx.save();
      charged.forEach((point) => {
        const charge = Number(point.atom.charge || 0);
        const magnitude = Math.min(3, Math.abs(charge));
        const fieldRadius = Math.max(46, point.radius * (4.3 + magnitude * 0.7));
        const color = chargeColor(charge);
        const gradient = ctx.createRadialGradient(
          point.x,
          point.y,
          Math.max(3, point.radius * 0.7),
          point.x,
          point.y,
          fieldRadius
        );
        gradient.addColorStop(0, hexToRgba(color, 0.30));
        gradient.addColorStop(0.48, hexToRgba(color, 0.13));
        gradient.addColorStop(1, hexToRgba(color, 0));
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(point.x, point.y, fieldRadius, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.restore();
    }

    function systemEdgeLaneMap(systems) {
      const edgeSystems = new Map();
      systems.forEach((system) => {
        system.edges.forEach((edge) => {
          const key = edgeKey(edge);
          if (!edgeSystems.has(key)) {
            edgeSystems.set(key, []);
          }
          edgeSystems.get(key).push(Number(system.id));
        });
      });
      const lanes = new Map();
      edgeSystems.forEach((ids, key) => {
        const uniqueIds = Array.from(new Set(ids)).sort((a, b) => a - b);
        uniqueIds.forEach((id, index) => {
          lanes.set(key + ":" + id, { index, count: uniqueIds.length });
        });
      });
      return lanes;
    }

    function hexToRgba(hex, alpha) {
      const clean = String(hex || "#000000").replace("#", "");
      const value = clean.length === 3
        ? clean.split("").map((part) => part + part).join("")
        : clean.padEnd(6, "0").slice(0, 6);
      const r = parseInt(value.slice(0, 2), 16);
      const g = parseInt(value.slice(2, 4), 16);
      const b = parseInt(value.slice(4, 6), 16);
      return "rgba(" + r + ", " + g + ", " + b + ", " + alpha + ")";
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function formatMeasure(value, digits = 3) {
      return Number(value).toFixed(digits).replace(/\\.?0+$/, "");
    }

    function numericOrNull(value) {
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }

    function distanceBetweenAtoms(left, right) {
      return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
    }

    function angleBetweenAtoms(left, center, right) {
      const ax = left.x - center.x;
      const ay = left.y - center.y;
      const az = left.z - center.z;
      const bx = right.x - center.x;
      const by = right.y - center.y;
      const bz = right.z - center.z;
      const alen = Math.hypot(ax, ay, az);
      const blen = Math.hypot(bx, by, bz);
      if (alen < 0.000001 || blen < 0.000001) {
        return null;
      }
      const cosine = Math.max(-1, Math.min(1, (ax * bx + ay * by + az * bz) / (alen * blen)));
      return Math.acos(cosine) * 180 / Math.PI;
    }

    function normalizeAngles(rawAngles, atomMap) {
      return (rawAngles || []).map((item) => ({
        a: Number(item.a ?? item.left),
        center: Number(item.center),
        b: Number(item.b ?? item.right),
        angle: numericOrNull(item.angle)
      })).filter((item) => (
        atomMap.has(item.a)
        && atomMap.has(item.center)
        && atomMap.has(item.b)
        && item.a !== item.center
        && item.b !== item.center
        && item.a !== item.b
        && item.angle !== null
      )).sort((left, right) => (
        left.center - right.center
        || left.a - right.a
        || left.b - right.b
      ));
    }

    function buildAnglePayloads(atomMap, bonds, systems) {
      const neighbors = new Map();
      function ensure(atomId) {
        if (!neighbors.has(atomId)) {
          neighbors.set(atomId, new Set());
        }
        return neighbors.get(atomId);
      }
      function addEdge(edge) {
        if (!atomMap.has(edge.a) || !atomMap.has(edge.b)) {
          return;
        }
        ensure(edge.a).add(edge.b);
        ensure(edge.b).add(edge.a);
      }
      bonds.forEach(addEdge);
      systems.forEach((system) => system.edges.forEach(addEdge));
      const angles = [];
      Array.from(neighbors.keys()).sort((a, b) => a - b).forEach((centerId) => {
        const center = atomMap.get(centerId);
        const adjacent = Array.from(neighbors.get(centerId)).sort((a, b) => a - b);
        for (let i = 0; i < adjacent.length; i += 1) {
          for (let j = i + 1; j < adjacent.length; j += 1) {
            const leftId = adjacent[i];
            const rightId = adjacent[j];
            const angle = angleBetweenAtoms(atomMap.get(leftId), center, atomMap.get(rightId));
            if (angle !== null) {
              angles.push({ a: leftId, center: centerId, b: rightId, angle });
            }
          }
        }
      });
      return angles;
    }

    function geometryEdgesForAtom(atom) {
      const edges = new Map();
      function addGeometryEdge(edge, label) {
        if (Number(edge.a) !== atom.id && Number(edge.b) !== atom.id) {
          return;
        }
        const otherId = Number(edge.a) === atom.id ? Number(edge.b) : Number(edge.a);
        const other = molecule.atomMap.get(otherId);
        if (!other) {
          return;
        }
        const key = edgeKey(edge);
        const edgeLength = numericOrNull(edge.length);
        const existing = edges.get(key) || {
          a: Number(edge.a),
          b: Number(edge.b),
          other,
          length: edgeLength,
          labels: []
        };
        if (existing.length === null && edgeLength !== null) {
          existing.length = edgeLength;
        }
        if (!existing.labels.includes(label)) {
          existing.labels.push(label);
        }
        edges.set(key, existing);
      }
      molecule.bonds.forEach((bond) => {
        addGeometryEdge(bond, "sigma order " + formatMeasure(bond.order || 1, 2));
      });
      molecule.systems.forEach((system) => {
        system.edges.forEach((edge) => addGeometryEdge(edge, system.label));
      });
      return Array.from(edges.values()).sort((left, right) => left.other.id - right.other.id);
    }

    function bondAnglesForAtom(atom) {
      const storedAngles = (molecule.angles || []).filter((item) => item.center === atom.id).map((item) => ({
        left: molecule.atomMap.get(item.a),
        right: molecule.atomMap.get(item.b),
        angle: item.angle
      })).filter((item) => item.left && item.right && item.angle !== null);
      if (storedAngles.length) {
        return storedAngles.sort((left, right) => left.left.id - right.left.id || left.right.id - right.right.id);
      }
      const neighbors = geometryEdgesForAtom(atom).map((edge) => edge.other);
      const angles = [];
      for (let i = 0; i < neighbors.length; i += 1) {
        for (let j = i + 1; j < neighbors.length; j += 1) {
          const angle = angleBetweenAtoms(neighbors[i], atom, neighbors[j]);
          if (angle !== null) {
            angles.push({
              left: neighbors[i],
              right: neighbors[j],
              angle
            });
          }
        }
      }
      return angles.sort((left, right) => left.left.id - right.left.id || left.right.id - right.right.id);
    }

    function geometrySummary(atom) {
      const coordinates = "("
        + formatMeasure(atom.x) + ", "
        + formatMeasure(atom.y) + ", "
        + formatMeasure(atom.z) + ") A";
      const edges = geometryEdgesForAtom(atom);
      const edgeRows = edges.length
        ? edges.map((edge) => {
          const length = edge.length ?? distanceBetweenAtoms(atom, edge.other);
          return escapeHtml(atom.label + " - " + edge.other.label)
            + ": " + escapeHtml(formatMeasure(length)) + " A"
            + " (" + escapeHtml(edge.labels.join(", ")) + ")";
        }).join("<br>")
        : "no geometric edges";
      const angles = bondAnglesForAtom(atom);
      const angleRows = angles.length
        ? angles.map((item) => (
          escapeHtml(item.left.label + " - " + atom.label + " - " + item.right.label)
          + ": " + escapeHtml(formatMeasure(item.angle, 1)) + " deg"
        )).join("<br>")
        : "no bond angles";
      return '<div class="geometry-block"><strong>Coordinates (A)</strong><div class="geometry-list">'
        + escapeHtml(coordinates)
        + '</div></div><div class="geometry-block"><strong>Edge Lengths From 3D Coordinates</strong><div class="geometry-list">'
        + edgeRows
        + '</div></div><div class="geometry-block"><strong>Bond Angles</strong><div class="geometry-list">'
        + angleRows
        + "</div></div>";
    }

    function flattenOrbitals(atom) {
      return (atom.shells || []).flatMap((shell) => (
        (shell.orbitals || []).map((orbital, index) => ({
          ...orbital,
          n: shell.n,
          index
        }))
      ));
    }

    function orbitalSummary(atom) {
      const shells = atom.shells || [];
      const orbitals = flattenOrbitals(atom);
      const electrons = orbitals.reduce((total, orbital) => total + Number(orbital.electrons || 0), 0);
      return shells.length + " shells, " + orbitals.length + " orbitals, " + electrons + " e";
    }

    function shellSummary(atom) {
      return (atom.shells || []).map((shell) => {
        const groups = new Map();
        (shell.orbitals || []).forEach((orbital) => {
          const items = groups.get(orbital.kind) || [];
          items.push(Number(orbital.electrons || 0));
          groups.set(orbital.kind, items);
        });
        const text = Array.from(groups.entries()).map(([kind, counts]) => kind + "(" + counts.join("/") + ")").join(" ");
        return "n" + shell.n + ": " + text;
      }).join("<br>");
    }

    function normalizeVector(vector, fallback) {
      const candidate = vector || fallback || { x: 1, y: 0, z: 0 };
      const x = Number(candidate.x || 0);
      const y = Number(candidate.y || 0);
      const z = Number(candidate.z || 0);
      const length = Math.hypot(x, y, z);
      if (length < 0.0001) {
        return fallback || { x: 1, y: 0, z: 0 };
      }
      return { x: x / length, y: y / length, z: z / length };
    }

    function cross(a, b) {
      return {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x
      };
    }

    function fallbackAxis(orbital) {
      const name = String(orbital.orbital || "");
      if (name.includes("py")) return { x: 0, y: 1, z: 0 };
      if (name.includes("pz")) return { x: 0, y: 0, z: 1 };
      if (name.includes("xy")) return normalizeVector({ x: 1, y: 1, z: 0 });
      if (name.includes("yz")) return normalizeVector({ x: 0, y: 1, z: 1 });
      if (name.includes("xz")) return normalizeVector({ x: 1, y: 0, z: 1 });
      if (name.includes("z")) return { x: 0, y: 0, z: 1 };
      return { x: 1, y: 0, z: 0 };
    }

    function perpendicularAxis(axis) {
      const seed = Math.abs(axis.z) < 0.82 ? { x: 0, y: 0, z: 1 } : { x: 0, y: 1, z: 0 };
      return normalizeVector(cross(axis, seed), { x: 0, y: 1, z: 0 });
    }

    function projectLocal(atom, vector, distance) {
      return project({
        vx: atom.vx + vector.x * distance,
        vy: atom.vy + vector.y * distance,
        vz: atom.vz + vector.z * distance
      });
    }

    function drawOrbitalLobe(center, guide, radius, color, alpha) {
      const angle = Math.atan2(center.y - guide.y, center.x - guide.x);
      ctx.save();
      ctx.translate(center.x, center.y);
      ctx.rotate(angle);
      ctx.shadowColor = hexToRgba(color, Math.min(0.35, alpha));
      ctx.shadowBlur = 18;
      ctx.beginPath();
      ctx.ellipse(0, 0, radius * 1.22, radius * 0.72, 0, 0, Math.PI * 2);
      ctx.fillStyle = hexToRgba(color, alpha);
      ctx.fill();
      ctx.shadowColor = "transparent";
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = hexToRgba(color, Math.min(0.92, alpha + 0.22));
      ctx.stroke();
      ctx.restore();
    }

    function drawSOrbital(point, orbital) {
      const n = Number(orbital.n || 1);
      const radius = Math.max(18, (17 + n * 6) * point.p * state.zoom);
      const alpha = orbital.electrons > 0 ? 0.13 : 0.055;
      const color = "#38bdf8";
      const gradient = ctx.createRadialGradient(point.x, point.y, Math.max(2, radius * 0.12), point.x, point.y, radius);
      gradient.addColorStop(0, hexToRgba(color, alpha * 1.8));
      gradient.addColorStop(0.66, hexToRgba(color, alpha));
      gradient.addColorStop(1, hexToRgba(color, 0));
      ctx.save();
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = hexToRgba(color, alpha + 0.18);
      ctx.stroke();
      ctx.restore();
    }

    function orbitalStyle(kind) {
      if (kind === "d") return { color: "#f59e0b", lobes: 4 };
      if (kind === "f") return { color: "#22a06b", lobes: 6 };
      if (kind === "p") return { color: "#a855f7", lobes: 2 };
      return { color: "#38bdf8", lobes: 1 };
    }

    function drawDirectionalOrbital(atom, point, orbital) {
      const style = orbitalStyle(orbital.kind);
      const axis = normalizeVector(orbital.orientation, fallbackAxis(orbital));
      const perpendicular = perpendicularAxis(axis);
      const third = normalizeVector(cross(axis, perpendicular), { x: 0, y: 0, z: 1 });
      const axes = style.lobes === 6
        ? [axis, { x: -axis.x, y: -axis.y, z: -axis.z }, perpendicular, { x: -perpendicular.x, y: -perpendicular.y, z: -perpendicular.z }, third, { x: -third.x, y: -third.y, z: -third.z }]
        : style.lobes === 4
          ? [axis, { x: -axis.x, y: -axis.y, z: -axis.z }, perpendicular, { x: -perpendicular.x, y: -perpendicular.y, z: -perpendicular.z }]
          : [axis, { x: -axis.x, y: -axis.y, z: -axis.z }];
      const n = Number(orbital.n || 1);
      const shellDistance = 0.34 + Math.min(5, n) * 0.06;
      const radius = Math.max(8, (10 + Math.min(5, n) * 1.5) * point.p * state.zoom);
      const alpha = orbital.electrons > 0 ? Math.min(0.52, 0.22 + orbital.electrons * 0.09) : 0.08;
      axes.forEach((localAxis) => {
        const lobe = projectLocal(point.atom, localAxis, shellDistance);
        drawOrbitalLobe(lobe, point, radius, style.color, alpha);
      });
      const forward = projectLocal(point.atom, axis, shellDistance);
      const backward = projectLocal(point.atom, { x: -axis.x, y: -axis.y, z: -axis.z }, shellDistance);
      drawLine(forward, backward, {
        width: 1.2,
        color: style.color,
        alpha: Math.min(0.36, alpha)
      });
    }

    function drawSelectedAtomOrbitals(points) {
      const atom = state.selectedAtom;
      if (!atom) {
        return;
      }
      const point = points.get(atom.id);
      if (!point) {
        return;
      }
      const orbitals = flattenOrbitals(atom).sort((left, right) => Number(left.n || 0) - Number(right.n || 0));
      ctx.save();
      orbitals.forEach((orbital) => {
        if (orbital.kind === "s") {
          drawSOrbital(point, orbital);
        } else {
          drawDirectionalOrbital(atom, point, orbital);
        }
      });
      ctx.lineWidth = 2.6;
      ctx.strokeStyle = "rgba(0, 143, 135, 0.86)";
      ctx.beginPath();
      ctx.arc(point.x, point.y, (point.radius || 12) + 7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    function roundedRect(x, y, width, height, radius) {
      const r = Math.min(radius, width / 2, height / 2);
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + width, y, x + width, y + height, r);
      ctx.arcTo(x + width, y + height, x, y + height, r);
      ctx.arcTo(x, y + height, x, y, r);
      ctx.arcTo(x, y, x + width, y, r);
      ctx.closePath();
    }

    function drawSystemLabel(system, points) {
      const atoms = system.atoms.map((id) => points.get(id)).filter(Boolean);
      if (!atoms.length) {
        return;
      }
      const centroid = atoms.reduce((acc, point) => {
        acc.x += point.x;
        acc.y += point.y;
        acc.z += point.z;
        return acc;
      }, { x: 0, y: 0, z: 0 });
      centroid.x /= atoms.length;
      centroid.y /= atoms.length;
      centroid.z /= atoms.length;
      const active = state.activeSystem === system.id;
      ctx.save();
      ctx.font = "800 12px ui-sans-serif, system-ui, sans-serif";
      const text = system.label;
      const width = Math.min(ctx.measureText(text).width + 18, 190);
      const x = centroid.x + 12;
      const y = centroid.y - 12;
      ctx.globalAlpha = active ? 0.98 : 0.86;
      roundedRect(x, y - 18, width, 24, 8);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.strokeStyle = system.color;
      ctx.lineWidth = active ? 2 : 1;
      ctx.stroke();
      ctx.fillStyle = "#1f2933";
      ctx.fillText(text.length > 24 ? text.slice(0, 23) + "." : text, x + 9, y - 2);
      ctx.restore();
    }

    function drawAtom(point) {
      const atom = point.atom;
      const radius = point.radius;
      const active = (state.hoverAtom && state.hoverAtom.id === atom.id)
        || (state.selectedAtom && state.selectedAtom.id === atom.id);
      const gradient = ctx.createRadialGradient(point.x - radius * 0.32, point.y - radius * 0.36, radius * 0.1, point.x, point.y, radius);
      gradient.addColorStop(0, "#ffffff");
      gradient.addColorStop(0.18, atom.color);
      gradient.addColorStop(1, atom.edge);
      ctx.save();
      ctx.shadowColor = "rgba(31, 41, 51, 0.22)";
      ctx.shadowBlur = active ? 18 : 8;
      ctx.shadowOffsetY = active ? 5 : 3;
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();
      ctx.shadowColor = "transparent";
      ctx.lineWidth = active ? 2.4 : 1.2;
      ctx.strokeStyle = active ? "#008f87" : "rgba(31, 41, 51, 0.32)";
      ctx.stroke();
      if (state.labels) {
        ctx.font = "800 11px ui-sans-serif, system-ui, sans-serif";
        ctx.fillStyle = atom.symbol === "C" ? "#ffffff" : "#17202a";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(atom.symbol, point.x, point.y);
      }
      ctx.restore();
    }

    function draw() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      const points = new Map(molecule.atoms.map((atom) => [atom.id, project(atom)]));
      const bondDraws = molecule.bonds.map((bond) => ({
        bond,
        a: points.get(bond.a),
        b: points.get(bond.b)
      })).filter((item) => item.a && item.b).sort((left, right) => ((left.a.z + left.b.z) - (right.a.z + right.b.z)));
      const standardByEdge = standardSystemByEdge(molecule.systems);

      drawCoordinateAxes();
      drawChargeField(points);
      bondDraws.forEach(({ bond, a, b }) => {
        const perspective = ((a.p + b.p) / 2) * state.zoom;
        const width = Math.max(1.8, 2.4 * perspective);
        const spacing = Math.max(4, 5.2 * perspective);
        const gradient = bond.kind === "ionic" ? chargeGradientForEdge(molecule.atomMap, bond) : null;
        drawBondLines(a, b, {
          lineCount: gradient ? 1 : lineCountFromBond(bond, standardByEdge),
          width: gradient ? Math.max(2.6, width + 0.8) : width,
          spacing,
          color: COVALENT_BOND_COLOR,
          colorA: gradient ? gradient.colorA : COVALENT_BOND_COLOR,
          colorB: gradient ? gradient.colorB : COVALENT_BOND_COLOR,
          alpha: gradient ? 0.92 : 0.86
        });
      });

      if (state.systems) {
        const lanes = systemEdgeLaneMap(molecule.systems.filter((system) => !isStandardCovalentSystem(system)));
        molecule.systems.forEach((system) => {
          const selected = state.activeSystem === system.id;
          const active = state.activeSystem === null || selected;
          system.edges.forEach((edge) => {
            const a = points.get(edge.a);
            const b = points.get(edge.b);
            if (a && b) {
              const perspective = ((a.p + b.p) / 2) * state.zoom;
              if (isIonicSystem(system)) {
                const gradient = chargeGradientForEdge(molecule.atomMap, edge);
                drawBondLines(a, b, {
                  lineCount: 1,
                  width: active ? Math.max(3.2, 3.4 * perspective) : Math.max(2.2, 2.4 * perspective),
                  spacing: Math.max(4, 5.2 * perspective),
                  colorA: gradient.colorA,
                  colorB: gradient.colorB,
                  alpha: active ? 0.95 : 0.22
                });
                return;
              }
              const covalentLines = standardCovalentLineCount(system);
              if (covalentLines) {
                if (!selected) return;
                drawBondLines(a, b, {
                  lineCount: covalentLines,
                  width: Math.max(2.2, 2.8 * perspective),
                  spacing: Math.max(4.4, 5.6 * perspective),
                  color: COVALENT_BOND_COLOR,
                  alpha: 0.98
                });
                return;
              }
              const lane = lanes.get(edgeKey(edge) + ":" + Number(system.id)) || { index: 0, count: 1 };
              const width = active ? 6.2 : 3.8;
              const laneOffset = lane.count > 1 ? (lane.index - (lane.count - 1) / 2) * (width + 2.0) : 0;
              const edgeColor = system.color;
              drawLine(a, b, {
                width,
                color: edgeColor,
                colorA: edgeColor,
                colorB: edgeColor,
                alpha: active ? 1 : 0.18,
                offset: laneOffset
              });
            }
          });
        });
      }

      drawSelectedAtomOrbitals(points);
      const sortedAtoms = Array.from(points.values()).sort((a, b) => a.z - b.z);
      sortedAtoms.forEach(drawAtom);
      state.screenAtoms = sortedAtoms;
      if (state.systems) {
        displaySystems(molecule.systems).forEach((system) => drawSystemLabel(system, points));
      }
    }

    function switchMolecule(index) {
      viewer.activeIndex = index;
      molecule = viewer.molecules[viewer.activeIndex] || emptyMolecule();
      state.activeSystem = null;
      state.selectedAtom = null;
      state.hoverAtom = null;
      renderPanel();
      draw();
    }

    function renderPanel() {
      document.getElementById("molecule-title").textContent = molecule.title;
      document.getElementById("atom-count").textContent = String(molecule.atoms.length);
      document.getElementById("bond-count").textContent = String(molecule.bonds.length);
      document.getElementById("system-count").textContent = String(molecule.systems.length);
      document.getElementById("toggle-labels").classList.toggle("active", state.labels);
      document.getElementById("toggle-systems").classList.toggle("active", state.systems);
      document.getElementById("toggle-axes").classList.toggle("active", state.axes);

      const moleculeSection = document.getElementById("molecule-section");
      const moleculeList = document.getElementById("molecule-list");
      moleculeSection.style.display = viewer.molecules.length > 1 ? "block" : "none";
      moleculeList.innerHTML = "";
      viewer.molecules.forEach((item, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "molecule-row";
        button.classList.toggle("active", index === viewer.activeIndex);
        button.innerHTML = '<span class="index-badge"></span><span><span class="row-title"></span><span class="row-meta"></span></span><span class="pill"></span>';
        button.querySelector(".index-badge").textContent = String(index + 1);
        button.querySelector(".row-title").textContent = item.title;
        button.querySelector(".row-meta").textContent = item.atoms.length + " atoms, " + item.systems.length + " systems";
        button.querySelector(".pill").textContent = item.bonds.length + " bonds";
        button.addEventListener("click", () => switchMolecule(index));
        moleculeList.append(button);
      });

      const systemList = document.getElementById("system-list");
      systemList.innerHTML = "";
      const visibleSystems = displaySystems(molecule.systems);
      if (!visibleSystems.length) {
        const empty = document.createElement("p");
        empty.className = "row-meta";
        empty.textContent = "None";
        systemList.append(empty);
      }
      visibleSystems.forEach((system) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "system-row";
        button.classList.toggle("active", state.activeSystem === system.id);
        button.innerHTML = '<span class="swatch"></span><span><span class="row-title"></span><span class="row-meta"></span></span><span class="pill"></span>';
        button.querySelector(".swatch").style.background = system.color;
        button.querySelector(".row-title").textContent = system.label;
        button.querySelector(".row-meta").textContent = system.atoms.length + " atoms, " + system.edges.length + " edges";
        button.querySelector(".pill").textContent = system.sharedElectrons + "e";
        button.addEventListener("click", () => {
          state.activeSystem = state.activeSystem === system.id ? null : system.id;
          renderPanel();
          draw();
        });
        systemList.append(button);
      });

      const atomList = document.getElementById("atom-list");
      atomList.innerHTML = "";
      molecule.atoms.forEach((atom) => {
        const row = document.createElement("div");
        row.className = "atom-row";
        row.classList.toggle("active", state.selectedAtom && state.selectedAtom.id === atom.id);
        const charge = atom.charge ? (atom.charge > 0 ? "+" + atom.charge : String(atom.charge)) : "0";
        row.innerHTML = '<span class="atom-dot"></span><span class="row-title"></span><span class="row-meta"></span>';
        row.querySelector(".atom-dot").style.background = atom.color;
        row.querySelector(".row-title").textContent = atom.label;
        row.querySelector(".row-meta").textContent = "charge " + charge + ", " + orbitalSummary(atom);
        row.addEventListener("click", () => {
          state.selectedAtom = state.selectedAtom && state.selectedAtom.id === atom.id ? null : atom;
          setSelection(state.selectedAtom);
          renderPanel();
          draw();
        });
        atomList.append(row);
      });
      setSelection(state.selectedAtom);
    }

    function setSelection(atom) {
      const selection = document.getElementById("selection");
      if (!atom) {
        selection.textContent = "None";
        return;
      }
      const systems = displaySystems(molecule.systems).filter((system) => system.atoms.includes(atom.id)).map((system) => system.label);
      const charge = atom.charge ? (atom.charge > 0 ? "+" + atom.charge : String(atom.charge)) : "0";
      const shellText = shellSummary(atom);
      selection.innerHTML = "<strong>" + escapeHtml(atom.label) + "</strong><br>charge " + escapeHtml(charge) + "<br>"
        + escapeHtml(orbitalSummary(atom)) + (shellText ? "<br>" + shellText.split("<br>").map(escapeHtml).join("<br>") : "")
        + "<br>" + (systems.length ? systems.map(escapeHtml).join("<br>") : "no bonding systems")
        + geometrySummary(atom);
    }

    function updateHover(event) {
      state.hoverAtom = pickAtom(event);
      if (state.hoverAtom) {
        tooltip.style.display = "block";
        tooltip.style.left = event.clientX + 14 + "px";
        tooltip.style.top = event.clientY + 14 + "px";
        tooltip.textContent = state.hoverAtom.label;
      } else {
        tooltip.style.display = "none";
      }
      draw();
    }

    function pickAtom(event) {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      let best = null;
      for (const point of state.screenAtoms) {
        const distance = Math.hypot(point.x - x, point.y - y);
        if (distance <= point.radius + 5 && (!best || distance < best.distance)) {
          best = { atom: point.atom, distance };
        }
      }
      return best ? best.atom : null;
    }

    canvas.addEventListener("pointerdown", (event) => {
      state.pointerDown = true;
      state.pointerMoved = false;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      state.pointerStartX = event.clientX;
      state.pointerStartY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener("pointermove", (event) => {
      if (state.pointerDown) {
        const dx = event.clientX - state.lastX;
        const dy = event.clientY - state.lastY;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        if (Math.hypot(event.clientX - state.pointerStartX, event.clientY - state.pointerStartY) > 4) {
          state.pointerMoved = true;
        }
        state.ry += dx * 0.009;
        state.rx += dy * 0.009;
        draw();
      } else {
        updateHover(event);
      }
    });

    canvas.addEventListener("pointerup", (event) => {
      if (!state.pointerMoved) {
        const atom = pickAtom(event);
        state.selectedAtom = atom && (!state.selectedAtom || state.selectedAtom.id !== atom.id) ? atom : null;
        setSelection(state.selectedAtom);
        renderPanel();
        draw();
      }
      state.pointerDown = false;
      canvas.releasePointerCapture(event.pointerId);
    });

    canvas.addEventListener("pointerleave", () => {
      state.pointerDown = false;
      state.hoverAtom = null;
      tooltip.style.display = "none";
      draw();
    });

    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const next = state.zoom * (event.deltaY > 0 ? 0.92 : 1.08);
      state.zoom = Math.min(2.8, Math.max(0.42, next));
      draw();
    }, { passive: false });

    document.getElementById("reset-view").addEventListener("click", () => {
      state.rx = -0.62;
      state.ry = 0.74;
      state.zoom = 1;
      state.activeSystem = null;
      state.selectedAtom = null;
      renderPanel();
      draw();
    });

    document.getElementById("toggle-labels").addEventListener("click", () => {
      state.labels = !state.labels;
      renderPanel();
      draw();
    });

    document.getElementById("toggle-systems").addEventListener("click", () => {
      state.systems = !state.systems;
      renderPanel();
      draw();
    });

    document.getElementById("toggle-axes").addEventListener("click", () => {
      state.axes = !state.axes;
      renderPanel();
      draw();
    });

    window.loadMolADT = (payload) => {
      viewer = normalizeViewerPayload(payload);
      molecule = viewer.molecules[viewer.activeIndex] || emptyMolecule();
      state.activeSystem = null;
      state.selectedAtom = null;
      state.hoverAtom = null;
      renderPanel();
      draw();
    };

    window.addEventListener("dragover", (event) => {
      event.preventDefault();
      document.body.classList.add("dragging");
    });

    window.addEventListener("dragleave", (event) => {
      if (!event.relatedTarget) {
        document.body.classList.remove("dragging");
      }
    });

    window.addEventListener("drop", async (event) => {
      event.preventDefault();
      document.body.classList.remove("dragging");
      const file = event.dataTransfer.files[0];
      if (!file) {
        return;
      }
      const text = await file.text();
      window.loadMolADT(JSON.parse(text));
    });

    window.addEventListener("resize", resizeCanvas);
    renderPanel();
    resizeCanvas();
  </script>
</body>
</html>
"""


__all__ = [
    "molecule_viewer_collection_html",
    "molecule_viewer_collection_payload",
    "molecule_viewer_html",
    "molecule_viewer_payload",
    "molecule_viewer_uri",
    "open_molecule_viewer",
    "write_molecule_viewer_collection_html",
    "write_molecule_viewer_html",
]
