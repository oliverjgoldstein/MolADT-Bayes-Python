from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import Iterable

from .coordinate import Coordinate
from .dietz import AtomId, BondingSystem, Edge
from .dietz import SystemId
from .molecule import Atom, Molecule, SmilesAtomStereo, SmilesBondStereo, molecule_edges
from .molecule_ops import effective_order, neighbors_sigma
from .validate import used_electrons_at
from .orbital import (
    Orbital,
    PureDOrbital,
    PureFOrbital,
    PureOrbital,
    PurePOrbital,
    PureSOrbital,
    Shell,
    Shells,
    SubShell,
)


@dataclass(frozen=True, slots=True)
class PrettyBlock:
    """A manuscript-facing rendering block for MolADT values."""

    lines: tuple[str, ...]

    def render(self) -> str:
        return "\n".join(self.lines)

    def indented(self, spaces: int) -> PrettyBlock:
        prefix = " " * spaces
        return PrettyBlock(tuple(prefix + line if line else line for line in self.lines))


def pretty_text(value: object) -> str:
    """Render a registered MolADT value using singledispatch as a lightweight typeclass."""

    return pretty_block(value).render()


def pretty_shells(shells: Shells | None) -> str:
    return PrettyBlock(tuple(_pretty_shell_lines(shells))).render()


@singledispatch
def pretty_block(value: object) -> PrettyBlock:
    raise TypeError(f"No pretty renderer is registered for {type(value)!r}")


@pretty_block.register
def _(molecule: Molecule) -> PrettyBlock:
    atom_items = sorted(molecule.atoms.items(), key=lambda item: item[0].value)
    system_items = sorted(molecule.systems, key=lambda item: item[0].value)
    graph_edges = _edge_network_edges(molecule, system_items)
    total_charge = sum(atom.formal_charge for _, atom in atom_items)
    atom_stereo = molecule.smiles_stereochemistry.atom_stereo
    bond_stereo = molecule.smiles_stereochemistry.bond_stereo

    lines: list[str] = [
        "Molecule Report",
        "===============",
        *_summary_section(atom_items, graph_edges, system_items, total_charge, atom_stereo, bond_stereo),
        "",
    ]

    lines.extend(_section_header("Atoms"))
    if atom_items:
        lines.extend(_format_atom_table(molecule, atom_items, system_items))
    else:
        lines.append("none")
    lines.append("")

    lines.extend(_section_header("Electron Shells"))
    if atom_items:
        for index, (_, atom) in enumerate(atom_items, start=1):
            lines.extend(_format_atom_shell_block(atom))
            if index != len(atom_items):
                lines.append("")
    else:
        lines.append("none")
    lines.append("")

    lines.extend(_section_header("Edge Network"))
    if graph_edges:
        for index, edge in enumerate(graph_edges, start=1):
            lines.append(f"{index:02d}. {_format_bond_line(molecule, edge)}")
    else:
        lines.append("none")
    lines.append("")

    lines.extend(_section_header("Bonding Systems"))
    if system_items:
        for index, (system_id, system) in enumerate(system_items, start=1):
            lines.extend(_format_system_block(molecule, system_id, system))
            if index != len(system_items):
                lines.append("")
    else:
        lines.append("none")
    lines.append("")

    if atom_stereo or bond_stereo:
        lines.extend(_section_header("SMILES Stereochemistry"))
        if atom_stereo:
            lines.append("atom-centered:")
            lines.extend(_indent((_format_atom_stereo(item) for item in atom_stereo), 2))
        if bond_stereo:
            if atom_stereo:
                lines.append("")
            lines.append("bond-directed:")
            lines.extend(_indent((_format_bond_stereo(molecule, item) for item in bond_stereo), 2))

    return PrettyBlock(tuple(lines[:-1] if lines and lines[-1] == "" else lines))


@pretty_block.register
def _(atom: Atom) -> PrettyBlock:
    return PrettyBlock(tuple(_atom_lines(atom)))


@pretty_block.register
def _(system: BondingSystem) -> PrettyBlock:
    label = _system_display_label(system)
    tag_suffix = f" [{label}]" if label else ""
    member_atoms = ", ".join(f"#{atom_id.value}" for atom_id in sorted(system.member_atoms)) or "none"
    edge_lines = [f"{edge.a.value}-{edge.b.value}" for edge in sorted(system.member_edges)]
    edge_electrons = _system_electrons_per_edge(system)
    lines = [
        f"Bonding system{tag_suffix}: {_format_electrons(system.shared_electrons.value)} shared electron pool",
        f"  member atoms: {member_atoms}",
    ]
    if edge_lines:
        lines.append(
            f"  member edges ({_format_electrons(edge_electrons)} shared over each edge, "
            f"order contribution {edge_electrons / 2.0:.2f}):"
        )
        lines.extend(f"    {edge_line}" for edge_line in edge_lines)
    else:
        lines.append("  member edges: (none)")
    return PrettyBlock(tuple(lines))


@pretty_block.register
def _(shell: Shell) -> PrettyBlock:
    return PrettyBlock(tuple(_pretty_shell_lines((shell,))))


@pretty_block.register
def _(subshell: SubShell) -> PrettyBlock:
    orbital_lines = [_format_orbital(orbital) for orbital in subshell.orbitals]
    return PrettyBlock(tuple(orbital_lines or ["(empty subshell)"]))


@pretty_block.register
def _(orbital: Orbital) -> PrettyBlock:
    return PrettyBlock((_format_orbital(orbital),))


def _count_label(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _summary_section(
    atom_items: list[tuple[AtomId, Atom]],
    graph_edges: list[Edge],
    system_items: list[tuple[SystemId, BondingSystem]],
    total_charge: int,
    atom_stereo: tuple[SmilesAtomStereo, ...],
    bond_stereo: tuple[SmilesBondStereo, ...],
) -> list[str]:
    stereo_bits: list[str] = []
    if atom_stereo:
        stereo_bits.append(f"{len(atom_stereo)} atom")
    if bond_stereo:
        stereo_bits.append(f"{len(bond_stereo)} bond")
    stereo_summary = ", ".join(stereo_bits) if stereo_bits else "none"
    return [
        _summary_line("atoms", str(len(atom_items))),
        _summary_line("heavy atoms", str(sum(1 for _, atom in atom_items if atom.attributes.symbol.value != "H"))),
        _summary_line("edges", str(len(graph_edges))),
        _summary_line("bonding systems", str(len(system_items))),
        _summary_line("net charge", f"{total_charge:+d}"),
        _summary_line("composition", _molecular_formula(atom_items)),
        _summary_line("stereo flags", stereo_summary),
    ]


def _summary_line(label: str, value: str) -> str:
    return f"{label:<16} {value}"


def _section_header(title: str) -> list[str]:
    return [title, "-" * len(title)]


def _molecular_formula(atom_items: list[tuple[AtomId, Atom]]) -> str:
    counts: dict[str, int] = {}
    for _, atom in atom_items:
        symbol = atom.attributes.symbol.value
        counts[symbol] = counts.get(symbol, 0) + 1
    if not counts:
        return "(empty)"
    symbols = list(counts)
    if "C" in counts:
        order = ["C"]
        if "H" in counts:
            order.append("H")
        order.extend(sorted(symbol for symbol in symbols if symbol not in {"C", "H"}))
    else:
        order = sorted(symbols)
    return " ".join(symbol if counts[symbol] == 1 else f"{symbol}{counts[symbol]}" for symbol in order)


def _indent(lines: Iterable[str], spaces: int) -> list[str]:
    prefix = " " * spaces
    return [prefix + line if line else line for line in lines]


def _format_atom_table(
    molecule: Molecule,
    atom_items: list[tuple[AtomId, Atom]],
    system_items: list[tuple[SystemId, BondingSystem]],
) -> list[str]:
    rows = []
    for atom_id, atom in atom_items:
        neighbor_refs = ", ".join(
            _render_atom_ref(molecule.atoms[neighbor_id])
            for neighbor_id in neighbors_sigma(molecule, atom_id)
        ) or "-"
        system_refs = ", ".join(_system_labels_for_atom(system_items, atom_id)) or "-"
        rows.append(
            (
                _render_atom_ref_bracketed(atom),
                str(atom.attributes.atomic_number),
                f"{atom.formal_charge:+d}",
                str(len(neighbors_sigma(molecule, atom_id))),
                f"{used_electrons_at(molecule, atom_id):.2f}",
                _format_coord(atom.coordinate),
                neighbor_refs,
                system_refs,
            )
        )

    headers = ("atom", "Z", "chg", "degree", "used", "xyz (Angstrom)", "edge neighbors", "systems")
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    return [
        _format_table_row(headers, widths),
        _format_table_separator(widths),
        *(_format_table_row(row, widths) for row in rows),
    ]


def _edge_network_edges(
    molecule: Molecule,
    system_items: list[tuple[SystemId, BondingSystem]],
) -> list[Edge]:
    edges = set(molecule_edges(molecule))
    for _, system in system_items:
        edges.update(system.member_edges)
    return sorted(edges)


def _format_atom_shell_block(atom: Atom) -> list[str]:
    shell_lines = _pretty_shell_lines(atom.shells)
    if not shell_lines:
        return [f"{_render_atom_ref_bracketed(atom)} shells: none"]
    return [f"{_render_atom_ref_bracketed(atom)} shells:", *_indent(shell_lines, 2)]


def _atom_lines(atom: Atom, *, extra_lines: Iterable[str] = ()) -> list[str]:
    lines = [
        _format_atom_header(atom),
        f"{_detail_label('xyz')} {_format_coord(atom.coordinate)}",
        f"{_detail_label('charge')} {atom.formal_charge:+d}",
        *extra_lines,
    ]
    shell_lines = _pretty_shell_lines(atom.shells)
    if shell_lines:
        lines.append("shells:")
        lines.extend(_indent(shell_lines, 2))
    return lines


def _format_table_row(values: Iterable[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(values, widths, strict=True)).rstrip()


def _format_table_separator(widths: list[int]) -> str:
    return "  ".join("-" * width for width in widths)


def _format_atom_header(atom: Atom) -> str:
    attrs = atom.attributes
    return f"[{attrs.symbol.value}#{atom.atom_id.value}] Z={attrs.atomic_number}  mass={attrs.atomic_weight:.4f} u"


def _detail_label(label: str) -> str:
    return f"{label + ':':<8}"


def _format_coord(coordinate: Coordinate) -> str:
    return f"({coordinate.x.value: .4f}, {coordinate.y.value: .4f}, {coordinate.z.value: .4f})"


def _format_bond_line(molecule: Molecule, edge: Edge) -> str:
    pair = _format_edge_short(molecule, edge)
    shared_text = f"shared={_format_electrons(_edge_shared_electrons(molecule.systems, edge))}"
    order_text = f"order={effective_order(molecule, edge):.2f}"
    system_labels = [
        _format_edge_system_ref(system_id, system)
        for system_id, system in molecule.systems
        if edge in system.member_edges
    ]
    suffix = f"  systems={', '.join(system_labels)}" if system_labels else ""
    return f"{pair}  {shared_text}  {order_text}{suffix}"


def _format_system_block(molecule: Molecule, system_id: SystemId, system: BondingSystem) -> list[str]:
    label = _system_display_label(system)
    title = f"[#{system_id.value}] {label}" if label else f"[#{system_id.value}]"
    lines = [
        title,
        f"  shared electrons: {_format_electrons(system.shared_electrons.value)} pool",
        f"  member edges:     {len(system.member_edges)}",
    ]
    atom_refs = ", ".join(_render_atom_ref(molecule.atoms[atom_id]) for atom_id in sorted(system.member_atoms))
    if atom_refs:
        lines.append(f"  member atoms:     {atom_refs}")
    if system.member_edges:
        edge_electrons = _system_electrons_per_edge(system)
        lines.append(f"  edge share:       {_format_electrons(edge_electrons)} per listed edge")
        lines.append(f"  bond-order part:  {edge_electrons / 2.0:.2f} per listed edge")
        lines.append("  edge list:")
        lines.extend(
            f"    - {_format_edge_short(molecule, edge)}  shares {_format_electrons(edge_electrons)}"
            for edge in sorted(system.member_edges)
        )
    else:
        lines.append("  edge list:        none")
    return lines


def _render_atom_ref(atom: Atom) -> str:
    return f"{atom.attributes.symbol.value}#{atom.atom_id.value}"


def _render_atom_ref_bracketed(atom: Atom) -> str:
    return f"[{_render_atom_ref(atom)}]"


def _format_edge_short(molecule: Molecule, edge: Edge) -> str:
    left = _render_atom_ref(molecule.atoms[edge.a])
    right = _render_atom_ref(molecule.atoms[edge.b])
    return f"{left} <-> {right}"


def _format_system_label(system_id: int, system: BondingSystem) -> str:
    label = _system_display_label(system)
    return f"#{system_id}[{label}]" if label else f"#{system_id}"


def _format_edge_system_ref(system_id: SystemId, system: BondingSystem) -> str:
    label = _format_system_label(system_id.value, system)
    edge_electrons = _system_electrons_per_edge(system)
    if _is_ionic_system(system):
        return f"{label}:{_format_electrons(edge_electrons)}"
    if len(system.member_edges) == 1 and system.shared_electrons.value in {2, 4, 6, 8}:
        return f"{label}:{_format_electrons(edge_electrons)}"
    return f"{label}:{_format_electrons(edge_electrons)}/edge from {_format_electrons(system.shared_electrons.value)}"


def _system_electrons_per_edge(system: BondingSystem) -> float:
    if not system.member_edges:
        return 0.0
    return system.shared_electrons.value / len(system.member_edges)


def _edge_shared_electrons(system_items: Iterable[tuple[SystemId, BondingSystem]], edge: Edge) -> float:
    return sum(_system_electrons_per_edge(system) for _, system in system_items if edge in system.member_edges)


def _system_display_label(system: BondingSystem) -> str | None:
    if _is_ionic_system(system):
        return "ionic"
    covalent_label = _covalent_label(system)
    if covalent_label is not None:
        return covalent_label
    return system.tag


def _is_ionic_system(system: BondingSystem) -> bool:
    return system.tag == "ionic" and len(system.member_edges) == 1 and system.shared_electrons.value == 0


def _covalent_label(system: BondingSystem) -> str | None:
    if system.tag not in {None, "single", "double", "triple", "quadruple"}:
        return None
    if len(system.member_edges) != 1:
        return None
    if system.shared_electrons.value == 2:
        return "single covalent"
    if system.shared_electrons.value == 4:
        return "double covalent"
    if system.shared_electrons.value == 6:
        return "triple covalent"
    if system.shared_electrons.value == 8:
        return "quadruple covalent"
    return None


def _format_electrons(value: float | int) -> str:
    numeric = float(value)
    rounded = round(numeric)
    if abs(numeric - rounded) <= 1e-9:
        return f"{rounded}e"
    return f"{numeric:.2f}e"


def _system_labels_for_atom(system_items: list[tuple[SystemId, BondingSystem]], atom_id: AtomId) -> list[str]:
    return [
        f"{_format_system_label(system_id.value, system)}:{_format_electrons(system.shared_electrons.value)}"
        for system_id, system in system_items
        if atom_id in system.member_atoms
    ]


def _format_atom_stereo(stereo: SmilesAtomStereo) -> str:
    return (
        f"center #{stereo.center.value}: {stereo.stereo_class.value}{stereo.configuration} "
        f"from token {stereo.token}"
    )


def _format_bond_stereo(molecule: Molecule, stereo: SmilesBondStereo) -> str:
    left = _render_atom_ref(molecule.atoms[stereo.start_atom])
    right = _render_atom_ref(molecule.atoms[stereo.end_atom])
    return f"{left} -> {right}: {stereo.direction.value}"


def _pretty_shell_lines(shells: Shells | None) -> list[str]:
    if shells is None:
        return []
    lines: list[str] = []
    for shell in shells:
        subshell_lines = _format_shell(shell)
        lines.extend(subshell_lines or [f"n={shell.principal_quantum_number} (empty)"])
    return lines


def _format_shell(shell: Shell) -> list[str]:
    summaries: list[str] = []
    orbital_lines: list[str] = []
    for label, subshell in (
        ("s", shell.s_subshell),
        ("p", shell.p_subshell),
        ("d", shell.d_subshell),
        ("f", shell.f_subshell),
    ):
        if subshell is None:
            continue
        total_electrons = sum(orbital.electron_count for orbital in subshell.orbitals)
        summaries.append(f"{label} {total_electrons}e")
        orbital_lines.extend(f"- {_format_orbital(orbital)}" for orbital in subshell.orbitals)
    if not summaries:
        return []
    return [f"n={shell.principal_quantum_number} :: {' | '.join(summaries)}", *_indent(orbital_lines, 2)]


def _format_orbital(orbital: Orbital) -> str:
    base = f"{orbital.orbital_type.value} ({orbital.electron_count} e)"
    if orbital.orientation is not None:
        base += f" @ {_format_orientation(orbital.orientation)}"
    if orbital.hybrid_components:
        base += f" hybrid {_format_hybrid(orbital.hybrid_components)}"
    return base


def _format_orientation(coordinate: Coordinate) -> str:
    return f"<{coordinate.x.value: .3f}, {coordinate.y.value: .3f}, {coordinate.z.value: .3f}>"


def _format_hybrid(components: tuple[tuple[float, PureOrbital], ...]) -> str:
    return " + ".join(f"{weight:.2f}x{_format_pure_orbital(pure_orbital)}" for weight, pure_orbital in components)


def _format_pure_orbital(pure_orbital: PureOrbital) -> str:
    match pure_orbital:
        case PureSOrbital():
            return "s"
        case PurePOrbital(orbital=orbital):
            return orbital.value
        case PureDOrbital(orbital=orbital):
            return orbital.value
        case PureFOrbital(orbital=orbital):
            return orbital.value
    raise TypeError(f"Unsupported pure orbital {pure_orbital!r}")
