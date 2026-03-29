from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import Iterable

from .coordinate import Coordinate
from .dietz import AtomId, BondingSystem, Edge
from .dietz import SystemId
from .molecule import Atom, Molecule, effective_order, neighbors_sigma
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


def pretty_shells(shells: Shells) -> str:
    return PrettyBlock(tuple(_pretty_shell_lines(shells))).render()


@singledispatch
def pretty_block(value: object) -> PrettyBlock:
    raise TypeError(f"No pretty renderer is registered for {type(value)!r}")


@pretty_block.register
def _(molecule: Molecule) -> PrettyBlock:
    atom_items = sorted(molecule.atoms.items(), key=lambda item: item[0].value)
    sigma_edges = sorted(molecule.local_bonds)
    system_items = sorted(molecule.systems, key=lambda item: item[0].value)

    lines: list[str] = [
        "Molecule with "
        f"{_count_label(len(atom_items), 'atom', 'atoms')}, "
        f"{_count_label(len(sigma_edges), 'sigma bond', 'sigma bonds')}, "
        f"{_count_label(len(system_items), 'bonding system', 'bonding systems')}",
        "",
    ]

    if atom_items:
        lines.append(f"Atoms ({len(atom_items)}):")
        for atom_id, atom in atom_items:
            lines.extend(_indent(_format_atom_block(molecule, atom_id, atom), 2))
            lines.append("")
    else:
        lines.append("Atoms: (none)")
        lines.append("")

    if sigma_edges:
        lines.append(f"Sigma bonds ({len(sigma_edges)}):")
        lines.extend(_indent((_format_bond_line(molecule, edge) for edge in sigma_edges), 2))
        lines.append("")
    else:
        lines.append("Sigma bonds: (none)")
        lines.append("")

    if system_items:
        lines.append(f"Bonding systems ({len(system_items)}):")
        for system_id, system in system_items:
            lines.extend(_indent(_format_system_block(molecule, system_id, system), 2))
            lines.append("")
    else:
        lines.append("Bonding systems: (none)")
        lines.append("")

    return PrettyBlock(tuple(lines[:-1] if lines and lines[-1] == "" else lines))


@pretty_block.register
def _(atom: Atom) -> PrettyBlock:
    return PrettyBlock(tuple(_atom_lines(atom)))


@pretty_block.register
def _(system: BondingSystem) -> PrettyBlock:
    tag_suffix = f" [{system.tag}]" if system.tag else ""
    member_atoms = ", ".join(f"#{atom_id.value}" for atom_id in sorted(system.member_atoms)) or "none"
    edge_lines = [f"{edge.a.value}-{edge.b.value}" for edge in sorted(system.member_edges)]
    per_edge = (
        system.shared_electrons.value / (2.0 * len(system.member_edges))
        if system.member_edges
        else 0.0
    )
    lines = [
        f"Bonding system{tag_suffix}: {system.shared_electrons.value} shared electrons",
        f"  member atoms: {member_atoms}",
    ]
    if edge_lines:
        lines.append(f"  member edges (+{per_edge:.2f} bond order each):")
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


def _indent(lines: Iterable[str], spaces: int) -> list[str]:
    prefix = " " * spaces
    return [prefix + line if line else line for line in lines]


def _format_atom_block(molecule: Molecule, atom_id: AtomId, atom: Atom) -> list[str]:
    neighbor_ids = neighbors_sigma(molecule, atom_id)
    neighbor_refs = ", ".join(_render_atom_ref(molecule.atoms[neighbor_id]) for neighbor_id in neighbor_ids) or "none"
    return _atom_lines(atom, extra_lines=(f"{_detail_label('sigma neighbors:')} {neighbor_refs}",))


def _atom_lines(atom: Atom, *, extra_lines: Iterable[str] = ()) -> list[str]:
    lines = [
        _format_atom_header(atom),
        f"{_detail_label('coordinates (Angstrom):')} {_format_coord(atom.coordinate)}",
        f"{_detail_label('formal charge:')} {atom.formal_charge:+d}",
        *extra_lines,
    ]
    shell_lines = _pretty_shell_lines(atom.shells)
    if shell_lines:
        lines.append(f"{_detail_label('electron shells:')} ")
        lines.extend(_indent(shell_lines, 2))
    return lines


def _format_atom_header(atom: Atom) -> str:
    attrs = atom.attributes
    return f"{attrs.symbol.value} #{atom.atom_id.value} (Z={attrs.atomic_number}, {attrs.atomic_weight:.4f} u)"


def _detail_label(label: str) -> str:
    return f"{label:<22}"


def _format_coord(coordinate: Coordinate) -> str:
    return f"({coordinate.x.value: .4f}, {coordinate.y.value: .4f}, {coordinate.z.value: .4f})"


def _format_bond_line(molecule: Molecule, edge: Edge) -> str:
    pair = _format_edge_short(molecule, edge)
    order_text = f"{effective_order(molecule, edge):.2f}"
    system_labels = [
        _format_system_label(system_id.value, system.tag)
        for system_id, system in molecule.systems
        if edge in system.member_edges
    ]
    suffix = f"; systems: {', '.join(system_labels)}" if system_labels else ""
    return f"{pair} (order {order_text}{suffix})"


def _format_system_block(molecule: Molecule, system_id: SystemId, system: BondingSystem) -> list[str]:
    tag_suffix = f" [{system.tag}]" if system.tag else ""
    lines = [f"System {system_id.value}{tag_suffix}: {system.shared_electrons.value} shared electrons"]
    atom_refs = ", ".join(_render_atom_ref(molecule.atoms[atom_id]) for atom_id in sorted(system.member_atoms))
    if atom_refs:
        lines.append(f"  Atoms: {atom_refs}")
    if system.member_edges:
        per_edge = system.shared_electrons.value / (2.0 * len(system.member_edges))
        lines.append(f"  Edges (+{per_edge:.2f} to bond order each):")
        lines.extend(f"    {_format_edge_short(molecule, edge)}" for edge in sorted(system.member_edges))
    else:
        lines.append("  Edges: (none)")
    return lines


def _render_atom_ref(atom: Atom) -> str:
    return f"{atom.attributes.symbol.value}#{atom.atom_id.value}"


def _format_edge_short(molecule: Molecule, edge: Edge) -> str:
    left = _render_atom_ref(molecule.atoms[edge.a])
    right = _render_atom_ref(molecule.atoms[edge.b])
    return f"{left} <-> {right}"


def _format_system_label(system_id: int, tag: str | None) -> str:
    return f"#{system_id} [{tag}]" if tag else f"#{system_id}"


def _pretty_shell_lines(shells: Shells) -> list[str]:
    lines: list[str] = []
    for shell in shells:
        subshell_lines = _format_shell(shell)
        lines.extend(subshell_lines or [f"n={shell.principal_quantum_number} (empty)"])
    return lines


def _format_shell(shell: Shell) -> list[str]:
    body: list[str] = []
    body.extend(_format_named_subshell("s", shell.s_subshell))
    body.extend(_format_named_subshell("p", shell.p_subshell))
    body.extend(_format_named_subshell("d", shell.d_subshell))
    body.extend(_format_named_subshell("f", shell.f_subshell))
    if not body:
        return []
    return [f"n={shell.principal_quantum_number}", *_indent(body, 2)]


def _format_named_subshell(label: str, subshell: SubShell | None) -> list[str]:
    if subshell is None:
        return []
    total_electrons = sum(orbital.electron_count for orbital in subshell.orbitals)
    return [f"{label}: {total_electrons} e", *_indent((_format_orbital(orbital) for orbital in subshell.orbitals), 2)]


def _format_orbital(orbital: Orbital) -> str:
    base = f"{orbital.orbital_type.value} ({orbital.electron_count} e)"
    if orbital.orientation is not None:
        base += f", orientation {_format_orientation(orbital.orientation)}"
    if orbital.hybrid_components:
        base += f", hybrid {_format_hybrid(orbital.hybrid_components)}"
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
