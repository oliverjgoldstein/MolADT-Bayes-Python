from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from re import search
import shlex
from types import MappingProxyType
from typing import Any, Mapping

from ..chem.constants import element_attributes, element_shells
from ..chem.coordinate import Coordinate, mk_angstrom
from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system, mk_edge
from ..chem.molecule import Atom, AtomicSymbol, Molecule
from .molecule_json import molecule_to_dict


@dataclass(frozen=True, slots=True)
class SDFRecord:
    molecule: Molecule
    properties: Mapping[str, str]
    title: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "properties", MappingProxyType(dict(self.properties)))

    def property(self, name: str) -> str | None:
        return self.properties.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "properties": dict(self.properties),
            "molecule": molecule_to_dict(self.molecule),
        }


def read_sdf(path: str | Path) -> Molecule:
    return read_sdf_record(path).molecule


def read_sdf_record(path: str | Path) -> SDFRecord:
    text = Path(path).read_text(encoding="latin-1")
    return parse_sdf_record(text)


def parse_sdf(text: str) -> Molecule:
    return parse_sdf_record(text).molecule


def parse_sdf_record(text: str) -> SDFRecord:
    records = parse_sdf_records(text)
    if not records:
        raise ValueError("No SDF record found")
    return records[0]


def read_sdf_records(path: str | Path, *, limit: int | None = None) -> list[SDFRecord]:
    return parse_sdf_records(Path(path).read_text(encoding="latin-1"), limit=limit)


def iter_sdf_records(path: str | Path, *, limit: int | None = None):
    if limit is not None and limit <= 0:
        return
    count = 0
    block_lines: list[str] = []
    with Path(path).open("r", encoding="latin-1") as handle:
        for line in handle:
            if line.rstrip("\n\r") == "$$$$":
                block = "".join(block_lines).strip("\n")
                block_lines.clear()
                if not block.strip():
                    continue
                yield _parse_block(block)
                count += 1
                if limit is not None and count >= limit:
                    return
                continue
            block_lines.append(line)
    block = "".join(block_lines).strip("\n")
    if block.strip() and (limit is None or count < limit):
        yield _parse_block(block)


def parse_sdf_records(text: str, *, limit: int | None = None) -> list[SDFRecord]:
    if limit is not None and limit <= 0:
        return []
    records: list[SDFRecord] = []
    for block in _iter_blocks(text):
        stripped = block.strip("\n")
        if not stripped.strip():
            continue
        records.append(_parse_block(stripped))
        if limit is not None and len(records) >= limit:
            break
    return records


def molecule_to_sdf(
    molecule: Molecule,
    *,
    title: str = "",
    properties: Mapping[str, str] | None = None,
) -> str:
    atom_lines = []
    for atom in molecule.atoms.values():
        atom_lines.append(
            f"{atom.coordinate.x.value:10.4f}{atom.coordinate.y.value:10.4f}{atom.coordinate.z.value:10.4f} "
            f"{atom.attributes.symbol.value:<3} 0  0  0  0  0  0  0  0  0  0  0  0"
        )
    bond_lines = [f"{edge.a.value:>3}{edge.b.value:>3}{1:>3}  0  0  0  0" for edge in sorted(molecule.local_bonds)]
    charge_lines = _format_charge_lines(molecule)
    payload = [
        title,
        "",
        "",
        f"{len(molecule.atoms):>3}{len(molecule.local_bonds):>3}  0  0  0  0  0  0  0  0  0  0 V2000",
        *atom_lines,
        *bond_lines,
        *charge_lines,
        "M  END",
    ]
    if properties:
        for key, value in properties.items():
            payload.extend([f"> <{key}>", value, ""])
    payload.append("$$$$")
    return "\n".join(payload)


def _parse_block(block: str) -> SDFRecord:
    lines = block.splitlines()
    if len(lines) < 2:
        raise ValueError("Incomplete SDF block")
    counts_index = next(
        (index for index, line in enumerate(lines) if "V2000" in line or "V3000" in line),
        3 if len(lines) >= 4 else None,
    )
    if counts_index is None:
        raise ValueError("Counts line not found")
    title = lines[0] if lines else ""
    if "V3000" in lines[counts_index]:
        atoms, bonds, properties = _parse_v3000_sections(lines, counts_index)
    else:
        atoms, bonds, properties = _parse_v2000_sections(lines, counts_index)
    molecule = _build_molecule(atoms, bonds)
    return SDFRecord(molecule=molecule, properties=properties, title=title)


def _parse_v2000_sections(lines: list[str], counts_index: int) -> tuple[list[Atom], list[tuple[Edge, int]], dict[str, str]]:
    atom_count, bond_count = _parse_counts_line(lines[counts_index])
    atom_start = counts_index + 1
    atom_lines = lines[atom_start : atom_start + atom_count]
    bond_start = atom_start + atom_count
    bond_lines = lines[bond_start : bond_start + bond_count]
    tail_lines = lines[bond_start + bond_count :]
    atoms = [_parse_atom_line(index + 1, line) for index, line in enumerate(atom_lines)]
    bonds = [_parse_bond_line(line) for line in bond_lines]
    formal_charges: dict[AtomId, int] = {}
    tail_index = 0
    while tail_index < len(tail_lines):
        line = tail_lines[tail_index]
        if line.startswith("M  CHG"):
            for atom_id, charge in _parse_charge_line(line):
                formal_charges[atom_id] = charge
            tail_index += 1
            continue
        if line == "M  END":
            tail_index += 1
            break
        tail_index += 1
    properties = _parse_properties(tail_lines[tail_index:])
    atoms = [
        Atom(
            atom_id=atom.atom_id,
            attributes=atom.attributes,
            coordinate=atom.coordinate,
            shells=atom.shells,
            formal_charge=formal_charges.get(atom.atom_id, atom.formal_charge),
        )
        for atom in atoms
    ]
    return atoms, bonds, properties


def _parse_v3000_sections(lines: list[str], counts_index: int) -> tuple[list[Atom], list[tuple[Edge, int]], dict[str, str]]:
    ctab_lines, tail_index = _collect_v3000_ctab_lines(lines[counts_index + 1 :])
    atoms: list[Atom] = []
    bonds: list[tuple[Edge, int]] = []
    state: str | None = None
    atom_count = 0
    bond_count = 0
    for line in ctab_lines:
        if line == "BEGIN CTAB" or line == "END CTAB":
            continue
        if line.startswith("COUNTS "):
            atom_count, bond_count = _parse_v3000_counts_line(line)
            continue
        if line == "BEGIN ATOM":
            state = "atom"
            continue
        if line == "END ATOM":
            state = None
            continue
        if line == "BEGIN BOND":
            state = "bond"
            continue
        if line == "END BOND":
            state = None
            continue
        if line.startswith("BEGIN ") or line.startswith("END "):
            state = None
            continue
        if state == "atom":
            atoms.append(_parse_v3000_atom_line(line))
        elif state == "bond":
            bonds.append(_parse_v3000_bond_line(line))
    if atom_count and len(atoms) != atom_count:
        raise ValueError(f"V3000 atom count mismatch: expected {atom_count}, got {len(atoms)}")
    if bond_count and len(bonds) != bond_count:
        raise ValueError(f"V3000 bond count mismatch: expected {bond_count}, got {len(bonds)}")
    properties = _parse_properties(lines[counts_index + 1 + tail_index :])
    return atoms, bonds, properties


def _collect_v3000_ctab_lines(lines: list[str]) -> tuple[list[str], int]:
    logical_lines: list[str] = []
    pending: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if line == "M  END":
            if pending is not None:
                raise ValueError("Unterminated V3000 continuation line")
            return logical_lines, index + 1
        if not line.startswith("M  V30 "):
            index += 1
            continue
        payload = line[7:]
        if pending is not None:
            payload = pending + payload.lstrip()
            pending = None
        if payload.endswith("-"):
            pending = payload[:-1].rstrip() + " "
        else:
            logical_lines.append(payload.strip())
        index += 1
    raise ValueError("V3000 M  END line not found")


def _parse_v3000_counts_line(line: str) -> tuple[int, int]:
    words = line.split()
    if len(words) < 3:
        raise ValueError("Invalid V3000 counts line")
    return int(words[1]), int(words[2])


def _parse_v3000_atom_line(line: str) -> Atom:
    words = shlex.split(line)
    if len(words) < 6:
        raise ValueError("Invalid V3000 atom line")
    atom_id = AtomId(int(words[0]))
    symbol = AtomicSymbol(words[1])
    x, y, z = map(float, words[2:5])
    formal_charge = 0
    for token in words[6:]:
        key, value = _split_v3000_token(token)
        if key == "CHG":
            formal_charge = int(value)
    return Atom(
        atom_id=atom_id,
        attributes=element_attributes(symbol),
        coordinate=Coordinate(mk_angstrom(x), mk_angstrom(y), mk_angstrom(z)),
        shells=element_shells(symbol),
        formal_charge=formal_charge,
    )


def _parse_v3000_bond_line(line: str) -> tuple[Edge, int]:
    words = shlex.split(line)
    if len(words) < 4:
        raise ValueError("Invalid V3000 bond line")
    bond_order = int(words[1])
    atom_i = AtomId(int(words[2]))
    atom_j = AtomId(int(words[3]))
    return mk_edge(atom_i, atom_j), bond_order


def _split_v3000_token(token: str) -> tuple[str, str]:
    if "=" not in token:
        return token, ""
    key, value = token.split("=", 1)
    return key, value.strip('"')


def _parse_properties(lines: list[str]) -> dict[str, str]:
    properties: dict[str, str] = {}
    tail_index = 0
    while tail_index < len(lines):
        line = lines[tail_index]
        if line.startswith(">"):
            property_name = _parse_property_name(line)
            tail_index += 1
            property_value_lines: list[str] = []
            while tail_index < len(lines) and lines[tail_index] != "":
                property_value_lines.append(lines[tail_index])
                tail_index += 1
            properties[property_name] = "\n".join(property_value_lines)
        tail_index += 1
    return properties


def _build_molecule(atoms: list[Atom], bonds: list[tuple[Edge, int]]) -> Molecule:
    atom_map = {atom.atom_id: atom for atom in atoms}
    local_bonds = frozenset(edge for edge, _ in bonds)
    aromatic_rings = _detect_six_rings(bonds)
    aromatic_ring_edges = frozenset(edge for ring in aromatic_rings for edge in ring)
    systems: list[tuple[SystemId, BondingSystem]] = []
    system_index = 1
    for edge, order in bonds:
        if order == 2 and edge not in aromatic_ring_edges:
            systems.append((SystemId(system_index), mk_bonding_system(NonNegative(2), frozenset({edge}))))
            system_index += 1
        elif order == 3:
            systems.append((SystemId(system_index), mk_bonding_system(NonNegative(4), frozenset({edge}))))
            system_index += 1
    for ring_edges in aromatic_rings:
        systems.append((SystemId(system_index), mk_bonding_system(NonNegative(6), ring_edges, "pi_ring")))
        system_index += 1
    return Molecule(atoms=atom_map, local_bonds=local_bonds, systems=tuple(systems))


def _parse_atom_line(index: int, line: str) -> Atom:
    words = line.split()
    if len(words) < 4:
        raise ValueError("Invalid atom line")
    x, y, z = map(float, words[:3])
    symbol_text = words[3]
    symbol = AtomicSymbol(symbol_text)
    atom_id = AtomId(index)
    return Atom(
        atom_id=atom_id,
        attributes=element_attributes(symbol),
        coordinate=Coordinate(mk_angstrom(x), mk_angstrom(y), mk_angstrom(z)),
        shells=element_shells(symbol),
        formal_charge=0,
    )


def _parse_bond_line(line: str) -> tuple[Edge, int]:
    words = line.split()
    if len(words) < 3:
        raise ValueError("Invalid bond line")
    atom_i = AtomId(int(words[0]))
    atom_j = AtomId(int(words[1]))
    bond_order = int(words[2])
    return mk_edge(atom_i, atom_j), bond_order


def _parse_charge_line(line: str) -> list[tuple[AtomId, int]]:
    words = line.split()
    if len(words) < 4 or words[0:2] != ["M", "CHG"]:
        raise ValueError("Invalid M  CHG line")
    pair_count = int(words[2])
    pairs = words[3:]
    if len(pairs) < pair_count * 2:
        raise ValueError("Invalid M  CHG pair count")
    result = []
    for index in range(pair_count):
        atom_id = AtomId(int(pairs[index * 2]))
        charge = int(pairs[index * 2 + 1])
        result.append((atom_id, charge))
    return result


def _parse_property_name(line: str) -> str:
    match = search(r"<([^>]+)>", line)
    if match is None:
        raise ValueError(f"Invalid property header: {line}")
    return match.group(1)


def _parse_counts_line(line: str) -> tuple[int, int]:
    fixed_atom = line[0:3].strip()
    fixed_bond = line[3:6].strip()
    if fixed_atom and fixed_bond:
        return int(fixed_atom), int(fixed_bond)
    words = line.split()
    if len(words) < 2:
        raise ValueError("Invalid counts line")
    return int(words[0]), int(words[1])


def _iter_blocks(text: str) -> list[str]:
    if "$$$$" not in text:
        return [text]
    blocks: list[str] = []
    start = 0
    marker = "$$$$"
    marker_len = len(marker)
    while True:
        index = text.find(marker, start)
        if index == -1:
            tail = text[start:]
            if tail.strip():
                blocks.append(tail)
            break
        blocks.append(text[start:index])
        start = index + marker_len
    return blocks


def _format_charge_lines(molecule: Molecule) -> list[str]:
    charged_atoms = [(atom.atom_id, atom.formal_charge) for atom in molecule.atoms.values() if atom.formal_charge != 0]
    if not charged_atoms:
        return []
    lines: list[str] = []
    for chunk_start in range(0, len(charged_atoms), 8):
        chunk = charged_atoms[chunk_start : chunk_start + 8]
        payload = " ".join(f"{atom_id.value:>3} {charge:>3}" for atom_id, charge in chunk)
        lines.append(f"M  CHG {len(chunk):>3} {payload}")
    return lines


def _detect_six_rings(bonds: list[tuple[Edge, int]]) -> list[frozenset[Edge]]:
    adjacency: dict[AtomId, list[tuple[AtomId, int]]] = {}
    for edge, order in bonds:
        adjacency.setdefault(edge.a, []).append((edge.b, order))
        adjacency.setdefault(edge.b, []).append((edge.a, order))

    def alternate(order: int) -> int:
        if order == 1:
            return 2
        if order == 2:
            return 1
        return 0

    discovered: set[frozenset[Edge]] = set()

    def search_alternating(path: list[AtomId], current: AtomId, previous_order: int | None) -> None:
        if len(path) == 6:
            if previous_order is None:
                return
            for neighbor, order in adjacency.get(current, []):
                if neighbor == path[0] and order == alternate(previous_order):
                    atoms = path + [path[0]]
                    ring_edges = frozenset(mk_edge(atoms[index], atoms[index + 1]) for index in range(6))
                    if path[0] == min(path, key=lambda atom_id: atom_id.value):
                        discovered.add(ring_edges)
            return
        for neighbor, order in adjacency.get(current, []):
            if order not in {1, 2}:
                continue
            if previous_order is not None and order != alternate(previous_order):
                continue
            if neighbor in path:
                continue
            search_alternating(path + [neighbor], neighbor, order)

    def search_aromatic(path: list[AtomId], current: AtomId) -> None:
        if len(path) == 6:
            for neighbor, order in adjacency.get(current, []):
                if neighbor == path[0] and order == 4:
                    atoms = path + [path[0]]
                    ring_edges = frozenset(mk_edge(atoms[index], atoms[index + 1]) for index in range(6))
                    if path[0] == min(path, key=lambda atom_id: atom_id.value):
                        discovered.add(ring_edges)
            return
        for neighbor, order in adjacency.get(current, []):
            if order != 4 or neighbor in path:
                continue
            search_aromatic(path + [neighbor], neighbor)

    for start in adjacency:
        search_alternating([start], start, None)
        search_aromatic([start], start)
    return sorted(discovered, key=lambda ring: sorted((edge.a.value, edge.b.value) for edge in ring))
