from __future__ import annotations

from dataclasses import dataclass

from ..chem.constants import element_attributes, element_shells
from ..chem.coordinate import Coordinate, mk_angstrom
from ..chem.dietz import AtomId, BondingSystem, Edge, NonNegative, SystemId, mk_bonding_system, mk_edge
from ..chem.molecule import Atom, AtomicSymbol, Molecule


BondKind = str

_BOND_SYMBOLS: dict[str, BondKind] = {
    "-": "single",
    "=": "double",
    "#": "triple",
    ":": "aromatic",
}

_TWO_CHAR_SYMBOLS: dict[str, AtomicSymbol] = {
    "Br": AtomicSymbol.Br,
    "Cl": AtomicSymbol.Cl,
    "Fe": AtomicSymbol.Fe,
    "Na": AtomicSymbol.Na,
    "Si": AtomicSymbol.Si,
}

_ONE_CHAR_SYMBOLS: dict[str, AtomicSymbol] = {
    "B": AtomicSymbol.B,
    "C": AtomicSymbol.C,
    "F": AtomicSymbol.F,
    "H": AtomicSymbol.H,
    "I": AtomicSymbol.I,
    "N": AtomicSymbol.N,
    "O": AtomicSymbol.O,
    "P": AtomicSymbol.P,
    "S": AtomicSymbol.S,
}

_AROMATIC_SYMBOLS: dict[str, AtomicSymbol] = {
    "b": AtomicSymbol.B,
    "c": AtomicSymbol.C,
    "n": AtomicSymbol.N,
    "o": AtomicSymbol.O,
    "p": AtomicSymbol.P,
    "s": AtomicSymbol.S,
}


@dataclass(frozen=True, slots=True)
class _AtomRef:
    atom_id: AtomId
    aromatic: bool


@dataclass(frozen=True, slots=True)
class _BracketAtom:
    symbol: AtomicSymbol
    aromatic: bool
    hydrogen_count: int
    charge: int


@dataclass(frozen=True, slots=True)
class _RingOpen:
    atom: _AtomRef
    bond_kind: BondKind | None


def parse_smiles(text: str) -> Molecule:
    parser = _SMILESParser(text)
    return parser.parse()


def molecule_to_smiles(molecule: Molecule) -> str:
    rendered_atoms, hydrogen_counts = _collapse_terminal_hydrogens(molecule)
    bond_orders = _render_bond_orders(molecule, rendered_atoms)
    adjacency = _build_render_adjacency(rendered_atoms, bond_orders)
    components = _connected_components(rendered_atoms, adjacency)
    parts = [
        _render_component(component, rendered_atoms, hydrogen_counts, adjacency, bond_orders)
        for component in components
    ]
    return ".".join(parts)


class _SMILESParser:
    def __init__(self, text: str) -> None:
        self.text = text.strip()
        self.index = 0
        self.next_atom_index = 1
        self.atoms: dict[AtomId, Atom] = {}
        self.local_bonds: set[Edge] = set()
        self.systems: list[BondingSystem] = []
        self.aromatic_candidate_edges: set[Edge] = set()
        self.branch_stack: list[_AtomRef] = []
        self.ring_opens: dict[str, _RingOpen] = {}

    def parse(self) -> Molecule:
        if not self.text:
            raise ValueError("SMILES string is empty")
        current: _AtomRef | None = None
        pending_bond: BondKind | None = None

        while self.index < len(self.text):
            char = self.text[self.index]
            if char == "(":
                if current is None:
                    raise ValueError("Branch opened before any atom")
                self.branch_stack.append(current)
                self.index += 1
                continue
            if char == ")":
                if not self.branch_stack:
                    raise ValueError("Unmatched ')'")
                current = self.branch_stack.pop()
                self.index += 1
                continue
            if char == ".":
                current = None
                pending_bond = None
                self.index += 1
                continue
            if char in _BOND_SYMBOLS:
                if pending_bond is not None:
                    raise ValueError("Multiple bond symbols in sequence")
                pending_bond = _BOND_SYMBOLS[char]
                self.index += 1
                continue
            if char.isdigit():
                if current is None:
                    raise ValueError("Ring digit encountered before any atom")
                self._handle_ring_digit(char, current, pending_bond)
                pending_bond = None
                self.index += 1
                continue

            atom = self._parse_atom()
            if current is not None:
                self._connect(current, atom, pending_bond)
            current = atom
            pending_bond = None

        if self.branch_stack:
            raise ValueError("Unclosed branch in SMILES")
        if self.ring_opens:
            raise ValueError("Unclosed ring digit in SMILES")
        if pending_bond is not None:
            raise ValueError("SMILES ended after a bond symbol")

        systems = _normalize_smiles_systems(self.local_bonds, self.systems, self.aromatic_candidate_edges)
        return Molecule(
            atoms=self.atoms,
            local_bonds=frozenset(self.local_bonds),
            systems=tuple((SystemId(index), system) for index, system in enumerate(systems, start=1)),
        )

    def _parse_atom(self) -> _AtomRef:
        if self.text[self.index] == "[":
            return self._parse_bracket_atom()
        return self._parse_bare_atom()

    def _parse_bracket_atom(self) -> _AtomRef:
        close_index = self.text.find("]", self.index + 1)
        if close_index == -1:
            raise ValueError("Unclosed bracket atom")
        content = self.text[self.index + 1 : close_index]
        if not content:
            raise ValueError("Empty bracket atom")
        bracket_atom = _parse_bracket_content(content)
        self.index = close_index + 1
        atom = self._new_atom(bracket_atom.symbol, bracket_atom.charge)
        atom_ref = _AtomRef(atom_id=atom.atom_id, aromatic=bracket_atom.aromatic)
        for _ in range(bracket_atom.hydrogen_count):
            hydrogen = self._new_atom(AtomicSymbol.H, 0)
            self._add_bond(atom.atom_id, hydrogen.atom_id, "single")
        return atom_ref

    def _parse_bare_atom(self) -> _AtomRef:
        token = self.text[self.index : self.index + 2]
        if token in _TWO_CHAR_SYMBOLS:
            self.index += 2
            atom = self._new_atom(_TWO_CHAR_SYMBOLS[token], 0)
            return _AtomRef(atom.atom_id, False)
        char = self.text[self.index]
        if char in _AROMATIC_SYMBOLS:
            self.index += 1
            atom = self._new_atom(_AROMATIC_SYMBOLS[char], 0)
            return _AtomRef(atom.atom_id, True)
        if char in _ONE_CHAR_SYMBOLS:
            self.index += 1
            atom = self._new_atom(_ONE_CHAR_SYMBOLS[char], 0)
            return _AtomRef(atom.atom_id, False)
        raise ValueError(f"Unsupported SMILES atom token at index {self.index}: {char!r}")

    def _new_atom(self, symbol: AtomicSymbol, charge: int) -> Atom:
        atom_id = AtomId(self.next_atom_index)
        self.next_atom_index += 1
        atom = Atom(
            atom_id=atom_id,
            attributes=element_attributes(symbol),
            coordinate=Coordinate(mk_angstrom(float(atom_id.value - 1)), mk_angstrom(0.0), mk_angstrom(0.0)),
            shells=element_shells(symbol),
            formal_charge=charge,
        )
        self.atoms[atom_id] = atom
        return atom

    def _handle_ring_digit(self, digit: str, current: _AtomRef, pending_bond: BondKind | None) -> None:
        if digit not in self.ring_opens:
            self.ring_opens[digit] = _RingOpen(atom=current, bond_kind=pending_bond)
            return
        ring_open = self.ring_opens.pop(digit)
        bond_kind = _resolve_bond_kind(
            ring_open.bond_kind,
            pending_bond,
            ring_open.atom.aromatic,
            current.aromatic,
        )
        self._add_bond(ring_open.atom.atom_id, current.atom_id, bond_kind)

    def _connect(self, left: _AtomRef, right: _AtomRef, pending_bond: BondKind | None) -> None:
        bond_kind = pending_bond if pending_bond is not None else _default_bond_kind(left, right)
        self._add_bond(left.atom_id, right.atom_id, bond_kind)

    def _add_bond(self, left: AtomId, right: AtomId, bond_kind: BondKind) -> None:
        edge = mk_edge(left, right)
        self.local_bonds.add(edge)
        if bond_kind == "double":
            self.systems.append(mk_bonding_system(NonNegative(2), frozenset({edge})))
        elif bond_kind == "triple":
            self.systems.append(mk_bonding_system(NonNegative(4), frozenset({edge})))
        elif bond_kind == "aromatic":
            self.aromatic_candidate_edges.add(edge)
        elif bond_kind != "single":
            raise ValueError(f"Unsupported bond kind: {bond_kind}")


def _parse_bracket_content(content: str) -> _BracketAtom:
    aromatic = False
    symbol_text = ""
    if len(content) >= 2 and content[:2] in _TWO_CHAR_SYMBOLS:
        symbol_text = content[:2]
    elif content[0] in _AROMATIC_SYMBOLS:
        symbol_text = content[0]
        aromatic = True
    elif content[0] in _ONE_CHAR_SYMBOLS:
        symbol_text = content[0]
    elif len(content) >= 2 and content[0].isalpha() and content[1].islower():
        symbol_text = content[:2]
    elif content[0].isalpha():
        symbol_text = content[0]
    else:
        raise ValueError(f"Unsupported bracket atom: [{content}]")

    if aromatic:
        symbol = _AROMATIC_SYMBOLS[symbol_text]
    elif symbol_text in _TWO_CHAR_SYMBOLS:
        symbol = _TWO_CHAR_SYMBOLS[symbol_text]
    elif symbol_text in _ONE_CHAR_SYMBOLS:
        symbol = _ONE_CHAR_SYMBOLS[symbol_text]
    else:
        raise ValueError(f"Unsupported bracket atom symbol: {symbol_text}")

    index = len(symbol_text)
    hydrogen_count = 0
    if index < len(content) and content[index] == "H":
        index += 1
        digits_start = index
        while index < len(content) and content[index].isdigit():
            index += 1
        hydrogen_count = int(content[digits_start:index]) if index > digits_start else 1

    charge = 0
    if index < len(content) and content[index] in "+-":
        sign = 1 if content[index] == "+" else -1
        index += 1
        if index < len(content) and content[index].isdigit():
            digits_start = index
            while index < len(content) and content[index].isdigit():
                index += 1
            charge = sign * int(content[digits_start:index])
        else:
            magnitude = 1
            while index < len(content) and content[index] in "+-":
                if (content[index] == "+") != (sign > 0):
                    raise ValueError(f"Mixed charge syntax in bracket atom: [{content}]")
                magnitude += 1
                index += 1
            charge = sign * magnitude

    if index != len(content):
        raise ValueError(f"Unsupported bracket atom feature: [{content}]")

    return _BracketAtom(
        symbol=symbol,
        aromatic=aromatic,
        hydrogen_count=hydrogen_count,
        charge=charge,
    )


def _default_bond_kind(left: _AtomRef, right: _AtomRef) -> BondKind:
    if left.aromatic and right.aromatic:
        return "aromatic"
    return "single"


def _resolve_bond_kind(
    left: BondKind | None,
    right: BondKind | None,
    left_aromatic: bool,
    right_aromatic: bool,
) -> BondKind:
    if left is not None and right is not None and left != right:
        raise ValueError("Conflicting bond specifications around ring closure")
    if left is not None:
        return left
    if right is not None:
        return right
    if left_aromatic and right_aromatic:
        return "aromatic"
    return "single"


def _normalize_smiles_systems(
    local_bonds: set[Edge],
    systems: list[BondingSystem],
    aromatic_candidate_edges: set[Edge],
) -> list[BondingSystem]:
    pi_rings = {ring for ring in _detect_aromatic_six_rings(aromatic_candidate_edges)}
    double_edges = {
        next(iter(system.member_edges))
        for system in systems
        if len(system.member_edges) == 1 and system.shared_electrons.value == 2
    }
    pi_rings.update(_detect_alternating_six_rings(local_bonds, double_edges))
    ring_edges = {edge for ring in pi_rings for edge in ring}

    normalized = [
        system
        for system in systems
        if not (
            len(system.member_edges) == 1
            and system.shared_electrons.value == 2
            and next(iter(system.member_edges)) in ring_edges
        )
    ]
    normalized.extend(
        mk_bonding_system(NonNegative(6), ring, "pi_ring")
        for ring in sorted(pi_rings, key=_ring_sort_key)
    )
    return normalized


def _detect_alternating_six_rings(local_bonds: set[Edge], double_edges: set[Edge]) -> set[frozenset[Edge]]:
    bonds = [(edge, 2 if edge in double_edges else 1) for edge in sorted(local_bonds)]
    return set(_detect_six_rings_with_orders(bonds))


def _detect_aromatic_six_rings(edges: set[Edge]) -> list[frozenset[Edge]]:
    adjacency: dict[AtomId, list[AtomId]] = {}
    for edge in edges:
        adjacency.setdefault(edge.a, []).append(edge.b)
        adjacency.setdefault(edge.b, []).append(edge.a)
    for atom_id in adjacency:
        adjacency[atom_id].sort()

    discovered: set[frozenset[Edge]] = set()

    def search(path: list[AtomId], current: AtomId) -> None:
        if len(path) == 6:
            for neighbor in adjacency.get(current, []):
                if neighbor == path[0]:
                    ring = frozenset(
                        mk_edge(path[index], path[index + 1] if index < 5 else path[0])
                        for index in range(6)
                    )
                    if path[0] == min(path):
                        discovered.add(ring)
            return
        for neighbor in adjacency.get(current, []):
            if neighbor in path:
                continue
            search(path + [neighbor], neighbor)

    for start in sorted(adjacency):
        search([start], start)
    return sorted(discovered, key=_ring_sort_key)


def _detect_six_rings_with_orders(bonds: list[tuple[Edge, int]]) -> list[frozenset[Edge]]:
    adjacency: dict[AtomId, list[tuple[AtomId, int]]] = {}
    for edge, order in bonds:
        adjacency.setdefault(edge.a, []).append((edge.b, order))
        adjacency.setdefault(edge.b, []).append((edge.a, order))
    for atom_id in adjacency:
        adjacency[atom_id].sort(key=lambda item: item[0].value)

    def alternate(order: int) -> int:
        if order == 1:
            return 2
        if order == 2:
            return 1
        return 0

    discovered: set[frozenset[Edge]] = set()

    def search(path: list[AtomId], current: AtomId, previous_order: int | None) -> None:
        if len(path) == 6:
            if previous_order is None:
                return
            for neighbor, order in adjacency.get(current, []):
                if neighbor == path[0] and order == alternate(previous_order):
                    atoms = path + [path[0]]
                    ring_edges = frozenset(mk_edge(atoms[index], atoms[index + 1]) for index in range(6))
                    if path[0] == min(path):
                        discovered.add(ring_edges)
            return
        for neighbor, order in adjacency.get(current, []):
            if order not in {1, 2}:
                continue
            if previous_order is not None and order != alternate(previous_order):
                continue
            if neighbor in path:
                continue
            search(path + [neighbor], neighbor, order)

    for start in sorted(adjacency):
        search([start], start, None)
    return sorted(discovered, key=_ring_sort_key)


def _ring_sort_key(ring: frozenset[Edge]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((edge.a.value, edge.b.value) for edge in ring))


def _collapse_terminal_hydrogens(molecule: Molecule) -> tuple[dict[AtomId, Atom], dict[AtomId, int]]:
    hydrogen_counts = {atom_id: 0 for atom_id in molecule.atoms}
    system_atoms = {atom_id for _, system in molecule.systems for atom_id in system.member_atoms}
    suppressed: set[AtomId] = set()

    for atom_id, atom in molecule.atoms.items():
        if atom.attributes.symbol is not AtomicSymbol.H:
            continue
        if atom.formal_charge != 0:
            continue
        if atom_id in system_atoms:
            continue
        incident = [edge for edge in molecule.local_bonds if edge.a == atom_id or edge.b == atom_id]
        if len(incident) != 1:
            continue
        edge = incident[0]
        host = edge.b if edge.a == atom_id else edge.a
        host_atom = molecule.atoms[host]
        if host_atom.attributes.symbol is AtomicSymbol.H:
            continue
        if host_atom.formal_charge != 0:
            continue
        suppressed.add(atom_id)
        hydrogen_counts[host] += 1

    rendered_atoms = {
        atom_id: atom
        for atom_id, atom in molecule.atoms.items()
        if atom_id not in suppressed
    }
    return rendered_atoms, hydrogen_counts


def _render_bond_orders(molecule: Molecule, rendered_atoms: dict[AtomId, Atom]) -> dict[Edge, int]:
    rendered_ids = frozenset(rendered_atoms)
    bond_orders = {
        edge: 1
        for edge in molecule.local_bonds
        if edge.a in rendered_ids and edge.b in rendered_ids
    }

    pi_rings: list[frozenset[Edge]] = []
    for _, system in molecule.systems:
        if system.tag == "pi_ring" and system.shared_electrons.value == 6 and len(system.member_edges) == 6:
            pi_rings.append(system.member_edges)
            continue
        if len(system.member_edges) == 1 and system.shared_electrons.value in {2, 4}:
            edge = next(iter(system.member_edges))
            if edge not in bond_orders:
                raise ValueError("SMILES rendering requires all bonded atoms to remain in the output graph")
            bond_orders[edge] = 1 + (system.shared_electrons.value // 2)
            continue
        raise ValueError("SMILES rendering only supports localized double/triple bonds and six-edge pi rings")

    for ring in pi_rings:
        cycle = _ordered_cycle(ring)
        for index in range(6):
            edge = mk_edge(cycle[index], cycle[(index + 1) % 6])
            bond_orders[edge] = 2 if index % 2 == 0 else 1
    return bond_orders


def _ordered_cycle(ring: frozenset[Edge]) -> list[AtomId]:
    adjacency: dict[AtomId, list[AtomId]] = {}
    for edge in ring:
        adjacency.setdefault(edge.a, []).append(edge.b)
        adjacency.setdefault(edge.b, []).append(edge.a)
    if len(adjacency) != 6 or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("pi_ring must be a simple six-membered cycle to render as SMILES")

    start = min(adjacency)
    paths: list[list[AtomId]] = []
    for neighbor in sorted(adjacency[start]):
        path = [start, neighbor]
        previous = start
        current = neighbor
        while True:
            next_candidates = sorted(candidate for candidate in adjacency[current] if candidate != previous)
            if not next_candidates:
                raise ValueError("Failed to order pi_ring edges")
            next_atom = next_candidates[0]
            if next_atom == start:
                break
            path.append(next_atom)
            previous, current = current, next_atom
            if len(path) > 6:
                raise ValueError("Invalid pi_ring cycle length")
        if len(path) == 6:
            paths.append(path)
    if not paths:
        raise ValueError("Failed to derive a six-membered cycle for pi_ring")
    return min(paths, key=lambda path: tuple(atom.value for atom in path))


def _build_render_adjacency(rendered_atoms: dict[AtomId, Atom], bond_orders: dict[Edge, int]) -> dict[AtomId, list[AtomId]]:
    adjacency: dict[AtomId, list[AtomId]] = {atom_id: [] for atom_id in rendered_atoms}
    for edge in bond_orders:
        adjacency[edge.a].append(edge.b)
        adjacency[edge.b].append(edge.a)
    for atom_id in adjacency:
        adjacency[atom_id].sort()
    return adjacency


def _connected_components(
    rendered_atoms: dict[AtomId, Atom],
    adjacency: dict[AtomId, list[AtomId]],
) -> list[list[AtomId]]:
    remaining = set(rendered_atoms)
    components: list[list[AtomId]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: list[AtomId] = []
        while stack:
            atom_id = stack.pop()
            if atom_id not in remaining:
                continue
            remaining.remove(atom_id)
            component.append(atom_id)
            for neighbor in reversed(adjacency.get(atom_id, [])):
                if neighbor in remaining:
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _render_component(
    component: list[AtomId],
    rendered_atoms: dict[AtomId, Atom],
    hydrogen_counts: dict[AtomId, int],
    adjacency: dict[AtomId, list[AtomId]],
    bond_orders: dict[Edge, int],
) -> str:
    root = component[0]
    component_set = set(component)
    tree_edges: set[Edge] = set()
    discovery: list[AtomId] = []
    visited: set[AtomId] = set()

    def build_tree(atom_id: AtomId, parent: AtomId | None) -> None:
        visited.add(atom_id)
        discovery.append(atom_id)
        for neighbor in adjacency[atom_id]:
            if neighbor not in component_set or neighbor == parent:
                continue
            edge = mk_edge(atom_id, neighbor)
            if neighbor not in visited:
                tree_edges.add(edge)
                build_tree(neighbor, atom_id)

    build_tree(root, None)
    discovery_index = {atom_id: index for index, atom_id in enumerate(discovery)}
    ring_edges = sorted(
        [
            edge
            for edge in bond_orders
            if edge.a in component_set and edge.b in component_set and edge not in tree_edges
        ],
        key=lambda edge: (min(discovery_index[edge.a], discovery_index[edge.b]), edge.a.value, edge.b.value),
    )
    if len(ring_edges) > 9:
        raise ValueError("SMILES rendering currently supports at most 9 ring closures per component")

    ring_starts: dict[AtomId, list[tuple[int, int]]] = {}
    ring_ends: dict[AtomId, list[int]] = {}
    for digit, edge in enumerate(ring_edges, start=1):
        first, second = (edge.a, edge.b)
        if discovery_index[first] > discovery_index[second]:
            first, second = second, first
        ring_starts.setdefault(first, []).append((digit, bond_orders[edge]))
        ring_ends.setdefault(second, []).append(digit)

    def render(atom_id: AtomId, parent: AtomId | None) -> str:
        pieces = [_render_atom_label(rendered_atoms[atom_id], hydrogen_counts.get(atom_id, 0))]
        for digit, order in sorted(ring_starts.get(atom_id, [])):
            pieces.append(_bond_symbol(order))
            pieces.append(str(digit))
        for digit in sorted(ring_ends.get(atom_id, [])):
            pieces.append(str(digit))

        children = [
            neighbor
            for neighbor in adjacency[atom_id]
            if mk_edge(atom_id, neighbor) in tree_edges and neighbor != parent
        ]
        children.sort()
        for child_index, child in enumerate(children):
            edge = mk_edge(atom_id, child)
            child_text = _bond_symbol(bond_orders[edge]) + render(child, atom_id)
            if child_index == 0:
                pieces.append(child_text)
            else:
                pieces.append(f"({child_text})")
        return "".join(pieces)

    return render(root, None)


def _render_atom_label(atom: Atom, hydrogen_count: int) -> str:
    symbol = atom.attributes.symbol.value
    hydrogen_part = ""
    if hydrogen_count == 1:
        hydrogen_part = "H"
    elif hydrogen_count > 1:
        hydrogen_part = f"H{hydrogen_count}"

    charge_part = ""
    if atom.formal_charge == 1:
        charge_part = "+"
    elif atom.formal_charge > 1:
        charge_part = f"+{atom.formal_charge}"
    elif atom.formal_charge == -1:
        charge_part = "-"
    elif atom.formal_charge < -1:
        charge_part = f"-{abs(atom.formal_charge)}"
    return f"[{symbol}{hydrogen_part}{charge_part}]"


def _bond_symbol(order: int) -> str:
    if order == 1:
        return ""
    if order == 2:
        return "="
    if order == 3:
        return "#"
    raise ValueError(f"Unsupported SMILES bond order: {order}")
