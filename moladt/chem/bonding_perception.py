from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import sqrt

from .dietz import AtomId, Edge, mk_edge
from .molecule import Atom, AtomicSymbol


@dataclass(frozen=True, slots=True)
class InferredBondingSystem:
    shared_electrons: int
    member_edges: frozenset[Edge]
    tag: str


@dataclass(frozen=True, slots=True)
class BondingPerception:
    inferred_systems: tuple[InferredBondingSystem, ...]
    sigma_override_edges: frozenset[Edge]
    suppressed_edges: frozenset[Edge]


def perceive_sdf_bonding(
    atoms: Mapping[AtomId, Atom] | Sequence[Atom],
    bonds: Sequence[tuple[Edge, int]],
) -> BondingPerception:
    """Infer non-local bonding systems from conservative SDF evidence.

    These rules are parser perception, not source-certified electronic truth.
    They add common delocalised or multicentre systems when the SDF bond table
    and coordinates provide strong enough evidence.
    """

    atom_map = _atom_mapping(atoms)
    bond_orders = {edge: order for edge, order in bonds}
    adjacency = _adjacency(bonds)
    systems: list[InferredBondingSystem] = []
    sigma_override_edges: set[Edge] = set()
    suppressed_edges: set[Edge] = set()
    seen: set[tuple[frozenset[Edge], str]] = set()

    def add_system(
        shared_electrons: int,
        edges: set[Edge] | frozenset[Edge],
        tag: str,
        *,
        sigma_override: bool = True,
    ) -> None:
        member_edges = frozenset(edges)
        if not member_edges:
            return
        signature = (member_edges, tag)
        if signature in seen:
            return
        seen.add(signature)
        systems.append(InferredBondingSystem(shared_electrons, member_edges, tag))
        if sigma_override:
            sigma_override_edges.update(member_edges)

    for ring in _detect_aromatic_six_rings(bonds):
        add_system(6, ring, "pi_ring")

    for tag, edges in _detect_two_edge_oxo_resonance(atom_map, adjacency, bond_orders):
        add_system(2, edges, tag)

    for edges in _detect_amide_resonance(atom_map, adjacency, bond_orders):
        add_system(2, edges, "inferred_amide_pi")

    occupied_edges = set(sigma_override_edges)
    for edges in _detect_conjugated_diene_paths(atom_map, adjacency, bond_orders):
        if not edges.isdisjoint(occupied_edges):
            continue
        add_system(4, edges, "inferred_conjugated_diene_pi")
        occupied_edges.update(edges)

    for hydrogen_id, boron_edges, bb_edge in _detect_borane_bridges(atom_map, bond_orders):
        add_system(2, boron_edges, f"inferred_borane_bridge_h{hydrogen_id.value}_3c2e", sigma_override=False)
        suppressed_edges.update(boron_edges)
        if bb_edge is not None:
            suppressed_edges.add(bb_edge)

    for index, cp_edges in enumerate(_detect_ferrocene_cp_systems(atom_map, adjacency, bond_orders), start=1):
        add_system(6, cp_edges, f"inferred_cp{index}_pi", sigma_override=False)

    return BondingPerception(
        inferred_systems=tuple(systems),
        sigma_override_edges=frozenset(sigma_override_edges),
        suppressed_edges=frozenset(suppressed_edges),
    )


def _atom_mapping(atoms: Mapping[AtomId, Atom] | Sequence[Atom]) -> Mapping[AtomId, Atom]:
    if isinstance(atoms, Mapping):
        return atoms
    return {atom.atom_id: atom for atom in atoms}


def _adjacency(bonds: Sequence[tuple[Edge, int]]) -> dict[AtomId, list[tuple[AtomId, int]]]:
    adjacency: dict[AtomId, list[tuple[AtomId, int]]] = {}
    for edge, order in bonds:
        adjacency.setdefault(edge.a, []).append((edge.b, order))
        adjacency.setdefault(edge.b, []).append((edge.a, order))
    return adjacency


def _symbol(atom_map: Mapping[AtomId, Atom], atom_id: AtomId) -> AtomicSymbol:
    return atom_map[atom_id].attributes.symbol


def _distance(left: Atom, right: Atom) -> float:
    dx = left.coordinate.x.value - right.coordinate.x.value
    dy = left.coordinate.y.value - right.coordinate.y.value
    dz = left.coordinate.z.value - right.coordinate.z.value
    return sqrt(dx * dx + dy * dy + dz * dz)


def _detect_aromatic_six_rings(bonds: Sequence[tuple[Edge, int]]) -> tuple[frozenset[Edge], ...]:
    adjacency = _adjacency(bonds)
    discovered: set[frozenset[Edge]] = set()

    def alternate(order: int) -> int:
        if order == 1:
            return 2
        if order == 2:
            return 1
        return 0

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
    return tuple(sorted(discovered, key=_edge_set_sort_key))


def _detect_two_edge_oxo_resonance(
    atom_map: Mapping[AtomId, Atom],
    adjacency: Mapping[AtomId, Sequence[tuple[AtomId, int]]],
    bond_orders: Mapping[Edge, int],
) -> tuple[tuple[str, frozenset[Edge]], ...]:
    systems: list[tuple[str, frozenset[Edge]]] = []
    for center_id, atom in atom_map.items():
        center_symbol = atom.attributes.symbol
        if center_symbol not in {AtomicSymbol.C, AtomicSymbol.N, AtomicSymbol.P, AtomicSymbol.S}:
            continue
        oxygen_edges = [
            (neighbor, mk_edge(center_id, neighbor), bond_orders.get(mk_edge(center_id, neighbor), 1))
            for neighbor, _ in adjacency.get(center_id, ())
            if _symbol(atom_map, neighbor) == AtomicSymbol.O
        ]
        if len(oxygen_edges) < 2:
            continue
        for left, right in combinations(oxygen_edges, 2):
            orders = sorted((left[2], right[2]))
            if orders[0] > 1 or orders[1] < 2:
                continue
            tag = {
                AtomicSymbol.C: "inferred_carboxylate_pi",
                AtomicSymbol.N: "inferred_nitro_pi",
                AtomicSymbol.P: "inferred_oxoanion_pi",
                AtomicSymbol.S: "inferred_oxoanion_pi",
            }[center_symbol]
            systems.append((tag, frozenset({left[1], right[1]})))
    return tuple(sorted(systems, key=lambda item: (item[0], _edge_set_sort_key(item[1]))))


def _detect_amide_resonance(
    atom_map: Mapping[AtomId, Atom],
    adjacency: Mapping[AtomId, Sequence[tuple[AtomId, int]]],
    bond_orders: Mapping[Edge, int],
) -> tuple[frozenset[Edge], ...]:
    systems: set[frozenset[Edge]] = set()
    for center_id, atom in atom_map.items():
        if atom.attributes.symbol != AtomicSymbol.C:
            continue
        oxygen_edges = [
            mk_edge(center_id, neighbor)
            for neighbor, _ in adjacency.get(center_id, ())
            if _symbol(atom_map, neighbor) == AtomicSymbol.O and bond_orders.get(mk_edge(center_id, neighbor), 1) >= 2
        ]
        nitrogen_edges = [
            mk_edge(center_id, neighbor)
            for neighbor, _ in adjacency.get(center_id, ())
            if _symbol(atom_map, neighbor) == AtomicSymbol.N and bond_orders.get(mk_edge(center_id, neighbor), 1) == 1
        ]
        for oxygen_edge in oxygen_edges:
            for nitrogen_edge in nitrogen_edges:
                systems.add(frozenset({oxygen_edge, nitrogen_edge}))
    return tuple(sorted(systems, key=_edge_set_sort_key))


def _detect_conjugated_diene_paths(
    atom_map: Mapping[AtomId, Atom],
    adjacency: Mapping[AtomId, Sequence[tuple[AtomId, int]]],
    bond_orders: Mapping[Edge, int],
) -> tuple[frozenset[Edge], ...]:
    allowed = {AtomicSymbol.C, AtomicSymbol.N, AtomicSymbol.O, AtomicSymbol.S, AtomicSymbol.P}
    systems: set[frozenset[Edge]] = set()
    for start in atom_map:
        stack = [(start, [start])]
        while stack:
            current, path = stack.pop()
            if len(path) == 4:
                if any(_symbol(atom_map, atom_id) not in allowed for atom_id in path):
                    continue
                edges = [mk_edge(path[index], path[index + 1]) for index in range(3)]
                if [bond_orders.get(edge, 1) for edge in edges] == [2, 1, 2]:
                    systems.add(frozenset(edges))
                continue
            for neighbor, _ in adjacency.get(current, ()):
                if neighbor in path:
                    continue
                stack.append((neighbor, path + [neighbor]))
    return tuple(sorted(systems, key=_edge_set_sort_key))


def _detect_borane_bridges(
    atom_map: Mapping[AtomId, Atom],
    bond_orders: Mapping[Edge, int],
) -> tuple[tuple[AtomId, frozenset[Edge], Edge | None], ...]:
    boron_ids = [atom_id for atom_id, atom in atom_map.items() if atom.attributes.symbol == AtomicSymbol.B]
    hydrogen_ids = [atom_id for atom_id, atom in atom_map.items() if atom.attributes.symbol == AtomicSymbol.H]
    bridges: list[tuple[AtomId, frozenset[Edge], Edge | None]] = []
    for left_id, right_id in combinations(sorted(boron_ids), 2):
        left = atom_map[left_id]
        right = atom_map[right_id]
        bb_edge = mk_edge(left_id, right_id)
        if bb_edge not in bond_orders and _distance(left, right) > 2.05:
            continue
        for hydrogen_id in hydrogen_ids:
            hydrogen = atom_map[hydrogen_id]
            if _distance(left, hydrogen) > 1.45 or _distance(right, hydrogen) > 1.45:
                continue
            bridge_edges = frozenset({mk_edge(left_id, hydrogen_id), mk_edge(right_id, hydrogen_id)})
            bridges.append((hydrogen_id, bridge_edges, bb_edge if bb_edge in bond_orders else None))
    return tuple(sorted(bridges, key=lambda item: item[0].value))


def _detect_ferrocene_cp_systems(
    atom_map: Mapping[AtomId, Atom],
    adjacency: Mapping[AtomId, Sequence[tuple[AtomId, int]]],
    bond_orders: Mapping[Edge, int],
) -> tuple[frozenset[Edge], ...]:
    iron_ids = [atom_id for atom_id, atom in atom_map.items() if atom.attributes.symbol == AtomicSymbol.Fe]
    if not iron_ids:
        return ()
    carbon_cycles = _detect_carbon_five_cycles(atom_map, adjacency)
    systems: list[frozenset[Edge]] = []
    for iron_id in sorted(iron_ids):
        iron = atom_map[iron_id]
        for cycle in carbon_cycles:
            if max(_distance(iron, atom_map[atom_id]) for atom_id in cycle) > 2.45:
                continue
            ring_edges = {mk_edge(cycle[index], cycle[(index + 1) % 5]) for index in range(5)}
            if any(bond_orders.get(edge, 1) > 2 for edge in ring_edges):
                continue
            iron_edges = {mk_edge(iron_id, atom_id) for atom_id in cycle}
            systems.append(frozenset(ring_edges | iron_edges))
    return tuple(sorted(systems, key=_edge_set_sort_key))


def _detect_carbon_five_cycles(
    atom_map: Mapping[AtomId, Atom],
    adjacency: Mapping[AtomId, Sequence[tuple[AtomId, int]]],
) -> tuple[tuple[AtomId, ...], ...]:
    cycles: set[tuple[AtomId, ...]] = set()

    def search(start: AtomId, current: AtomId, path: list[AtomId]) -> None:
        if len(path) == 5:
            if any(neighbor == start for neighbor, _ in adjacency.get(current, ())):
                if all(_symbol(atom_map, atom_id) == AtomicSymbol.C for atom_id in path):
                    ordered = tuple(path)
                    rotations = [ordered[index:] + ordered[:index] for index in range(5)]
                    reversed_ordered = tuple(reversed(ordered))
                    rotations += [reversed_ordered[index:] + reversed_ordered[:index] for index in range(5)]
                    cycles.add(min(rotations, key=lambda values: tuple(atom_id.value for atom_id in values)))
            return
        for neighbor, _ in adjacency.get(current, ()):
            if neighbor in path:
                continue
            search(start, neighbor, path + [neighbor])

    for start in sorted(atom_map):
        search(start, start, [start])
    return tuple(sorted(cycles, key=lambda values: tuple(atom_id.value for atom_id in values)))


def _edge_set_sort_key(edges: frozenset[Edge]) -> tuple[tuple[int, int], ...]:
    return tuple((edge.a.value, edge.b.value) for edge in sorted(edges))
