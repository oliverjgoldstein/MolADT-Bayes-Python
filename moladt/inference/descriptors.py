from __future__ import annotations

from dataclasses import dataclass
from math import log, pi

from ..chem.dietz import AtomId, Edge
from ..chem.molecule import Atom, AtomicSymbol, Molecule, effective_order


@dataclass(frozen=True, slots=True)
class MolecularDescriptors:
    weight: float
    polar: float
    surface: float
    bond_order: float
    heavy_atoms: float
    halogens: float
    aromatic_rings: float
    aromatic_atom_fraction: float
    rotatable_bonds: float

    def to_dict(self) -> dict[str, float]:
        return {
            "weight": self.weight,
            "polar": self.polar,
            "surface": self.surface,
            "bond_order": self.bond_order,
            "heavy_atoms": self.heavy_atoms,
            "halogens": self.halogens,
            "aromatic_rings": self.aromatic_rings,
            "aromatic_atom_fraction": self.aromatic_atom_fraction,
            "rotatable_bonds": self.rotatable_bonds,
        }


def atom_list(molecule: Molecule) -> list[Atom]:
    return list(molecule.atoms.values())


def bool_to_float(value: bool) -> float:
    return 1.0 if value else 0.0


def molecule_size(molecule: Molecule) -> float:
    return float(len(molecule.atoms))


def molecule_weight(molecule: Molecule) -> float:
    return sum(atom.attributes.atomic_weight for atom in atom_list(molecule))


def molecule_surface_area(molecule: Molecule) -> float:
    size = molecule_size(molecule)
    return float(4.0 * pi * (size ** (2.0 / 3.0)))


def hetero_atom_count(molecule: Molecule) -> float:
    return sum(bool_to_float(atom.attributes.symbol not in {AtomicSymbol.C, AtomicSymbol.H}) for atom in atom_list(molecule))


def molecule_bond_order(molecule: Molecule) -> float:
    edge_set = set(molecule.local_bonds)
    for _, system in molecule.systems:
        edge_set.update(system.member_edges)
    return sum(effective_order(molecule, edge) for edge in edge_set)


def hydrogen_bond_acceptor_count(molecule: Molecule) -> float:
    return sum(bool_to_float(atom.attributes.symbol in {AtomicSymbol.N, AtomicSymbol.O, AtomicSymbol.S}) for atom in atom_list(molecule))


def hydrogen_bond_donor_count(molecule: Molecule) -> float:
    total = 0.0
    for atom in atom_list(molecule):
        hetero = atom.attributes.symbol in {AtomicSymbol.N, AtomicSymbol.O, AtomicSymbol.S}
        has_hydrogen_neighbor = any(molecule.atoms[neighbor].attributes.symbol == AtomicSymbol.H for neighbor in _neighbors_sigma(molecule, atom.atom_id))
        total += bool_to_float(hetero and has_hydrogen_neighbor)
    return total


def heavy_atom_count(molecule: Molecule) -> float:
    return sum(bool_to_float(atom.attributes.symbol != AtomicSymbol.H) for atom in atom_list(molecule))


def halogen_atom_count(molecule: Molecule) -> float:
    return sum(bool_to_float(atom.attributes.symbol in {AtomicSymbol.F, AtomicSymbol.Cl, AtomicSymbol.Br, AtomicSymbol.I}) for atom in atom_list(molecule))


def aromatic_ring_count(molecule: Molecule) -> float:
    return float(sum(1 for _, system in molecule.systems if system.tag == "pi_ring"))


def aromatic_atom_fraction(molecule: Molecule) -> float:
    aromatic_atoms: set[AtomId] = set()
    for _, system in molecule.systems:
        if system.tag == "pi_ring":
            aromatic_atoms.update(system.member_atoms)
    heavy = heavy_atom_count(molecule)
    if heavy <= 0.0:
        return 0.0
    return len(aromatic_atoms) / heavy


def build_adjacency(molecule: Molecule) -> dict[AtomId, list[AtomId]]:
    adjacency: dict[AtomId, list[AtomId]] = {}
    for edge in molecule.local_bonds:
        adjacency.setdefault(edge.a, []).append(edge.b)
        adjacency.setdefault(edge.b, []).append(edge.a)
    return adjacency


def has_path(adjacency: dict[AtomId, list[AtomId]], start: AtomId, target: AtomId) -> bool:
    visited: set[AtomId] = set()
    queue = [start]
    while queue:
        current = queue.pop(0)
        if current == target:
            return True
        if current in visited:
            continue
        visited.add(current)
        queue.extend(neighbor for neighbor in adjacency.get(current, []) if neighbor not in visited)
    return False


def edge_in_cycle(adjacency: dict[AtomId, list[AtomId]], edge: Edge) -> bool:
    reduced = {node: list(neighbors) for node, neighbors in adjacency.items()}
    reduced[edge.a] = [neighbor for neighbor in reduced.get(edge.a, []) if neighbor != edge.b]
    reduced[edge.b] = [neighbor for neighbor in reduced.get(edge.b, []) if neighbor != edge.a]
    return has_path(reduced, edge.a, edge.b)


def rotatable_bond_count(molecule: Molecule) -> float:
    adjacency = build_adjacency(molecule)

    def is_heavy(atom_id: AtomId) -> bool:
        return molecule.atoms[atom_id].attributes.symbol != AtomicSymbol.H

    def heavy_degree(atom_id: AtomId) -> int:
        return sum(1 for neighbor in adjacency.get(atom_id, []) if is_heavy(neighbor))

    total = 0.0
    for edge in molecule.local_bonds:
        both_heavy = is_heavy(edge.a) and is_heavy(edge.b)
        not_terminal = heavy_degree(edge.a) > 1 and heavy_degree(edge.b) > 1
        single_bond = effective_order(molecule, edge) <= 1.1
        not_ring = not edge_in_cycle(adjacency, edge)
        total += bool_to_float(both_heavy and not_terminal and single_bond and not_ring)
    return total


def log1p_positive(value: float) -> float:
    return log(1.0 + max(0.0, value))


def compute_descriptors(molecule: Molecule) -> MolecularDescriptors:
    return MolecularDescriptors(
        weight=molecule_weight(molecule),
        polar=hetero_atom_count(molecule) + hydrogen_bond_donor_count(molecule) + hydrogen_bond_acceptor_count(molecule),
        surface=molecule_surface_area(molecule),
        bond_order=molecule_bond_order(molecule),
        heavy_atoms=heavy_atom_count(molecule),
        halogens=halogen_atom_count(molecule),
        aromatic_rings=aromatic_ring_count(molecule),
        aromatic_atom_fraction=aromatic_atom_fraction(molecule),
        rotatable_bonds=rotatable_bond_count(molecule),
    )


def _neighbors_sigma(molecule: Molecule, atom_id: AtomId) -> tuple[AtomId, ...]:
    neighbors = []
    for edge in molecule.local_bonds:
        if edge.a == atom_id:
            neighbors.append(edge.b)
        elif edge.b == atom_id:
            neighbors.append(edge.a)
    return tuple(sorted(neighbors))
