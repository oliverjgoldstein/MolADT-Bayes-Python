from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt

from ..chem.dietz import AtomId, Edge
from ..chem.molecule import Atom, AtomicSymbol, Molecule
from ..chem.molecule_ops import effective_order


@dataclass(frozen=True, slots=True)
class MolecularDescriptors:
    weight: float
    polar: float
    surface: float
    bond_order: float
    donor_count: float
    acceptor_count: float
    heavy_atoms: float
    halogens: float
    atom_count_c: float
    atom_count_n: float
    atom_count_o: float
    atom_count_f: float
    atom_count_p: float
    atom_count_s: float
    atom_count_cl: float
    atom_count_br: float
    atom_count_i: float
    atom_count_h: float
    formal_charge_sum: float
    abs_formal_charge_sum: float
    positive_charge_count: float
    negative_charge_count: float
    bonding_system_count: float
    multicentre_system_count: float
    pi_ring_system_count: float
    zero_electron_system_count: float
    sigma_edge_count: float
    effective_bond_order_sum: float
    effective_bond_order_mean: float
    effective_bond_order_max: float
    aromatic_rings: float
    aromatic_atom_count: float
    aromatic_atom_fraction: float
    ring_edge_fraction: float
    rotatable_bonds: float
    heavy_atom_degree_mean: float
    heavy_atom_degree_max: float

    def to_dict(self) -> dict[str, float]:
        return {
            "weight": self.weight,
            "polar": self.polar,
            "surface": self.surface,
            "bond_order": self.bond_order,
            "donor_count": self.donor_count,
            "acceptor_count": self.acceptor_count,
            "heavy_atoms": self.heavy_atoms,
            "halogens": self.halogens,
            "atom_count_c": self.atom_count_c,
            "atom_count_n": self.atom_count_n,
            "atom_count_o": self.atom_count_o,
            "atom_count_f": self.atom_count_f,
            "atom_count_p": self.atom_count_p,
            "atom_count_s": self.atom_count_s,
            "atom_count_cl": self.atom_count_cl,
            "atom_count_br": self.atom_count_br,
            "atom_count_i": self.atom_count_i,
            "atom_count_h": self.atom_count_h,
            "formal_charge_sum": self.formal_charge_sum,
            "abs_formal_charge_sum": self.abs_formal_charge_sum,
            "positive_charge_count": self.positive_charge_count,
            "negative_charge_count": self.negative_charge_count,
            "bonding_system_count": self.bonding_system_count,
            "multicentre_system_count": self.multicentre_system_count,
            "pi_ring_system_count": self.pi_ring_system_count,
            "zero_electron_system_count": self.zero_electron_system_count,
            "sigma_edge_count": self.sigma_edge_count,
            "effective_bond_order_sum": self.effective_bond_order_sum,
            "effective_bond_order_mean": self.effective_bond_order_mean,
            "effective_bond_order_max": self.effective_bond_order_max,
            "aromatic_rings": self.aromatic_rings,
            "aromatic_atom_count": self.aromatic_atom_count,
            "aromatic_atom_fraction": self.aromatic_atom_fraction,
            "ring_edge_fraction": self.ring_edge_fraction,
            "rotatable_bonds": self.rotatable_bonds,
            "heavy_atom_degree_mean": self.heavy_atom_degree_mean,
            "heavy_atom_degree_max": self.heavy_atom_degree_max,
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


def aromatic_atom_count(molecule: Molecule) -> float:
    aromatic_atoms: set[AtomId] = set()
    for _, system in molecule.systems:
        if system.tag == "pi_ring":
            aromatic_atoms.update(system.member_atoms)
    return float(len(aromatic_atoms))


def aromatic_atom_fraction(molecule: Molecule) -> float:
    heavy = heavy_atom_count(molecule)
    if heavy <= 0.0:
        return 0.0
    return aromatic_atom_count(molecule) / heavy


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


def atom_count_by_symbol(molecule: Molecule, symbol: AtomicSymbol) -> float:
    return sum(bool_to_float(atom.attributes.symbol == symbol) for atom in atom_list(molecule))


def formal_charge_sum(molecule: Molecule) -> float:
    return float(sum(atom.formal_charge for atom in atom_list(molecule)))


def abs_formal_charge_sum(molecule: Molecule) -> float:
    return float(sum(abs(atom.formal_charge) for atom in atom_list(molecule)))


def positive_charge_count(molecule: Molecule) -> float:
    return sum(bool_to_float(atom.formal_charge > 0) for atom in atom_list(molecule))


def negative_charge_count(molecule: Molecule) -> float:
    return sum(bool_to_float(atom.formal_charge < 0) for atom in atom_list(molecule))


def bonding_system_count(molecule: Molecule) -> float:
    return float(len(molecule.systems))


def multicentre_system_count(molecule: Molecule) -> float:
    return float(sum(1 for _, system in molecule.systems if len(system.member_edges) > 1))


def zero_electron_system_count(molecule: Molecule) -> float:
    return float(sum(1 for _, system in molecule.systems if system.shared_electrons.value == 0))


def sigma_edge_count(molecule: Molecule) -> float:
    return float(len(molecule.local_bonds))


def unique_bond_edges(molecule: Molecule) -> tuple[Edge, ...]:
    edges = set(molecule.local_bonds)
    for _, system in molecule.systems:
        edges.update(system.member_edges)
    return tuple(sorted(edges))


def effective_bond_order_summary(molecule: Molecule) -> dict[str, float]:
    values = [effective_order(molecule, edge) for edge in unique_bond_edges(molecule)]
    if not values:
        return {
            "effective_bond_order_sum": 0.0,
            "effective_bond_order_mean": 0.0,
            "effective_bond_order_max": 0.0,
        }
    return {
        "effective_bond_order_sum": float(sum(values)),
        "effective_bond_order_mean": float(sum(values) / len(values)),
        "effective_bond_order_max": float(max(values)),
    }


def ring_edge_fraction(molecule: Molecule) -> float:
    adjacency = build_adjacency(molecule)
    edges = unique_bond_edges(molecule)
    if not edges:
        return 0.0
    ring_edges = sum(1 for edge in edges if edge_in_cycle(adjacency, edge))
    return float(ring_edges / len(edges))


def heavy_atom_degree_summary(molecule: Molecule) -> dict[str, float]:
    adjacency = build_adjacency(molecule)
    degrees = [
        float(sum(1 for neighbor in adjacency.get(atom.atom_id, []) if molecule.atoms[neighbor].attributes.symbol != AtomicSymbol.H))
        for atom in atom_list(molecule)
        if atom.attributes.symbol != AtomicSymbol.H
    ]
    if not degrees:
        return {"heavy_atom_degree_mean": 0.0, "heavy_atom_degree_max": 0.0}
    return {
        "heavy_atom_degree_mean": float(sum(degrees) / len(degrees)),
        "heavy_atom_degree_max": float(max(degrees)),
    }


def log1p_positive(value: float) -> float:
    return log(1.0 + max(0.0, value))


def compute_descriptors(molecule: Molecule) -> MolecularDescriptors:
    bond_summary = effective_bond_order_summary(molecule)
    degree_summary = heavy_atom_degree_summary(molecule)
    return MolecularDescriptors(
        weight=molecule_weight(molecule),
        polar=hetero_atom_count(molecule) + hydrogen_bond_donor_count(molecule) + hydrogen_bond_acceptor_count(molecule),
        surface=molecule_surface_area(molecule),
        bond_order=molecule_bond_order(molecule),
        donor_count=hydrogen_bond_donor_count(molecule),
        acceptor_count=hydrogen_bond_acceptor_count(molecule),
        heavy_atoms=heavy_atom_count(molecule),
        halogens=halogen_atom_count(molecule),
        atom_count_c=atom_count_by_symbol(molecule, AtomicSymbol.C),
        atom_count_n=atom_count_by_symbol(molecule, AtomicSymbol.N),
        atom_count_o=atom_count_by_symbol(molecule, AtomicSymbol.O),
        atom_count_f=atom_count_by_symbol(molecule, AtomicSymbol.F),
        atom_count_p=atom_count_by_symbol(molecule, AtomicSymbol.P),
        atom_count_s=atom_count_by_symbol(molecule, AtomicSymbol.S),
        atom_count_cl=atom_count_by_symbol(molecule, AtomicSymbol.Cl),
        atom_count_br=atom_count_by_symbol(molecule, AtomicSymbol.Br),
        atom_count_i=atom_count_by_symbol(molecule, AtomicSymbol.I),
        atom_count_h=atom_count_by_symbol(molecule, AtomicSymbol.H),
        formal_charge_sum=formal_charge_sum(molecule),
        abs_formal_charge_sum=abs_formal_charge_sum(molecule),
        positive_charge_count=positive_charge_count(molecule),
        negative_charge_count=negative_charge_count(molecule),
        bonding_system_count=bonding_system_count(molecule),
        multicentre_system_count=multicentre_system_count(molecule),
        pi_ring_system_count=aromatic_ring_count(molecule),
        zero_electron_system_count=zero_electron_system_count(molecule),
        sigma_edge_count=sigma_edge_count(molecule),
        effective_bond_order_sum=bond_summary["effective_bond_order_sum"],
        effective_bond_order_mean=bond_summary["effective_bond_order_mean"],
        effective_bond_order_max=bond_summary["effective_bond_order_max"],
        aromatic_rings=aromatic_ring_count(molecule),
        aromatic_atom_count=aromatic_atom_count(molecule),
        aromatic_atom_fraction=aromatic_atom_fraction(molecule),
        ring_edge_fraction=ring_edge_fraction(molecule),
        rotatable_bonds=rotatable_bond_count(molecule),
        heavy_atom_degree_mean=degree_summary["heavy_atom_degree_mean"],
        heavy_atom_degree_max=degree_summary["heavy_atom_degree_max"],
    )


def _neighbors_sigma(molecule: Molecule, atom_id: AtomId) -> tuple[AtomId, ...]:
    neighbors = []
    for edge in molecule.local_bonds:
        if edge.a == atom_id:
            neighbors.append(edge.b)
        elif edge.b == atom_id:
            neighbors.append(edge.a)
    return tuple(sorted(neighbors))


def coordinate_descriptors(molecule: Molecule) -> dict[str, float]:
    coordinates = [
        (
            atom.coordinate.x.value,
            atom.coordinate.y.value,
            atom.coordinate.z.value,
        )
        for atom in atom_list(molecule)
    ]
    if not coordinates:
        return {
            "radius_of_gyration": 0.0,
            "distance_mean": 0.0,
            "distance_std": 0.0,
            "distance_max": 0.0,
            "inertia_eigenvalue_min": 0.0,
            "inertia_eigenvalue_mid": 0.0,
            "inertia_eigenvalue_max": 0.0,
        }
    centroid_x = sum(point[0] for point in coordinates) / len(coordinates)
    centroid_y = sum(point[1] for point in coordinates) / len(coordinates)
    centroid_z = sum(point[2] for point in coordinates) / len(coordinates)
    centered = [
        (
            point[0] - centroid_x,
            point[1] - centroid_y,
            point[2] - centroid_z,
        )
        for point in coordinates
    ]
    squared_radius = [point[0] ** 2 + point[1] ** 2 + point[2] ** 2 for point in centered]
    pairwise = _pairwise_distances(coordinates)
    pairwise_mean = sum(pairwise) / len(pairwise) if pairwise else 0.0
    pairwise_variance = sum((value - pairwise_mean) ** 2 for value in pairwise) / len(pairwise) if pairwise else 0.0
    inertia = _principal_inertia_values(centered)
    return {
        "radius_of_gyration": sqrt(sum(squared_radius) / len(squared_radius)),
        "distance_mean": pairwise_mean,
        "distance_std": sqrt(pairwise_variance),
        "distance_max": max(pairwise) if pairwise else 0.0,
        "inertia_eigenvalue_min": inertia[0],
        "inertia_eigenvalue_mid": inertia[1],
        "inertia_eigenvalue_max": inertia[2],
    }


def _pairwise_distances(coordinates: list[tuple[float, float, float]]) -> list[float]:
    values: list[float] = []
    for left_index, left in enumerate(coordinates):
        for right in coordinates[left_index + 1 :]:
            values.append(
                sqrt(
                    (left[0] - right[0]) ** 2
                    + (left[1] - right[1]) ** 2
                    + (left[2] - right[2]) ** 2
                )
            )
    return values


def _principal_inertia_values(centered: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    if len(centered) <= 1:
        return (0.0, 0.0, 0.0)
    covariance = [[0.0, 0.0, 0.0] for _ in range(3)]
    for x, y, z in centered:
        covariance[0][0] += x * x
        covariance[0][1] += x * y
        covariance[0][2] += x * z
        covariance[1][0] += y * x
        covariance[1][1] += y * y
        covariance[1][2] += y * z
        covariance[2][0] += z * x
        covariance[2][1] += z * y
        covariance[2][2] += z * z
    scale = 1.0 / len(centered)
    matrix = [[value * scale for value in row] for row in covariance]
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    off_diag = matrix[0][1] ** 2 + matrix[0][2] ** 2 + matrix[1][2] ** 2
    if off_diag == 0.0:
        eigenvalues = sorted([matrix[0][0], matrix[1][1], matrix[2][2]])
        return float(eigenvalues[0]), float(eigenvalues[1]), float(eigenvalues[2])
    q = trace / 3.0
    centered_diag = [
        [matrix[i][j] - (q if i == j else 0.0) for j in range(3)]
        for i in range(3)
    ]
    p2 = (
        centered_diag[0][0] ** 2
        + centered_diag[1][1] ** 2
        + centered_diag[2][2] ** 2
        + 2.0 * off_diag
    ) / 6.0
    p = sqrt(max(p2, 0.0))
    if p == 0.0:
        return (q, q, q)
    normalized = [[value / p for value in row] for row in centered_diag]
    determinant = (
        normalized[0][0] * normalized[1][1] * normalized[2][2]
        + normalized[0][1] * normalized[1][2] * normalized[2][0]
        + normalized[0][2] * normalized[1][0] * normalized[2][1]
        - normalized[0][2] * normalized[1][1] * normalized[2][0]
        - normalized[0][1] * normalized[1][0] * normalized[2][2]
        - normalized[0][0] * normalized[1][2] * normalized[2][1]
    )
    r = max(-1.0, min(1.0, determinant / 2.0))
    phi = 0.0
    if r <= -1.0:
        phi = pi / 3.0
    elif r < 1.0:
        phi = (1.0 / 3.0) * __import__("math").acos(r)
    eig1 = q + 2.0 * p * __import__("math").cos(phi)
    eig3 = q + 2.0 * p * __import__("math").cos(phi + (2.0 * pi / 3.0))
    eig2 = 3.0 * q - eig1 - eig3
    eigenvalues = sorted([eig1, eig2, eig3])
    return float(eigenvalues[0]), float(eigenvalues[1]), float(eigenvalues[2])
