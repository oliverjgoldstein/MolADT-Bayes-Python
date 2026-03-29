from __future__ import annotations

from dataclasses import dataclass

from .constants import get_max_bonds_symbol
from .dietz import AtomId, BondingSystem, Edge
from .molecule import Molecule, neighbors_sigma


class ValidationError(ValueError):
    pass


BondMap = dict[tuple[AtomId, AtomId], float]


def used_electrons_at(molecule: Molecule, atom_id: AtomId) -> float:
    sigma = float(len(neighbors_sigma(molecule, atom_id)))
    system = sum(_system_electron_part(atom_id, bonding_system) for _, bonding_system in molecule.systems)
    return sigma + system


def _system_electron_part(atom_id: AtomId, bonding_system: BondingSystem) -> float:
    degree = sum(1 for edge in bonding_system.member_edges if edge.a == atom_id or edge.b == atom_id)
    total_edges = len(bonding_system.member_edges)
    if total_edges == 0:
        return 0.0
    return bonding_system.shared_electrons.value * degree / (2.0 * total_edges)


def validate_molecule(molecule: Molecule) -> Molecule:
    atom_set = set(molecule.atoms)
    sigma_map: BondMap = {}
    for edge in molecule.local_bonds:
        sigma_map = _accumulate_bond(atom_set, 2.0, sigma_map, edge)
    full_map = sigma_map
    for _, bonding_system in molecule.systems:
        full_map = _add_system_bonds(atom_set, bonding_system, full_map)
    _ensure_symmetric(full_map)
    _ensure_valence(molecule, atom_set, full_map)
    return molecule


def _accumulate_bond(atom_set: set[AtomId], value: float, acc: BondMap, edge: Edge) -> BondMap:
    if edge.a == edge.b:
        raise ValidationError(f"Atom {edge.a.value} is bonded to itself")
    if edge.a not in atom_set or edge.b not in atom_set:
        raise ValidationError("Bond references non-existent atom")
    updated = dict(acc)
    _add_directed(updated, edge.a, edge.b, value)
    _add_directed(updated, edge.b, edge.a, value)
    return updated


def _add_system_bonds(atom_set: set[AtomId], bonding_system: BondingSystem, acc: BondMap) -> BondMap:
    edge_count = len(bonding_system.member_edges)
    if edge_count == 0:
        return acc
    contribution = bonding_system.shared_electrons.value / edge_count
    updated = dict(acc)
    for edge in bonding_system.member_edges:
        if edge.a == edge.b:
            raise ValidationError(f"Atom {edge.a.value} is bonded to itself")
        if edge.a not in atom_set or edge.b not in atom_set:
            raise ValidationError("Bond references non-existent atom")
        _add_directed(updated, edge.a, edge.b, contribution)
        _add_directed(updated, edge.b, edge.a, contribution)
    return updated


def _ensure_symmetric(bond_map: BondMap) -> None:
    for (atom_i, atom_j), value in bond_map.items():
        mirrored = bond_map.get((atom_j, atom_i))
        if mirrored is None or not _approx_equal(value, mirrored):
            raise ValidationError("Bond map is not symmetric")


def _ensure_valence(molecule: Molecule, atom_set: set[AtomId], bond_map: BondMap) -> None:
    for atom_id in atom_set:
        atom = molecule.atoms.get(atom_id)
        if atom is None:
            raise ValidationError("Bond references non-existent atom")
        total = sum(value for (source, _), value in bond_map.items() if source == atom_id)
        used = total / 2.0
        if used > get_max_bonds_symbol(atom.attributes.symbol) + 1e-9:
            raise ValidationError(f"Atom {atom_id.value} exceeds maximum valence")


def _add_directed(bond_map: BondMap, atom_i: AtomId, atom_j: AtomId, value: float) -> None:
    bond_map[(atom_i, atom_j)] = bond_map.get((atom_i, atom_j), 0.0) + value


def _approx_equal(value_a: float, value_b: float) -> bool:
    return abs(value_a - value_b) <= 1e-9

