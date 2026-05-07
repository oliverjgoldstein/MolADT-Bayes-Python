from __future__ import annotations

from math import isfinite

from .constants import get_max_bonds_symbol
from .dietz import AtomId, BondingSystem, Edge, SystemId, atoms_of_edge
from .molecule import Molecule


class ValidationError(ValueError):
    pass


BondMap = dict[tuple[AtomId, AtomId], float]
SystemSignature = tuple[int, tuple[Edge, ...], str | None]

_STANDARD_COVALENT_ELECTRONS = {2, 4, 6, 8}
_RESERVED_COVALENT_TAGS = {"single", "double", "triple", "quadruple"}


def used_electrons_at(molecule: Molecule, atom_id: AtomId) -> float:
    return sum(_system_electron_part(atom_id, bonding_system) for _, bonding_system in molecule.systems)


def _system_electron_part(atom_id: AtomId, bonding_system: BondingSystem) -> float:
    degree = sum(1 for edge in bonding_system.member_edges if edge.a == atom_id or edge.b == atom_id)
    total_edges = len(bonding_system.member_edges)
    if total_edges == 0:
        return 0.0
    return bonding_system.shared_electrons.value * degree / (2.0 * total_edges)


def validate_molecule(molecule: Molecule) -> Molecule:
    atom_set = set(molecule.atoms)
    _ensure_atoms(molecule)
    _ensure_system_ids(molecule.systems)
    _ensure_stereochemistry(molecule, atom_set)
    _ensure_unique_systems(molecule.systems)
    full_map: BondMap = {}
    for system_id, bonding_system in molecule.systems:
        _ensure_system_shape(atom_set, system_id, bonding_system)
        full_map = _add_system_bonds(atom_set, bonding_system, full_map)
    _ensure_symmetric(full_map)
    _ensure_valence(molecule, atom_set, full_map)
    return molecule


def _ensure_atoms(molecule: Molecule) -> None:
    for atom_id, atom in molecule.atoms.items():
        if atom.atom_id != atom_id:
            raise ValidationError(f"Atom map key {atom_id.value} does not match Atom.atom_id")
        coordinate_values = (
            atom.coordinate.x.value,
            atom.coordinate.y.value,
            atom.coordinate.z.value,
        )
        if any(not isfinite(value) for value in coordinate_values):
            raise ValidationError(f"Atom {atom_id.value} has non-finite coordinate")
        if atom.attributes.atomic_number <= 0:
            raise ValidationError(f"Atom {atom_id.value} has invalid atomic number")
        if not isfinite(atom.attributes.atomic_weight) or atom.attributes.atomic_weight <= 0:
            raise ValidationError(f"Atom {atom_id.value} has invalid atomic weight")


def _ensure_system_ids(systems: tuple[tuple[SystemId, BondingSystem], ...]) -> None:
    seen: set[int] = set()
    for system_id, _ in systems:
        if system_id.value < 1:
            raise ValidationError("SystemId must be positive")
        if system_id.value in seen:
            raise ValidationError(f"Duplicate SystemId {system_id.value}")
        seen.add(system_id.value)


def _ensure_stereochemistry(molecule: Molecule, atom_set: set[AtomId]) -> None:
    for annotation in molecule.smiles_stereochemistry.atom_stereo:
        if annotation.center not in atom_set:
            raise ValidationError("SMILES atom stereochemistry references non-existent atom")
    for annotation in molecule.smiles_stereochemistry.bond_stereo:
        if annotation.start_atom not in atom_set or annotation.end_atom not in atom_set:
            raise ValidationError("SMILES bond stereochemistry references non-existent atom")


def _ensure_unique_systems(systems: tuple[tuple[SystemId, BondingSystem], ...]) -> None:
    seen: set[SystemSignature] = set()
    for system_id, system in systems:
        signature = _system_signature(system)
        if signature in seen:
            raise ValidationError(f"Duplicate Dietz bonding system at SystemId {system_id.value}")
        seen.add(signature)


def _system_signature(system: BondingSystem) -> SystemSignature:
    return (
        system.shared_electrons.value,
        tuple(sorted(system.member_edges)),
        system.tag,
    )


def _ensure_system_shape(atom_set: set[AtomId], system_id: SystemId, bonding_system: BondingSystem) -> None:
    if not bonding_system.member_edges:
        raise ValidationError(f"Bonding system {system_id.value} has no member edges")
    derived_atoms = frozenset(atom for edge in bonding_system.member_edges for atom in atoms_of_edge(edge))
    if bonding_system.member_atoms != derived_atoms:
        raise ValidationError(f"Bonding system {system_id.value} member atoms do not match member edges")
    _ensure_system_tag_contract(system_id, bonding_system)
    for edge in bonding_system.member_edges:
        if edge.a == edge.b:
            raise ValidationError(f"Atom {edge.a.value} is bonded to itself")
        if edge.a not in atom_set or edge.b not in atom_set:
            raise ValidationError("Bond references non-existent atom")


def _ensure_system_tag_contract(system_id: SystemId, bonding_system: BondingSystem) -> None:
    tag = bonding_system.tag
    shared = bonding_system.shared_electrons.value
    edge_count = len(bonding_system.member_edges)
    if tag is not None and not tag.strip():
        raise ValidationError(f"Bonding system {system_id.value} has an empty tag")
    if tag == "ionic":
        if shared != 0 or edge_count != 1:
            raise ValidationError("Ionic bonding systems must be one-edge systems with 0 shared electrons")
        return
    if tag in _RESERVED_COVALENT_TAGS:
        raise ValidationError("Covalent bond names are display-derived; leave the bonding-system tag unset")
    if edge_count == 1 and shared == 0:
        raise ValidationError("Zero-electron one-edge systems must be tagged ionic")
    if edge_count == 1 and shared in _STANDARD_COVALENT_ELECTRONS and tag is not None:
        raise ValidationError("Ordinary covalent one-edge systems must not carry a bonding-system tag")


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
