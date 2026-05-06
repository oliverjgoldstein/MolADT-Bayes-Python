from __future__ import annotations

from math import sqrt

from .coordinate import Angstrom, mk_angstrom
from .dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system, mk_edge
from .molecule import Atom, Molecule


def add_sigma(atom_a: AtomId, atom_b: AtomId, molecule: Molecule) -> Molecule:
    edge = mk_edge(atom_a, atom_b)
    if edge in molecule.local_bonds:
        return molecule
    next_system_id = max((system_id.value for system_id, _ in molecule.systems), default=0) + 1
    return Molecule(
        atoms=molecule.atoms,
        local_bonds=molecule.local_bonds | {edge},
        systems=molecule.systems
        + (
            (
                SystemId(next_system_id),
                mk_bonding_system(NonNegative(2), frozenset({edge})),
            ),
        ),
        smiles_stereochemistry=molecule.smiles_stereochemistry,
    )


def distance_angstrom(atom_a: Atom, atom_b: Atom) -> Angstrom:
    dx = atom_a.coordinate.x.value - atom_b.coordinate.x.value
    dy = atom_a.coordinate.y.value - atom_b.coordinate.y.value
    dz = atom_a.coordinate.z.value - atom_b.coordinate.z.value
    return mk_angstrom(sqrt(dx * dx + dy * dy + dz * dz))


def neighbors_sigma(molecule: Molecule, atom_id: AtomId) -> tuple[AtomId, ...]:
    neighbors: list[AtomId] = []
    for edge in molecule.local_bonds:
        if edge.a == atom_id:
            neighbors.append(edge.b)
        elif edge.b == atom_id:
            neighbors.append(edge.a)
    return tuple(sorted(neighbors))


def edge_systems(molecule: Molecule, edge: Edge) -> tuple[SystemId, ...]:
    return tuple(system_id for system_id, system in molecule.systems if edge in system.member_edges)


def effective_order(molecule: Molecule, edge: Edge) -> float:
    contribution = 0.0
    for _, system in molecule.systems:
        if edge in system.member_edges and system.member_edges:
            contribution += system.shared_electrons.value / (2.0 * len(system.member_edges))
    return contribution


def pretty_print_molecule(molecule: Molecule) -> str:
    from .pretty import pretty_text

    return pretty_text(molecule)
