from __future__ import annotations

from moladt.chem.dietz import AtomId, mk_bonding_system, mk_edge
from moladt.chem.molecule import Atom, Molecule, neighbors_sigma
from moladt.chem.validate import used_electrons_at, validate_molecule
from moladt.examples import benzene


def relabel_molecule(molecule: Molecule, permutation: list[AtomId]) -> Molecule:
    old_ids = list(molecule.atoms)
    mapping = dict(zip(old_ids, permutation))
    atoms = {
        mapping[atom_id]: Atom(
            atom_id=mapping[atom_id],
            attributes=atom.attributes,
            coordinate=atom.coordinate,
            shells=atom.shells,
            formal_charge=atom.formal_charge,
        )
        for atom_id, atom in molecule.atoms.items()
    }
    local_bonds = frozenset(mk_edge(mapping[edge.a], mapping[edge.b]) for edge in molecule.local_bonds)
    systems = tuple(
        (
            system_id,
            mk_bonding_system(
                bonding_system.shared_electrons,
                frozenset(mk_edge(mapping[edge.a], mapping[edge.b]) for edge in bonding_system.member_edges),
                bonding_system.tag,
            ),
        )
        for system_id, bonding_system in molecule.systems
    )
    return Molecule(atoms=atoms, local_bonds=local_bonds, systems=systems)


def test_validation_is_invariant_under_relabeling() -> None:
    permutation = [AtomId(value) for value in [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]
    relabeled = relabel_molecule(benzene, permutation)
    assert validate_molecule(relabeled) == relabeled
    assert validate_molecule(benzene) == benzene


def test_benzene_electron_accounting() -> None:
    for atom_id in [AtomId(value) for value in range(1, 7)]:
        sigma = float(len(neighbors_sigma(benzene, atom_id)))
        total = used_electrons_at(benzene, atom_id)
        system = total - sigma
        assert system == 1.0
        assert total == 4.0

