from __future__ import annotations

import pytest

from moladt.chem.constants import element_attributes
from moladt.chem.coordinate import Coordinate, mk_angstrom
from moladt.chem.dietz import AtomId, BondingSystem, NonNegative, SystemId, mk_bonding_system, mk_edge
from moladt.chem.molecule import Atom, AtomicSymbol, Molecule
from moladt.chem.molecule_ops import neighbors_sigma
from moladt.chem.validate import ValidationError, used_electrons_at, validate_molecule
from moladt.examples import benzene


def _atom(atom_id: int, symbol: AtomicSymbol = AtomicSymbol.C) -> Atom:
    return Atom(
        atom_id=AtomId(atom_id),
        attributes=element_attributes(symbol),
        coordinate=Coordinate(mk_angstrom(float(atom_id)), mk_angstrom(0.0), mk_angstrom(0.0)),
        shells=element_attributes(symbol).shells,
    )


def _small_molecule(*systems: tuple[SystemId, BondingSystem]) -> Molecule:
    return Molecule(
        atoms={
            AtomId(1): _atom(1),
            AtomId(2): _atom(2),
            AtomId(3): _atom(3),
        },
        systems=systems,
    )


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
    return Molecule(atoms=atoms, systems=systems)


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


def test_validator_rejects_duplicate_system_ids() -> None:
    molecule = _small_molecule(
        (
            SystemId(1),
            mk_bonding_system(NonNegative(2), {mk_edge(AtomId(1), AtomId(2))}),
        ),
        (
            SystemId(1),
            mk_bonding_system(NonNegative(2), {mk_edge(AtomId(2), AtomId(3))}),
        ),
    )

    with pytest.raises(ValidationError, match="Duplicate SystemId 1"):
        validate_molecule(molecule)


def test_validator_rejects_empty_bonding_systems() -> None:
    molecule = _small_molecule(
        (
            SystemId(1),
            BondingSystem(NonNegative(0), frozenset(), frozenset()),
        )
    )

    with pytest.raises(ValidationError, match="has no member edges"):
        validate_molecule(molecule)


def test_validator_rejects_named_ordinary_covalent_edges() -> None:
    molecule = _small_molecule(
        (
            SystemId(1),
            mk_bonding_system(NonNegative(4), {mk_edge(AtomId(1), AtomId(2))}, "alkene_bridge"),
        )
    )

    with pytest.raises(ValidationError, match="Ordinary covalent one-edge systems"):
        validate_molecule(molecule)


def test_validator_rejects_nonzero_ionic_systems() -> None:
    molecule = _small_molecule(
        (
            SystemId(1),
            mk_bonding_system(NonNegative(2), {mk_edge(AtomId(1), AtomId(2))}, "ionic"),
        )
    )

    with pytest.raises(ValidationError, match="Ionic bonding systems"):
        validate_molecule(molecule)


def test_validator_rejects_nonfinite_coordinates() -> None:
    molecule = Molecule(
        atoms={
            AtomId(1): Atom(
                atom_id=AtomId(1),
                attributes=element_attributes(AtomicSymbol.C),
                coordinate=Coordinate(mk_angstrom(float("nan")), mk_angstrom(0.0), mk_angstrom(0.0)),
                shells=element_attributes(AtomicSymbol.C).shells,
            ),
            AtomId(2): _atom(2),
        },
        systems=(
            (
                SystemId(1),
                mk_bonding_system(NonNegative(2), {mk_edge(AtomId(1), AtomId(2))}),
            ),
        ),
    )

    with pytest.raises(ValidationError, match="non-finite coordinate"):
        validate_molecule(molecule)
