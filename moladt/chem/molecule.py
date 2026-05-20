from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, TypeAlias

from .coordinate import Coordinate
from .dietz import AtomId, BondingSystem, Edge, NonNegative, SystemId
from .orbital import Shells


class AtomicSymbol(Enum):
    H = "H"
    C = "C"
    N = "N"
    O = "O"
    S = "S"
    P = "P"
    Si = "Si"
    F = "F"
    Cl = "Cl"
    Br = "Br"
    I = "I"
    Fe = "Fe"
    B = "B"
    Na = "Na"
    He = "He"
    Li = "Li"
    Be = "Be"
    Ne = "Ne"
    Mg = "Mg"
    Al = "Al"
    Ar = "Ar"
    K = "K"
    Ca = "Ca"
    Sc = "Sc"
    Ti = "Ti"
    V = "V"
    Cr = "Cr"
    Mn = "Mn"
    Co = "Co"
    Ni = "Ni"
    Cu = "Cu"
    Zn = "Zn"
    Ga = "Ga"
    Ge = "Ge"
    As = "As"
    Se = "Se"
    Kr = "Kr"
    Rb = "Rb"
    Sr = "Sr"
    Y = "Y"
    Zr = "Zr"
    Nb = "Nb"
    Mo = "Mo"
    Tc = "Tc"
    Ru = "Ru"
    Rh = "Rh"
    Pd = "Pd"
    Ag = "Ag"
    Cd = "Cd"
    In = "In"
    Sn = "Sn"
    Sb = "Sb"
    Te = "Te"
    Xe = "Xe"
    Cs = "Cs"
    Ba = "Ba"
    La = "La"
    Ce = "Ce"
    Pr = "Pr"
    Nd = "Nd"
    Pm = "Pm"
    Sm = "Sm"
    Eu = "Eu"
    Gd = "Gd"
    Tb = "Tb"
    Dy = "Dy"
    Ho = "Ho"
    Er = "Er"
    Tm = "Tm"
    Yb = "Yb"
    Lu = "Lu"
    Hf = "Hf"
    Ta = "Ta"
    W = "W"
    Re = "Re"
    Os = "Os"
    Ir = "Ir"
    Pt = "Pt"
    Au = "Au"
    Hg = "Hg"
    Tl = "Tl"
    Pb = "Pb"
    Bi = "Bi"
    Po = "Po"
    At = "At"
    Rn = "Rn"
    Fr = "Fr"
    Ra = "Ra"
    Ac = "Ac"
    Th = "Th"
    Pa = "Pa"
    U = "U"
    Np = "Np"
    Pu = "Pu"
    Am = "Am"
    Cm = "Cm"
    Bk = "Bk"
    Cf = "Cf"
    Es = "Es"
    Fm = "Fm"
    Md = "Md"
    No = "No"
    Lr = "Lr"
    Rf = "Rf"
    Db = "Db"
    Sg = "Sg"
    Bh = "Bh"
    Hs = "Hs"
    Mt = "Mt"
    Ds = "Ds"
    Rg = "Rg"
    Cn = "Cn"
    Nh = "Nh"
    Fl = "Fl"
    Mc = "Mc"
    Lv = "Lv"
    Ts = "Ts"
    Og = "Og"

    def __str__(self) -> str:
        return self.value


class SmilesAtomStereoClass(Enum):
    TETRAHEDRAL = "TH"
    ALLENE = "AL"
    SQUARE_PLANAR = "SP"
    TRIGONAL_BIPYRAMIDAL = "TB"
    OCTAHEDRAL = "OH"


class SmilesBondStereoDirection(Enum):
    UP = "/"
    DOWN = "\\"


@dataclass(frozen=True, slots=True)
class SmilesAtomStereo:
    center: AtomId
    stereo_class: SmilesAtomStereoClass
    configuration: int
    token: str

    def __post_init__(self) -> None:
        if self.configuration <= 0:
            raise ValueError("SMILES atom stereochemistry configuration must be positive")


@dataclass(frozen=True, slots=True)
class SmilesBondStereo:
    start_atom: AtomId
    end_atom: AtomId
    direction: SmilesBondStereoDirection


@dataclass(frozen=True, slots=True)
class SmilesStereochemistry:
    atom_stereo: tuple[SmilesAtomStereo, ...] = ()
    bond_stereo: tuple[SmilesBondStereo, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "atom_stereo",
            tuple(
                sorted(
                    self.atom_stereo,
                    key=lambda item: (
                        item.center.value,
                        item.stereo_class.value,
                        item.configuration,
                        item.token,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "bond_stereo",
            tuple(
                sorted(
                    self.bond_stereo,
                    key=lambda item: (
                        item.start_atom.value,
                        item.end_atom.value,
                        item.direction.value,
                    ),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ElementAttributes:
    symbol: AtomicSymbol
    atomic_number: int
    atomic_weight: float
    shells: Shells | None = None

    def __str__(self) -> str:
        return f"{self.symbol.value}(Z={self.atomic_number}, {self.atomic_weight:.4f} u)"


@dataclass(frozen=True, slots=True)
class Atom:
    atom_id: AtomId
    attributes: ElementAttributes
    coordinate: Coordinate
    shells: Shells | None = None
    formal_charge: int = 0

    def __post_init__(self) -> None:
        if self.shells is None and self.attributes.shells is not None:
            object.__setattr__(self, "shells", self.attributes.shells)

    def pretty(self) -> str:
        from .pretty import pretty_text

        return pretty_text(self)

    def __str__(self) -> str:
        return self.pretty()


MoleculeSystems: TypeAlias = tuple[tuple[SystemId, BondingSystem], ...]
MoleculeFields: TypeAlias = tuple[
    Mapping[AtomId, Atom],
    MoleculeSystems,
    SmilesStereochemistry,
]
CanonicalBondingSystem: TypeAlias = tuple[
    NonNegative,
    tuple[AtomId, ...],
    tuple[Edge, ...],
    str | None,
]
CanonicalStereochemistry: TypeAlias = tuple[
    tuple[tuple[AtomId, SmilesAtomStereoClass, int, str], ...],
    tuple[tuple[AtomId, AtomId, SmilesBondStereoDirection], ...],
]
CanonicalMolecule: TypeAlias = tuple[
    tuple[tuple[AtomId, Atom], ...],
    tuple[tuple[SystemId, CanonicalBondingSystem], ...],
    CanonicalStereochemistry,
]


def _normalize_atom_map(atoms: Mapping[AtomId, Atom]) -> Mapping[AtomId, Atom]:
    atom_map = dict(sorted(atoms.items(), key=lambda item: item[0].value))
    for atom_id, atom in atom_map.items():
        if atom.atom_id != atom_id:
            raise ValueError("Atom map keys must match Atom.atom_id")
    return MappingProxyType(atom_map)


def _normalize_systems(systems: MoleculeSystems) -> MoleculeSystems:
    return tuple(sorted(systems, key=lambda item: item[0].value))


@dataclass(frozen=True, slots=True)
class Molecule:
    atoms: Mapping[AtomId, Atom]
    systems: MoleculeSystems = ()
    smiles_stereochemistry: SmilesStereochemistry = field(default_factory=SmilesStereochemistry)

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", _normalize_atom_map(self.atoms))
        object.__setattr__(self, "systems", _normalize_systems(self.systems))


def same_molecule(left: Molecule, right: Molecule) -> bool:
    """Return True when two MolADT values differ only by container ordering.

    Atom and system identifiers still matter. This is not graph isomorphism or
    atom relabelling; it is equality after putting maps, edge sets, bonding
    systems, and stereochemistry annotations into a canonical order.
    """

    return molecule_canonical_key(left) == molecule_canonical_key(right)


def molecule_canonical_key(molecule: Molecule) -> CanonicalMolecule:
    system_entries = (
        (system_id, _canonical_bonding_system(system))
        for system_id, system in molecule.systems
    )
    return (
        tuple(sorted(molecule.atoms.items(), key=lambda item: item[0].value)),
        tuple(
            sorted(
                system_entries,
                key=lambda item: (
                    item[0].value,
                    _canonical_bonding_system_sort_key(item[1]),
                ),
            )
        ),
        _canonical_stereochemistry(molecule.smiles_stereochemistry),
    )


def _canonical_bonding_system(system: BondingSystem) -> CanonicalBondingSystem:
    return (
        system.shared_electrons,
        tuple(sorted(system.member_atoms)),
        tuple(sorted(system.member_edges)),
        system.tag,
    )


def _canonical_bonding_system_sort_key(
    system: CanonicalBondingSystem,
) -> tuple[int, tuple[int, ...], tuple[tuple[int, int], ...], tuple[int, str]]:
    tag = system[3]
    return (
        system[0].value,
        tuple(atom.value for atom in system[1]),
        tuple((edge.a.value, edge.b.value) for edge in system[2]),
        (0, "") if tag is None else (1, tag),
    )


def _canonical_stereochemistry(stereo: SmilesStereochemistry) -> CanonicalStereochemistry:
    atom_stereo = (
        (
            item.center,
            item.stereo_class,
            item.configuration,
            item.token,
        )
        for item in stereo.atom_stereo
    )
    bond_stereo = (
        (
            item.start_atom,
            item.end_atom,
            item.direction,
        )
        for item in stereo.bond_stereo
    )
    return (
        tuple(
            sorted(
                atom_stereo,
                key=lambda item: (
                    item[0].value,
                    item[1].value,
                    item[2],
                    item[3],
                ),
            )
        ),
        tuple(
            sorted(
                bond_stereo,
                key=lambda item: (
                    item[0].value,
                    item[1].value,
                    item[2].value,
                ),
            )
        ),
    )


def molecule_atoms(molecule: Molecule) -> Mapping[AtomId, Atom]:
    return molecule.atoms


def molecule_edges(molecule: Molecule) -> frozenset[Edge]:
    return frozenset(edge for _, system in molecule.systems for edge in system.member_edges)


def molecule_systems(molecule: Molecule) -> MoleculeSystems:
    return molecule.systems


def molecule_smiles_stereochemistry(molecule: Molecule) -> SmilesStereochemistry:
    return molecule.smiles_stereochemistry


def molecule_fields(molecule: Molecule) -> MoleculeFields:
    return (
        molecule.atoms,
        molecule.systems,
        molecule.smiles_stereochemistry,
    )
