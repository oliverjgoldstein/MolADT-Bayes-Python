# MolADT ADT Representation

MolADT in Python is meant to behave like a typed record ADT, not like a large object-oriented chemistry API. The canonical value is `Molecule`, and every other type exists to make that record explicit.

## Full Shape

At a high level, the full nested shape is:

```text
Molecule
  = atoms: Map AtomId Atom
  + local_bonds: Set Edge
  + systems: [(SystemId, BondingSystem)]
  + smiles_stereochemistry: SmilesStereochemistry

Atom
  = atom_id
  + attributes: ElementAttributes
  + coordinate: Coordinate
  + shells: Shells
  + formal_charge

BondingSystem
  = shared_electrons
  + member_atoms
  + member_edges
  + tag
```

So the representation is not just a graph. It is:

- a typed atom table
- an explicit sigma-network
- a separate Dietz bonding-system layer for delocalized or multicenter structure
- a stereo annotation layer for boundary SMILES information
- orbital shell structure stored on each atom

## Core Molecule Record

```python
@dataclass(frozen=True, slots=True)
class Molecule:
    atoms: Mapping[AtomId, Atom]
    local_bonds: frozenset[Edge]
    systems: tuple[tuple[SystemId, BondingSystem], ...]
    smiles_stereochemistry: SmilesStereochemistry = field(default_factory=SmilesStereochemistry)
```

This is the Python analogue of a Haskell record ADT: one product type whose fields are the molecule.

- `atoms` is the main atom table, keyed by stable `AtomId`
- `local_bonds` is the explicit undirected sigma-network
- `systems` is the extra non-local bonding layer
- `smiles_stereochemistry` stores stereochemical annotations that came from a SMILES-like boundary format

The accessors are intentionally plain functions:

```python
molecule_atoms(molecule)
molecule_local_bonds(molecule)
molecule_systems(molecule)
molecule_smiles_stereochemistry(molecule)
molecule_fields(molecule)
```

That keeps the surface close to record selection and destructuring.

## Dietz Layer

The non-local chemistry lives in the Dietz primitives from `moladt.chem.dietz`.

```python
@dataclass(frozen=True, slots=True, order=True)
class AtomId:
    value: int


@dataclass(frozen=True, slots=True, order=True)
class SystemId:
    value: int


@dataclass(frozen=True, slots=True, order=True)
class Edge:
    a: AtomId
    b: AtomId


@dataclass(frozen=True, slots=True)
class BondingSystem:
    shared_electrons: NonNegative
    member_atoms: frozenset[AtomId]
    member_edges: frozenset[Edge]
    tag: str | None = None
```

This is the part that makes MolADT more than a plain string or plain graph.

- `Edge` is a canonical undirected pair
- `local_bonds` gives the localized sigma framework
- `BondingSystem` overlays an electron-sharing system on top of a set of edges
- `tag` is an optional label such as a ring or bridge name

In other words, `local_bonds` says where the local edges are, and `systems` says which of those edges participate in delocalized or multicenter pools.

## Atom Record

Each atom is also a typed record:

```python
@dataclass(frozen=True, slots=True)
class ElementAttributes:
    symbol: AtomicSymbol
    atomic_number: int
    atomic_weight: float


@dataclass(frozen=True, slots=True)
class Atom:
    atom_id: AtomId
    attributes: ElementAttributes
    coordinate: Coordinate
    shells: Shells
    formal_charge: int = 0
```

That means an atom carries:

- identity: `atom_id`
- chemistry: `ElementAttributes`
- geometry: `coordinate`
- electronic shell structure: `shells`
- explicit charge: `formal_charge`

## Stereo Layer

The SMILES stereo layer is kept separate from the main bonding structure.

```python
@dataclass(frozen=True, slots=True)
class SmilesAtomStereo:
    center: AtomId
    stereo_class: SmilesAtomStereoClass
    configuration: int
    token: str


@dataclass(frozen=True, slots=True)
class SmilesBondStereo:
    start_atom: AtomId
    end_atom: AtomId
    direction: SmilesBondStereoDirection


@dataclass(frozen=True, slots=True)
class SmilesStereochemistry:
    atom_stereo: tuple[SmilesAtomStereo, ...] = ()
    bond_stereo: tuple[SmilesBondStereo, ...] = ()
```

This is important structurally:

- stereochemistry is not overloaded into `local_bonds`
- atom-centered and bond-centered stereo are explicit records
- the canonical `Molecule` still keeps stereo as one field of the overall ADT

## Orbitals And Shells

The orbital hierarchy is also explicit and typed.

```python
@dataclass(frozen=True, slots=True)
class Orbital(Generic[SubshellType]):
    orbital_type: SubshellType
    electron_count: int
    orientation: Coordinate | None = None
    hybrid_components: tuple[tuple[float, PureOrbital], ...] | None = None


@dataclass(frozen=True, slots=True)
class SubShell(Generic[SubshellType]):
    orbitals: tuple[Orbital[SubshellType], ...]


@dataclass(frozen=True, slots=True)
class Shell:
    principal_quantum_number: int
    s_subshell: SubShell[So] | None = None
    p_subshell: SubShell[P] | None = None
    d_subshell: SubShell[D] | None = None
    f_subshell: SubShell[F] | None = None


Shells = tuple[Shell, ...]
```

So the path is:

`Molecule -> Atom -> Shells -> Shell -> SubShell -> Orbital`

This is why the representation is closer to an ADT tree than to a flat interchange format.

## MutableMolecule

`MutableMolecule` is the editable scratch version of the same record shape.

```python
@dataclass(slots=True)
class MutableMolecule:
    atoms: dict[AtomId, Atom]
    local_bonds: set[Edge]
    systems: list[tuple[SystemId, BondingSystem]]
    smiles_stereochemistry: SmilesStereochemistry = field(default_factory=SmilesStereochemistry)
```

Use it when you want local graph surgery or proposal edits, then call `freeze()` to return to canonical `Molecule`.

```python
mutable = MutableMolecule.from_molecule(molecule)
# edit mutable.atoms / mutable.local_bonds / mutable.systems
molecule = mutable.freeze()
```

The important point is that `MutableMolecule` is not the main representation. It is just a writable builder around the immutable ADT-like form.

## Relation To Haskell

The Haskell repo carries the same model more literally:

```haskell
data Molecule = Molecule
  { atoms :: Map AtomId Atom
  , localBonds :: Set Edge
  , systems :: [(SystemId, BondingSystem)]
  , smilesStereochemistry :: SmilesStereochemistry
  }
```

That is why the Python design stays minimal.

- `@dataclass(frozen=True, slots=True)` plays the role of the immutable record value
- module-level accessors play the role of record selectors
- `MutableMolecule` is only a convenience for edits before returning to the main record shape

If you want the shortest summary, MolADT is:

`Molecule = atoms + sigma edges + bonding systems + stereo annotations`

with shell and orbital structure stored directly on each atom.
