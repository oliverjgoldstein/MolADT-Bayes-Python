# MolADT Data Model

MolADT in Python is intentionally shaped like a small typed algebraic data type. The canonical value is `Molecule`: one immutable record with four fields.

## Molecule

```python
@dataclass(frozen=True, slots=True)
class Molecule:
    atoms: Mapping[AtomId, Atom]
    local_bonds: frozenset[Edge]
    systems: tuple[tuple[SystemId, BondingSystem], ...]
    smiles_stereochemistry: SmilesStereochemistry = field(default_factory=SmilesStereochemistry)
```

This is the Python analogue of a Haskell record ADT: a product type whose meaning comes from its fields.

- `atoms`: the typed atom table
- `local_bonds`: the explicit sigma-network edges
- `systems`: the non-local Dietz bonding systems
- `smiles_stereochemistry`: boundary stereochemistry annotations that came from SMILES-style notation

The field accessors are also available as plain functions:

```python
molecule_atoms(molecule)
molecule_local_bonds(molecule)
molecule_systems(molecule)
molecule_smiles_stereochemistry(molecule)
molecule_fields(molecule)
```

That keeps the surface close to destructuring a Haskell record instead of attaching a large method API to the object.

## Atom And Orbitals

Each `Atom` carries coordinates, charge, and shell structure.

```python
@dataclass(frozen=True, slots=True)
class Atom:
    atom_id: AtomId
    attributes: ElementAttributes
    coordinate: Coordinate
    shells: Shells
    formal_charge: int = 0
```

The orbital hierarchy is also explicit and typed:

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
```

So the full shape is: `Molecule` contains `Atom`, and each `Atom` contains `Shells`.

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

Use it when you want to add or remove atoms, bonds, or systems locally, then call `freeze()` to return to the canonical immutable `Molecule`.

```python
mutable = MutableMolecule.from_molecule(molecule)
# edit mutable.atoms / mutable.local_bonds / mutable.systems
molecule = mutable.freeze()
```

In Haskell there is no direct mutable twin. The normal style there is to construct a fresh immutable value instead.

## Relation To Haskell ADTs

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
- module-level accessors play the role of small record selectors
- `MutableMolecule` is only a convenience for editing before returning to the ADT-like form

If you want the shortest summary, think of MolADT as:

`Molecule = atoms + local bonds + bonding systems + stereo annotations`

with orbital structure stored directly on each atom.
