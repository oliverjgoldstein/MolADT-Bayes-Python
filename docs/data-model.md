# ADT Model

MolADT is a small typed record model. The main value is `Molecule`.

```text
Molecule
  atoms: Map AtomId Atom
  local_bonds: Set Edge
  systems: [(SystemId, BondingSystem)]
  smiles_stereochemistry: SmilesStereochemistry
```

## Main Fields

| Field | Meaning |
| --- | --- |
| `atoms` | Atom table keyed by stable `AtomId`. |
| `local_bonds` | Ordinary localized sigma edges. |
| `systems` | Dietz bonding systems for delocalized or multicenter structure. |
| `smiles_stereochemistry` | Stereo annotations parsed from SMILES-like boundary formats. |

## Atom

```text
Atom
  atom_id
  attributes
  coordinate
  shells
  formal_charge
```

Atoms carry element data, 3D coordinates, local shell/orbital structure, and explicit charge.

## Bonding System

```text
BondingSystem
  shared_electrons
  member_atoms
  member_edges
  tag
```

`local_bonds` is the local graph. `systems` overlays electron-sharing pools on top of that graph.

Examples:

- benzene: a six-electron `pi_ring`
- diborane: two `3c-2e` bridge systems
- ferrocene: Cp `pi` systems and an Fe back-donation pool

## Mutable Edits

`Molecule` is immutable. Use `MutableMolecule` as a scratchpad:

```python
from moladt import AtomId, MutableMolecule, mk_edge
from moladt.chem.validate import ValidationError, validate_molecule
from moladt.examples import water

mutable = MutableMolecule.from_molecule(water)
mutable.local_bonds.add(mk_edge(AtomId(1), AtomId(3)))

try:
    validate_molecule(mutable.freeze())
except ValidationError as exc:
    print(f"invalid edit: {exc}")

mutable.local_bonds.remove(mk_edge(AtomId(1), AtomId(3)))
molecule = validate_molecule(mutable.freeze())
```

The inverse-design code uses this pattern: edit, freeze, validate, score.

## Haskell Alignment

The sibling Haskell repo uses the same shape as a record ADT. Python keeps that style with frozen dataclasses and plain fields rather than a large object-oriented API.

Shortest summary:

```text
Molecule = atoms + sigma edges + bonding systems + stereo annotations
```

See also [Representation](representation.md), [Orbitals](orbitals.md), and [Haskell interop](haskell_interop.md).
