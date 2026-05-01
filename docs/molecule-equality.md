# Molecule Equality

Use `same_molecule` when you want to ask whether two `Molecule` values are the
same MolADT object after harmless ordering differences are removed.

It is stricter than graph isomorphism, but more useful than raw dataclass
equality for serialized molecules.

## What It Ignores

`same_molecule` ignores incidental ordering in:

- atom mappings
- local edge sets
- bonding-system tuples
- `member_edges` inside each `BondingSystem`
- stereochemistry annotation tuples
- endpoint order inside an `Edge`

Atom IDs and system IDs still matter. If atom `1` and atom `2` are swapped
throughout the whole molecule, that is a relabelled molecule, not the same
MolADT value under `same_molecule`.

## Python Example

```python
from moladt.chem import (
    AtomId,
    AtomicSymbol,
    Edge,
    Molecule,
    NonNegative,
    SystemId,
    mk_bonding_system,
    same_molecule,
)
from moladt.examples._literal import atom

molecule = Molecule(
    atoms={
        AtomId(1): atom(1, AtomicSymbol.O, 0.0, 0.0, 0.0),
        AtomId(2): atom(2, AtomicSymbol.H, 0.0, 0.8, 0.0),
        AtomId(3): atom(3, AtomicSymbol.H, 0.8, 0.0, 0.0),
    },
    local_bonds=frozenset(
        {
            Edge(AtomId(1), AtomId(2)),
            Edge(AtomId(1), AtomId(3)),
        }
    ),
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(2),
                frozenset({Edge(AtomId(1), AtomId(2))}),
                "oh_a",
            ),
        ),
        (
            SystemId(2),
            mk_bonding_system(
                NonNegative(2),
                frozenset({Edge(AtomId(1), AtomId(3))}),
                "oh_b",
            ),
        ),
    ),
)

reordered = Molecule(
    atoms={
        AtomId(3): molecule.atoms[AtomId(3)],
        AtomId(2): molecule.atoms[AtomId(2)],
        AtomId(1): molecule.atoms[AtomId(1)],
    },
    local_bonds=frozenset(
        {
            Edge(AtomId(2), AtomId(1)),
            Edge(AtomId(3), AtomId(1)),
        }
    ),
    systems=(molecule.systems[1], molecule.systems[0]),
)

assert same_molecule(molecule, reordered)
```

The two values serialize differently, but they carry the same atoms, local
bonds, system IDs, and bonding systems.

## Structural Changes Still Fail

```python
changed = Molecule(
    atoms=molecule.atoms,
    local_bonds=frozenset({Edge(AtomId(1), AtomId(2))}),
    systems=molecule.systems,
)

assert not same_molecule(molecule, changed)
```

Removing a local bond changes the molecule, so the equality check fails.

## Canonical Keys

`molecule_canonical_key(molecule)` exposes the normalized tuple used by
`same_molecule`. Use it when you need a stable key for deduplication or a
regression assertion.

```python
from moladt.chem import molecule_canonical_key

seen = {molecule_canonical_key(molecule)}
assert molecule_canonical_key(reordered) in seen
```

## When To Use It

Use `same_molecule` for round-trip tests, JSON comparisons, generated molecule
deduplication, and checks where the source may serialize maps, sets, or systems
in a different order.

Use a separate isomorphism or relabelling check if you want to treat different
atom IDs as equivalent.
