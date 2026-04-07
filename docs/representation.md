# MolADT Representation

MolADT is a molecule representation designed to stay explicit where SMILES is compressed.

## What Stays Explicit

MolADT keeps:

- atoms with coordinates
- localized sigma bonds
- Dietz-style bonding systems for delocalized and multicentre structure
- shell and orbital information on atoms

This makes the object closer to the chemistry we actually want to reason over. It is not just a token string and it is not just a graph with a few labels attached.

## Why This Is Better Than Plain SMILES

SMILES is useful as an interchange string, but it is not a good scientific working representation for this project.

- It hides geometry unless you attach another file or model beside it.
- It collapses richer bonding structure into a renderer-oriented syntax.
- It is awkward for non-classical cases such as diborane or ferrocene.
- It gives the model a text serialization instead of the chemistry object we actually care about.

MolADT starts from the chemistry object first and treats SMILES as one possible boundary format, not the center of the system.

## Concrete Example

Take the built-in diborane example.

In a SMILES-first view, this is awkward immediately:

```text
Diborane has two bridging hydrogens with 3c-2e bonding.
There is no clean conservative SMILES string in this repo that says that directly.
Any SMILES-style encoding would have to hide, approximate, or flatten the bridge systems.
```

In MolADT, the structure is stated directly:

```text
Diborane (B2H6)
Dietz-style ADT with two explicit 3c-2e bridging hydrogen bonding systems.

Molecule with 8 atoms, 5 sigma bonds, 2 bonding systems

Bonding systems (2):
  System 1 [bridge_h3_3c2e]: 2 shared electrons
    Atoms: B#1, B#2, H#3
    Edges (+0.50 to bond order each):
      B#1 <-> H#3
      B#2 <-> H#3

  System 2 [bridge_h4_3c2e]: 2 shared electrons
    Atoms: B#1, B#2, H#4
    Edges (+0.50 to bond order each):
      B#1 <-> H#4
      B#2 <-> H#4
```

That is the point of the representation. The unusual bonding is not an edge case bolted onto a string format later; it is part of the molecule object itself.

Ferrocene makes the same point even more strongly: the built-in `pretty-example ferrocene` output can show two cyclopentadienyl `pi` systems plus an Fe back-donation-style pool, while the current conservative SMILES path intentionally refuses to pretend that this is just an ordinary localized graph.

## Benchmark Views

The benchmark uses three MolADT-facing representation branches:

- `moladt`
  The compact ADT-native descriptor table.
- `moladt_typed`
  The richer ADT table, shown in reports as `MolADT+`.
  It adds typed pair channels, radial distance channels, bond-angle channels, torsion channels, and bonding-system summaries.
- `moladt_typed_geom`
  The coordinate-aware branch, shown as `MolADT+ 3D`.
  It keeps atomic numbers and coordinates for the geometry model and also carries the richer global ADT descriptors.

## What We Are Testing

There are two different questions in the reports:

- Fair tabular question:
  if the learner is held fixed, does MolADT beat SMILES?
- Geometry question:
  if the model can use 3D coordinates as well, how far can `MolADT+ 3D` push the result?

That is why the repo now writes both a fair tabular comparison and a mixed-family frontier comparison.

## See Also

- [Parsing and Rendering](parsing.md)
- [Models and Benchmarks](models.md)
- [Outputs](outputs.md)
