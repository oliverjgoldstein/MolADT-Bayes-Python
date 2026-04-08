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

In a SMILES-first view, standard notation is already flattening the chemistry:

```text
[BH2]1[H][BH2][H]1

That is a recognizable boundary string, but it still does not say
"two explicit 3c-2e bridge systems".
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

Ferrocene makes the same point even more strongly. Standard SMILES can write it as:

```text
[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]
```

That is useful as a boundary string, but it breaks the sandwich into disconnected ionic fragments instead of the shared Cp/metal pools that the MolADT example stores directly.

## Morphine and Ring Closures

Morphine shows the classical side of the same boundary. The standard stereochemical SMILES is:

```text
CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5
```

That boundary string is faithful for the classical graph, but it still pushes fused-ring bookkeeping, localized double-bond syntax, and atom-centered `@`/`@@` flags into string syntax.

In the built-in morphine example at [`moladt/examples/morphine.py`](../moladt/examples/morphine.py), the five classical ring-closure edges from the standard sketch are already ordinary `local_bonds`:

- `O#1 <-> C#11`
- `C#2 <-> C#8`
- `C#7 <-> C#18`
- `C#9 <-> C#21`
- `C#10 <-> C#16`

The explicit Dietz object then keeps the remaining structure equally direct:

- `C#5 <-> C#6` is an explicit two-electron system tagged `alkene_bridge`
- the phenyl fragment `C#10-C#11-C#12-C#14-C#15-C#16` is an explicit six-electron system tagged `phenyl_pi_ring`
- the five atom-centered flags are preserved in `smiles_stereochemistry.atom_stereo` as centers `#2`, `#3`, `#7`, `#8`, and `#18`

That is the direct comparison to the image. SMILES is still a useful boundary notation, but MolADT stores the fused polycycle, its delocalization, and its parsed stereochemistry flags directly instead of making string syntax the center of the representation. The conservative `parse-smiles` path keeps the Kekule-style double bonds from the boundary string explicit; the hand-built morphine example then groups the alkene and phenyl fragment into Dietz systems on purpose.

Use:

```bash
./.venv/bin/python -m moladt.cli pretty-example morphine
./.venv/bin/python -m moladt.cli parse-smiles "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
```

The first command shows the explicit Dietz object. The second shows the boundary-format parse path based on the same figure.

## Side-by-Side Examples

| Example | SMILES Side | MolADT Side |
| --- | --- | --- |
| Diborane | `[BH2]1[H][BH2][H]1` is a standard boundary string, but it flattens the two bridging hydrogens instead of saying "two explicit 3c-2e bridges". | [`moladt/examples/diborane.py`](../moladt/examples/diborane.py) stores 5 ordinary sigma edges and 2 explicit Dietz bridge systems: `bridge_h3_3c2e` and `bridge_h4_3c2e`. |
| Ferrocene | `[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]` is a standard boundary string, but it splits the sandwich into ionic fragments instead of shared Cp/metal pools. | [`moladt/examples/ferrocene.py`](../moladt/examples/ferrocene.py) stores the sandwich graph directly, plus `cp1_pi`, `cp2_pi`, and `fe_backdonation` Dietz systems. |
| Morphine | `CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5` expresses the fused graph through ring digits, localized double-bond syntax, and five atom-centered stereo flags. | [`moladt/examples/morphine.py`](../moladt/examples/morphine.py) stores the same fused graph directly in `local_bonds`, keeps the alkene and phenyl delocalization explicit, and preserves the five atom-centered SMILES flags in `smiles_stereochemistry`. |

That is the recurring boundary in this repo: use SMILES where the notation actually carries the information, and switch to explicit Dietz systems where SMILES would only approximate or omit it.

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
