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

## Morphine and Ring Closures

The classic morphine figure takes the opposite route from diborane. It stays classical, but it still needs SMILES ring-closure machinery to talk about the fused skeleton:

```text
O1C2C(O)C=C(C3C2(C4)C5c1c(O)ccc5CC3N(C)C4)
```

Those digits are serialization backreferences. They mean:

- `1` reconnects `O#1` and `C#11`
- `2` reconnects `C#2` and `C#8`
- `3` reconnects `C#7` and `C#18`
- `4` reconnects `C#9` and `C#21`
- `5` reconnects `C#10` and `C#16`

In the built-in morphine example at [`moladt/examples/morphine.py`](../moladt/examples/morphine.py), those same five closures are already ordinary `local_bonds`. The explicit Dietz object then goes beyond the boundary string:

- `C#5 <-> C#6` is an explicit two-electron system tagged `alkene_bridge`
- the phenyl fragment `C#10-C#11-C#12-C#14-C#15-C#16` is an explicit six-electron system tagged `phenyl_pi_ring`

That is the direct comparison to the image. The image explains how to turn a fused polycycle into SMILES text. MolADT stores the fused polycycle and its delocalization directly, without ring digits as an internal representation.

Use:

```bash
./.venv/bin/python -m moladt.cli pretty-example morphine
./.venv/bin/python -m moladt.cli parse-smiles "O1C2C(O)C=C(C3C2(C4)C5c1c(O)ccc5CC3N(C)C4)"
```

The first command shows the explicit Dietz object. The second shows the boundary-format parse path based on the same figure.

## Side-by-Side Examples

| Example | SMILES Side | MolADT Side |
| --- | --- | --- |
| Diborane | No faithful conservative SMILES string in this repo. A classical SMILES graph would flatten or hide the two `3c-2e` bridges. | [`moladt/examples/diborane.py`](../moladt/examples/diborane.py) stores 5 ordinary sigma edges and 2 explicit Dietz bridge systems: `bridge_h3_3c2e` and `bridge_h4_3c2e`. |
| Ferrocene | No faithful conservative SMILES string in this repo. Standard classical SMILES does not encode the Fe-centered shared pools used here. | [`moladt/examples/ferrocene.py`](../moladt/examples/ferrocene.py) stores the sandwich graph directly, plus `cp1_pi`, `cp2_pi`, and `fe_backdonation` Dietz systems. |
| Morphine | `O1C2C(O)C=C(C3C2(C4)C5c1c(O)ccc5CC3N(C)C4)` expresses the fused graph through ring digits and aromatic lowercase syntax. | [`moladt/examples/morphine.py`](../moladt/examples/morphine.py) stores the same fused graph directly in `local_bonds`, then makes the delocalization explicit with `alkene_bridge` and `phenyl_pi_ring`. |

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
