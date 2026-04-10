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

## Ferrocene in Code

The built-in ferrocene object is intentionally almost identical in Python and Haskell. Same atom ids, same sigma framework, same three Dietz systems.

Python:

```python
fe = AtomId(1)
ring1_c = tuple(AtomId(index) for index in range(2, 7))
ring2_c = tuple(AtomId(index) for index in range(7, 12))
ring1_h = tuple(AtomId(index) for index in range(12, 17))
ring2_h = tuple(AtomId(index) for index in range(17, 22))

ring1_cc = _ring_pairs(ring1_c)
ring2_cc = _ring_pairs(ring2_c)
ring1_ch = tuple(zip(ring1_c, ring1_h))
ring2_ch = tuple(zip(ring2_c, ring2_h))
fe_to_ring1 = tuple((fe, atom_id) for atom_id in ring1_c)
fe_to_ring2 = tuple((fe, atom_id) for atom_id in ring2_c)

ferrocene_pretty = Molecule(
    local_bonds=frozenset(mk_edge(a, b) for a, b in ring1_cc + ring2_cc + ring1_ch + ring2_ch),
    systems=(
        (SystemId(1), mk_bonding_system(NonNegative(6), frozenset(mk_edge(a, b) for a, b in fe_to_ring1 + ring1_cc), "cp1_pi")),
        (SystemId(2), mk_bonding_system(NonNegative(6), frozenset(mk_edge(a, b) for a, b in fe_to_ring2 + ring2_cc), "cp2_pi")),
        (SystemId(3), mk_bonding_system(NonNegative(6), frozenset(mk_edge(a, b) for a, b in fe_to_ring1 + fe_to_ring2), "fe_backdonation")),
    ),
)
```

Haskell:

```haskell
fe      = AtomId 1
ring1C  = AtomId <$> [2..6]
ring2C  = AtomId <$> [7..11]
ring1H  = AtomId <$> [12..16]
ring2H  = AtomId <$> [17..21]

ring1CCPairs = ringPairs ring1C
ring2CCPairs = ringPairs ring2C
ring1CHPairs = zip ring1C ring1H
ring2CHPairs = zip ring2C ring2H
feToRing1    = [(fe, c) | c <- ring1C]
feToRing2    = [(fe, c) | c <- ring2C]

ferrocenePretty = Molecule
  { localBonds = mkEdges (ring1CCPairs ++ ring2CCPairs ++ ring1CHPairs ++ ring2CHPairs)
  , systems =
      [ (SystemId 1, mkBondingSystem (NonNegative 6) (mkEdges (feToRing1 ++ ring1CCPairs)) (Just "cp1_pi"))
      , (SystemId 2, mkBondingSystem (NonNegative 6) (mkEdges (feToRing2 ++ ring2CCPairs)) (Just "cp2_pi"))
      , (SystemId 3, mkBondingSystem (NonNegative 6) (mkEdges (feToRing1 ++ feToRing2)) (Just "fe_backdonation"))
      ]
  }
```

The close alignment is deliberate:

- atom `#1` is `Fe` in both repos
- atoms `#2..#6` and `#7..#11` are the two Cp rings in both repos
- `local_bonds` or `localBonds` contains only the localized `C-C` and `C-H` sigma framework
- the same three six-electron Dietz systems appear in both repos: `cp1_pi`, `cp2_pi`, and `fe_backdonation`
- both repos use canonical undirected Dietz edges via `mk_edge` or `mkEdge`; the representation is not turned into a duplicated multigraph just to make downstream helpers easier

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
| Ferrocene | `[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]` is a standard boundary string, but it splits the sandwich into ionic fragments instead of shared Cp/metal pools. | [`moladt/examples/ferrocene.py`](../moladt/examples/ferrocene.py) stores the sandwich graph directly, plus the aligned `cp1_pi`, `cp2_pi`, and `fe_backdonation` Dietz systems used in the Haskell example too. |
| Morphine | `CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5` expresses the fused graph through ring digits, localized double-bond syntax, and five atom-centered stereo flags. | [`moladt/examples/morphine.py`](../moladt/examples/morphine.py) stores the same fused graph directly in `local_bonds`, keeps the alkene and phenyl delocalization explicit, and preserves the five atom-centered SMILES flags in `smiles_stereochemistry`. |

That is the recurring boundary in this repo: use SMILES where the notation actually carries the information, and switch to explicit Dietz systems where SMILES would only approximate or omit it.

## Benchmark Reminder

The benchmark object is just `moladt`: the typed MolADT representation built from the boundary input.

## What We Are Testing

The current benchmark asks one narrow question:

- if we fit the Stan MolADT models and keep the best local run, how does that MolADT result compare with the matching MoleculeNet row for FreeSolv or QM9?

## See Also

- [Parsing and Rendering](parsing.md)
- [Models and Benchmarks](models.md)
- [Outputs](outputs.md)
