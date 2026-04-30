# Examples

Example molecules live as explicit Python ADTs in `moladt/examples/`. The `molecules/` SDF files are parser fixtures and conversion inputs, not the source used by the viewer examples.

## Quick Commands

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
./.venv/bin/python -m moladt.cli pretty-example diborane
./.venv/bin/python -m moladt.cli pretty-example ferrocene
./.venv/bin/python -m moladt.cli pretty-example morphine
make molecule-viewer VIEWER_EXAMPLES=ferrocene
```

## Main Examples

| Molecule | Why it is here |
| --- | --- |
| Benzene | Classical ring plus a six-electron `pi_ring`. |
| Water | Small parser, validation, and SMILES sanity check. |
| Diborane | Two explicit `3c-2e` bridge systems. |
| Ferrocene | Organometallic Cp/metal bonding systems. |
| Morphine | Fused graph, delocalization, and stored stereochemistry flags. |

## Where They Live

| Molecule | Files |
| --- | --- |
| Benzene | `moladt/examples/benzene.py` |
| Water | `moladt/examples/sample_molecules.py` |
| Diborane | `moladt/examples/diborane.py` |
| Ferrocene | `moladt/examples/ferrocene.py` |
| Morphine | `moladt/examples/morphine.py` |

## What To Notice

- SMILES is a boundary format.
- MolADT is the working object.
- Non-classical bonding is explicit in `systems`, not squeezed into a string.
- The same object can be validated, viewed, serialized, featurized, and scored.
- Checked-in examples are expanded as literal `AtomId`, `Edge`, and `BondingSystem` values. They do not hide their atoms or bonds behind loops.

The examples use this shape:

```python
from __future__ import annotations

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import AtomicSymbol, Molecule
from ._literal import atom


benzene = Molecule(
    atoms={
        AtomId(1): atom(1, AtomicSymbol.C, 2.866, 1.000, 0.000),
        AtomId(2): atom(2, AtomicSymbol.C, 2.000, 0.500, 0.000),
        AtomId(3): atom(3, AtomicSymbol.C, 3.732, 0.500, 0.000),
        AtomId(4): atom(4, AtomicSymbol.C, 2.000, -0.500, 0.000),
        AtomId(5): atom(5, AtomicSymbol.C, 3.732, -0.500, 0.000),
        AtomId(6): atom(6, AtomicSymbol.C, 2.866, -1.000, 0.000),
        AtomId(7): atom(7, AtomicSymbol.H, 2.866, 1.620, 0.000),
        AtomId(8): atom(8, AtomicSymbol.H, 1.463, 0.810, 0.000),
        AtomId(9): atom(9, AtomicSymbol.H, 4.269, 0.810, 0.000),
        AtomId(10): atom(10, AtomicSymbol.H, 1.463, -0.810, 0.000),
        AtomId(11): atom(11, AtomicSymbol.H, 4.269, -0.810, 0.000),
        AtomId(12): atom(12, AtomicSymbol.H, 2.866, -1.620, 0.000),
    },
    local_bonds=frozenset(
        {
            Edge(AtomId(1), AtomId(2)),
            Edge(AtomId(1), AtomId(6)),
            Edge(AtomId(1), AtomId(7)),
            Edge(AtomId(2), AtomId(3)),
            Edge(AtomId(2), AtomId(8)),
            Edge(AtomId(3), AtomId(4)),
            Edge(AtomId(3), AtomId(9)),
            Edge(AtomId(4), AtomId(5)),
            Edge(AtomId(4), AtomId(10)),
            Edge(AtomId(5), AtomId(6)),
            Edge(AtomId(5), AtomId(11)),
            Edge(AtomId(6), AtomId(12)),
        }
    ),
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(6),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(2)),
                        Edge(AtomId(1), AtomId(6)),
                        Edge(AtomId(2), AtomId(3)),
                        Edge(AtomId(3), AtomId(4)),
                        Edge(AtomId(4), AtomId(5)),
                        Edge(AtomId(5), AtomId(6)),
                    }
                ),
                "pi_ring",
            ),
        ),
    ),
)
```

Inspect one built-in example:

```python
from moladt.chem.validate import validate_molecule
from moladt.examples import diborane_pretty

molecule = validate_molecule(diborane_pretty)

for system_id, system in molecule.systems:
    print(system_id.value, system.tag, system.shared_electrons.value)
```

Convert an SDF parser fixture into the same explicit shape:

```bash
./.venv/bin/python -m moladt.cli to-python molecules/benzene.sdf --name benzene_generated
./.venv/bin/python -m moladt.cli to-example molecules/benzene.sdf --name benzene_generated --output moladt/examples/benzene_generated.py
```

See also [Representation](representation.md), [Parsing](parsing.md), and [CLI](cli.md).
