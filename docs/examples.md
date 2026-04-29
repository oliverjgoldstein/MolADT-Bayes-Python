# Examples

Example molecules live in `molecules/` and `moladt/examples/`.

## Quick Commands

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
./.venv/bin/python -m moladt.cli pretty-example diborane
./.venv/bin/python -m moladt.cli pretty-example ferrocene
./.venv/bin/python -m moladt.cli pretty-example morphine
make molecule-viewer VIEWER_INPUT=molecules/benzene.sdf
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
| Benzene | `molecules/benzene.sdf`, `moladt/examples/benzene.py` |
| Water | `molecules/water.sdf`, `moladt/examples/sample_molecules.py` |
| Diborane | `molecules/diborane.sdf`, `moladt/examples/diborane.py` |
| Ferrocene | `molecules/ferrocene.sdf`, `moladt/examples/ferrocene.py` |
| Morphine | `molecules/morphine.sdf`, `moladt/examples/morphine.py` |

## What To Notice

- SMILES is a boundary format.
- MolADT is the working object.
- Non-classical bonding is explicit in `systems`, not squeezed into a string.
- The same object can be validated, viewed, serialized, featurized, and scored.

Inspect one built-in example:

```python
from moladt.chem.validate import validate_molecule
from moladt.examples import diborane_pretty

molecule = validate_molecule(diborane_pretty)

for system_id, system in molecule.systems:
    print(system_id.value, system.tag, system.shared_electrons.value)
```

See also [Representation](representation.md), [Parsing](parsing.md), and [CLI](cli.md).
