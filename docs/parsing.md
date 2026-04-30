# Parsing

MolADT can start from SDF, SMILES, or MolADT JSON.

## SDF

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
./.venv/bin/python -m moladt.cli parse --properties molecules/benzene.sdf
```

The parser accepts SDF V2000 and a core V3000 CTAB subset: atom coordinates, bond tables, atom-local charges, and property blocks.

## MolADT JSON

```bash
./.venv/bin/python -m moladt.cli to-json molecules/benzene.sdf > benzene.moladt.json
./.venv/bin/python -m moladt.cli from-json benzene.moladt.json
```

MolADT JSON is the shared boundary format used by the Python and Haskell repos.

## Explicit Python ADT

```bash
./.venv/bin/python -m moladt.cli to-python molecules/benzene.sdf --name benzene_generated > benzene_generated.py
./.venv/bin/python -m moladt.cli to-example molecules/benzene.sdf --name benzene_generated --output moladt/examples/benzene_generated.py
```

`to-python` and its `to-example` alias convert parsed geometry into canonical expanded `Molecule(...)` source with literal `AtomId(...)`, `Edge(...)`, and `mk_bonding_system(...)` entries. Atoms are sorted by `AtomId`, edges are normalized and sorted, and systems are sorted by `SystemId`. The output is meant for checked examples and reviewable fixtures, so it does not hide atoms or bonds behind generated loops.

Python API:

```python
from moladt.chem.validate import validate_molecule
from moladt.io.molecule_json import molecule_from_json, molecule_to_json
from moladt.io.sdf import read_sdf
from moladt.io.smiles import molecule_to_smiles

molecule = validate_molecule(read_sdf("molecules/benzene.sdf"))
payload = molecule_to_json(molecule)
round_tripped = validate_molecule(molecule_from_json(payload))

print(molecule_to_smiles(round_tripped))
```

## SMILES

```bash
./.venv/bin/python -m moladt.cli parse-smiles 'c1ccccc1'
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
```

The SMILES path is conservative. It covers the classical subset used by tests and examples. It does not try to encode every MolADT molecule.

## Viewer

```bash
make molecule-viewer VIEWER_EXAMPLES=ferrocene
./.venv/bin/python -m moladt.cli view-html benzene.moladt.json --output benzene.viewer.html
```

This writes a standalone HTML viewer under `results/viewer/`. The Make target uses built-in explicit ADT examples. The direct `view-html` command accepts MolADT JSON. Click an atom in the viewer to show its stored shell and orbital glyphs, coordinates, 3D edge lengths, and bond angles calculated from the molecule's coordinate data.

## Rule Of Thumb

- Use SDF only as an import boundary or parser fixture.
- Use SMILES for compact classical boundary strings.
- Use explicit Python ADTs for checked examples.
- Use MolADT JSON when another tool or the viewer needs the typed molecule as a file.

See [SMILES scope](smiles-scope-and-validation.md) for parser limits.
