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
make molecule-viewer VIEWER_INPUT=molecules/benzene.sdf
```

This writes a standalone HTML viewer under `results/viewer/`.

## Rule Of Thumb

- Use SDF when geometry is the source of truth.
- Use SMILES for compact classical boundary strings.
- Use MolADT JSON when another tool needs the typed molecule.

See [SMILES scope](smiles-scope-and-validation.md) for parser limits.
