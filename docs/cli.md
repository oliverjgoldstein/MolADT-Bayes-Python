# CLI

Run:

```bash
./.venv/bin/python -m moladt.cli --help
```

## Commands

| Command | Use |
| --- | --- |
| `parse` | Read one SDF record, validate it, and print a MolADT report. |
| `parse-smiles` | Read a supported SMILES string and print the typed molecule. |
| `to-smiles` | Render a supported classical MolADT molecule back to SMILES. |
| `to-json` | Convert one SDF molecule to shared MolADT JSON. |
| `from-json` | Read MolADT JSON back into a validated molecule. |
| `view-html` | Export a standalone HTML molecule viewer. |
| `pretty-example` | Print built-in examples such as diborane, ferrocene, or morphine. |

## Examples

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
./.venv/bin/python -m moladt.cli parse --properties molecules/benzene.sdf
./.venv/bin/python -m moladt.cli parse-smiles 'c1ccccc1'
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
./.venv/bin/python -m moladt.cli to-json molecules/benzene.sdf > benzene.moladt.json
./.venv/bin/python -m moladt.cli from-json benzene.moladt.json
./.venv/bin/python -m moladt.cli view-html molecules/benzene.sdf --output benzene.viewer.html
./.venv/bin/python -m moladt.cli pretty-example ferrocene
```

## Notes

- `parse` accepts SDF V2000 and the core V3000 CTAB subset.
- `view-html` accepts SDF or MolADT JSON.
- `to-smiles` is intentionally conservative. It does not cover non-classical multicenter examples such as diborane or ferrocene.
- Parsed SMILES stereochemistry is stored on the molecule, but the current SMILES renderer does not emit all stereo syntax back out.

Implementation files:

- [`moladt/cli.py`](../moladt/cli.py)
- [`moladt/io/sdf.py`](../moladt/io/sdf.py)
- [`moladt/io/smiles.py`](../moladt/io/smiles.py)
- [`moladt/io/molecule_json.py`](../moladt/io/molecule_json.py)
- [`moladt/viewer.py`](../moladt/viewer.py)
