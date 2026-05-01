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
| `to-python` | Convert one SDF molecule to explicit Python `Molecule(...)` source. |
| `to-example` | Alias for `to-python`, useful when writing checked example modules. |
| `from-json` | Read MolADT JSON back into a validated molecule. |
| `view-html` | Export a standalone HTML molecule viewer from MolADT JSON. |
| `view-examples` | Export the built-in ADT examples in one viewer while preserving their bonding systems. |
| `pretty-example` | Print built-in examples such as diborane, ferrocene, or morphine. |

## Examples

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
./.venv/bin/python -m moladt.cli parse --properties molecules/benzene.sdf
./.venv/bin/python -m moladt.cli parse-smiles 'c1ccccc1'
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
./.venv/bin/python -m moladt.cli to-json molecules/benzene.sdf > benzene.moladt.json
./.venv/bin/python -m moladt.cli to-python molecules/benzene.sdf --name benzene_generated > benzene_generated.py
./.venv/bin/python -m moladt.cli to-example molecules/benzene.sdf --name benzene_generated --output moladt/examples/benzene_generated.py
./.venv/bin/python -m moladt.cli from-json benzene.moladt.json
./.venv/bin/python -m moladt.cli view-html benzene.moladt.json --output benzene.viewer.html
./.venv/bin/python -m moladt.cli view-html benzene.moladt.json --output benzene.viewer.html --open-viewer
./.venv/bin/python -m moladt.cli view-examples --output examples.viewer.html --open-viewer
./.venv/bin/python -m moladt.cli pretty-example ferrocene
./.venv/bin/python -m moladt.cli pretty-example ferrocene --open-viewer
```

## Notes

- `parse` accepts SDF V2000 and the core V3000 CTAB subset.
- `to-python` and `to-example` emit the canonical explicit form: sorted `AtomId(...)` atoms, normalized sorted `Edge(...)` entries, sorted systems, and no generated loops in the output.
- `view-html` accepts MolADT JSON. Pass multiple JSON files to write one HTML page with a scrollable molecule list.
- `view-examples` uses the built-in ADT examples, preserving diborane, ferrocene, benzene, and morphine bonding systems.
- `--open-viewer` opens the written HTML viewer in the default browser.
- `pretty-example --open-viewer` still prints the report, then writes and opens `results/viewer/<example>.viewer.html`.
- `to-smiles` is intentionally conservative. It does not cover non-classical multicenter examples such as diborane or ferrocene.

Implementation files:

- [`moladt/cli.py`](../moladt/cli.py)
- [`moladt/io/sdf.py`](../moladt/io/sdf.py)
- [`moladt/io/smiles.py`](../moladt/io/smiles.py)
- [`moladt/io/molecule_json.py`](../moladt/io/molecule_json.py)
- [`moladt/viewer.py`](../moladt/viewer.py)
