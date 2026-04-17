# Parsing and Rendering

This is the shortest way to see MolADT directly.

## If You Have an SDF File

Use `parse`.

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
```

This reads the file-backed molecule, validates it, and prints a minimal MolADT report plus the SDF title.

If you also want the raw SDF metadata:

```bash
./.venv/bin/python -m moladt.cli parse --properties molecules/benzene.sdf
```

That adds the trailing SDF `Properties:` block.

The SDF parser accepts:

- V2000 records
- the core V3000 CTAB subset with atom coordinates, bond tables, and atom-local formal charges

The writer remains V2000-only.

If the SDF payload contains several molecules, use `from moladt.io.sdf import read_sdf_records; titles = [record.title for record in read_sdf_records("bundle.sdf", limit=5)]` or iterate lazily with `from moladt.io.sdf import iter_sdf_records; first = next(iter_sdf_records("bundle.sdf"))`.

Use `parse` when the source of truth is a structure file.

## If You Want MolADT JSON

Use `to-json`.

```bash
./.venv/bin/python -m moladt.cli to-json molecules/benzene.sdf > benzene.moladt.json
```

This reads the SDF file, validates the resulting `Molecule`, and writes the shared MolADT JSON boundary format.

If you want the same flow in Python code:

```python
from moladt.io import molecule_to_json, read_sdf_record

record = read_sdf_record("molecules/benzene.sdf")
payload = molecule_to_json(record.molecule)
```

That JSON payload is also accepted by the Haskell repo's `from-json` command.

## If You Want MolADT JSON Back Into the Typed Object

Use `from-json`.

```bash
./.venv/bin/python -m moladt.cli from-json benzene.moladt.json
```

This reads the shared MolADT JSON payload, rebuilds the typed `Molecule`, validates it, and prints the usual MolADT report.

If you want the same step in Python code:

```python
from moladt.io import molecule_from_json

molecule = molecule_from_json(open("benzene.moladt.json", "rb").read())
```

## If You Have a SMILES String

Use `parse-smiles`.

```bash
./.venv/bin/python -m moladt.cli parse-smiles '<smiles>'
```

This parses the conservative SMILES subset, validates it, and prints the MolADT structure.

Inside that boundary, `parse-smiles` does three useful lifts after reading the graph:

- it infers terminal hydrogens for supported bare atoms such as `C`, `N`, `O`, and aromatic lowercase atoms
- it promotes recoverable six-membered delocalized cycles into explicit Dietz `pi_ring` systems when the SMILES uses aromatic lowercase syntax, including ring-closure edges
- it preserves atom-centered `@`/`@@` and bond-directed `/` `\` annotations in `smiles_stereochemistry`

Use `parse-smiles` when you want to inspect a boundary string. The example molecules shipped with this repo are all SDF-backed instead.

## If You Want SMILES Back Out

Use `to-smiles`.

```bash
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
```

This renders a validated MolADT molecule back to the supported classical SMILES subset.

Not every MolADT molecule can be rendered to SMILES. That is a feature, not a bug: MolADT is broader than the current renderer.
The parser keeps SMILES stereochemistry flags on the MolADT object, but the current renderer does not yet synthesize those flags back out.

## If You Want a Rich Example

Use `pretty-example`.

```bash
./.venv/bin/python -m moladt.cli pretty-example ferrocene
./.venv/bin/python -m moladt.cli pretty-example diborane
./.venv/bin/python -m moladt.cli pretty-example morphine
```

These examples show the point of the representation more clearly than benzene does.

## SDF vs SMILES in One Sentence

- SDF parsing starts from a structure-backed molecule and preserves geometry-first context.
- SMILES parsing starts from a compact text notation and lifts it into MolADT.

## Technical Reference

For the full CLI reference, renderer limits, and implementation files, see [CLI](cli.md).
