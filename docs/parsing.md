# Parsing and Rendering

This is the shortest way to see MolADT directly.

## If You Have an SDF File

Use `parse`.

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
```

This reads the file-backed molecule, validates it, prints the MolADT structure, and keeps SDF metadata such as the title and properties block when present.

Use `parse` when the source of truth is a structure file.

## If You Have a SMILES String

Use `parse-smiles`.

```bash
./.venv/bin/python -m moladt.cli parse-smiles "c1ccccc1"
```

This parses the conservative SMILES subset, validates it, and prints the MolADT structure.

Use `parse-smiles` when you want to see what a SMILES string becomes inside MolADT.

## If You Want SMILES Back Out

Use `to-smiles`.

```bash
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
```

This renders a validated MolADT molecule back to the supported classical SMILES subset.

Not every MolADT molecule can be rendered to SMILES. That is a feature, not a bug: MolADT is broader than the current renderer.

## If You Want a Rich Example

Use `pretty-example`.

```bash
./.venv/bin/python -m moladt.cli pretty-example ferrocene
./.venv/bin/python -m moladt.cli pretty-example diborane
```

These examples show the point of the representation more clearly than benzene does.

## SDF vs SMILES in One Sentence

- SDF parsing starts from a structure-backed molecule and preserves geometry-first context.
- SMILES parsing starts from a compact text notation and lifts it into MolADT.

## Technical Reference

For the full CLI reference, renderer limits, and implementation files, see [CLI](cli.md).
