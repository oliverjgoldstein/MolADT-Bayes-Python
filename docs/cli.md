# CLI

The Python CLI entrypoint is:

```bash
./.venv/bin/python -m moladt.cli --help
```

It currently exposes four subcommands.

## `parse`

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
```

What it does:

- reads one SDF record from the given path
- accepts SDF V2000 and the core V3000 CTAB subset
- validates the resulting MolADT structure with `validate_molecule`
- prints `Title: ...`
- prints the pretty-printed molecule
- prints a `Properties:` section if the SDF record contains properties

Use `parse` when the source of truth is a file-backed molecule.

The current V3000 support is intentionally narrow: atom coordinates, bond tables, atom-local formal charges, and trailing property blocks. The local writer still emits V2000.

## `parse-smiles`

```bash
./.venv/bin/python -m moladt.cli parse-smiles '<smiles>'
```

What it validates:

- the conservative SMILES grammar implemented in [`moladt/io/smiles.py`](../moladt/io/smiles.py)
- atom, bond, and ring-closure syntax inside that subset
- terminal-hydrogen inference for supported bare atoms
- six-membered `pi_ring` recovery from aromatic lowercase syntax
- atom-centered `@`/`@@` and bond-directed `/` `\` annotations, stored on `smiles_stereochemistry`
- structural validation through `validate_molecule`, including valence checks and bond-map consistency

On success it prints the pretty-printed MolADT structure. It does not print a title or property block because the source is a SMILES string, not an SDF record.

## `to-smiles`

```bash
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
```

What it accepts:

- validated molecules in the conservative classical subset
- localized single, double, and triple bonds
- six-edge `pi_ring` systems that can be rendered as aromatic or deterministic Kekule-style output

Current limitation:

- stored `smiles_stereochemistry` annotations are preserved on parse, but the renderer does not yet emit `@`, `@@`, `/`, or `\`

What it rejects:

- empty molecules
- structures where rendering would require dropping bonded atoms
- structures outside the supported classical subset, including non-classical multicenter systems like diborane and ferrocene
- components that would need more than 9 ring closures

Current rejection messages come directly from the SMILES renderer, for example:

- `SMILES rendering only supports localized double/triple bonds and six-edge pi rings`
- `pi_ring must be a simple six-membered cycle to render as SMILES`

## `pretty-example`

```bash
./.venv/bin/python -m moladt.cli pretty-example ferrocene
./.venv/bin/python -m moladt.cli pretty-example diborane
./.venv/bin/python -m moladt.cli pretty-example morphine
./.venv/bin/python -m moladt.cli pretty-example psilocybin
```

This command loads named built-in examples from [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py), validates them, and prints the manuscript-facing pretty rendering. It currently supports `ferrocene`, `diborane`, `morphine`, and `psilocybin`.

## How the Commands Differ

- `parse` starts from an SDF file and can print record title and properties.
- `parse-smiles` starts from a SMILES string and only prints the validated MolADT structure.
- `to-smiles` starts from an SDF file and emits only the rendered SMILES string.
- `pretty-example` starts from a built-in example object written as an explicit typed molecule with orbital shells intact.

## Related Files

- [`moladt/cli.py`](../moladt/cli.py)
- [`moladt/io/sdf.py`](../moladt/io/sdf.py)
- [`moladt/io/smiles.py`](../moladt/io/smiles.py)
- [`moladt/chem/validate.py`](../moladt/chem/validate.py)
