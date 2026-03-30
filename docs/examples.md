# Example Molecules

This repo includes two small SDF files in `molecules/` and several built-in MolADT objects under `moladt/examples/`. The easiest inspection path is the CLI for file-backed examples and `pretty-example` for the non-classical built-ins.

## At a Glance

| Example | Where it lives | Easiest inspection | What it demonstrates | Backing |
| --- | --- | --- | --- | --- |
| Benzene | [`molecules/benzene.sdf`](../molecules/benzene.sdf), [`moladt/examples/benzene.py`](../moladt/examples/benzene.py), [`moladt/examples/benzene_pretty.py`](../moladt/examples/benzene_pretty.py) | `./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf` | A classical six-membered ring with one `pi_ring` Dietz system | Both file-backed and built-in |
| Water | [`molecules/water.sdf`](../molecules/water.sdf), [`moladt/examples/sample_molecules.py`](../moladt/examples/sample_molecules.py) | `./.venv/bin/python -m moladt.cli parse molecules/water.sdf` | A minimal classical molecule used for round-trip and SMILES tests | Both file-backed and built-in |
| Ferrocene | [`moladt/examples/ferrocene.py`](../moladt/examples/ferrocene.py), [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py) | `./.venv/bin/python -m moladt.cli pretty-example ferrocene` | Two cyclopentadienyl `pi` systems plus an Fe back-donation-style pool | Built-in object |
| Diborane | [`moladt/examples/diborane.py`](../moladt/examples/diborane.py), [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py) | `./.venv/bin/python -m moladt.cli pretty-example diborane` | Two explicit `3c-2e` bridging hydrogen systems | Built-in object |

## Benzene

- File-backed source: [`molecules/benzene.sdf`](../molecules/benzene.sdf)
- Built-in source: [`moladt/examples/benzene.py`](../moladt/examples/benzene.py)
- Pretty alias: [`moladt/examples/benzene_pretty.py`](../moladt/examples/benzene_pretty.py)

Use:

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
```

This is the main classical example for parser, validator, SDF, and conservative SMILES round-trip coverage.

## Water

- File-backed source: [`molecules/water.sdf`](../molecules/water.sdf)
- Built-in source: [`moladt/examples/sample_molecules.py`](../moladt/examples/sample_molecules.py)

Use:

```bash
./.venv/bin/python -m moladt.cli parse molecules/water.sdf
```

Water is a small sanity-check molecule used in parser and renderer tests. It stays inside the conservative SMILES subset and renders to `[OH2]`.

## Ferrocene

- Built-in source: [`moladt/examples/ferrocene.py`](../moladt/examples/ferrocene.py)
- Manuscript-facing wrapper: [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py)

Use:

```bash
./.venv/bin/python -m moladt.cli pretty-example ferrocene
```

Ferrocene demonstrates why the MolADT core is broader than the current SMILES renderer. The molecule is representable and pretty-printable in MolADT, but it is not part of the supported classical `to-smiles` subset.

## Diborane

- Built-in source: [`moladt/examples/diborane.py`](../moladt/examples/diborane.py)
- Manuscript-facing wrapper: [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py)

Use:

```bash
./.venv/bin/python -m moladt.cli pretty-example diborane
```

Diborane shows explicit multicenter bonding with two bridging hydrogen systems. Like ferrocene, it remains a MolADT example, not a supported `to-smiles` target.

## Related Example Modules

- [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py)
- [`moladt/examples/benzene.py`](../moladt/examples/benzene.py)
- [`moladt/examples/benzene_pretty.py`](../moladt/examples/benzene_pretty.py)
- [`moladt/examples/diborane.py`](../moladt/examples/diborane.py)
- [`moladt/examples/ferrocene.py`](../moladt/examples/ferrocene.py)
- [`moladt/examples/sample_molecules.py`](../moladt/examples/sample_molecules.py)
