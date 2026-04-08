# MolADT-Bayes-Python

MolADT is a typed molecular data format. The goal is to represent molecules in a way that exposes chemically meaningful structure and invariances, especially when doing Bayesian sampling over molecules or building models over molecular structure.

Start here: [Quickstart](docs/quickstart.md)

## Example

Diborane, ferrocene, and morphine show the boundary quickly.

- diborane in standard SMILES: `[BH2]1[H][BH2][H]1`
- ferrocene in standard SMILES: `[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]`
- morphine in standard stereochemical SMILES: `CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5`

Those are useful boundary strings, but they are poor central representations.

- diborane wants two explicit `3c-2e` bridge systems
- ferrocene wants shared Cp/metal bonding systems, not disconnected ionic fragments
- morphine still pushes fused-ring bookkeeping and five atom-centered stereo flags into string syntax

MolADT instead stores that chemistry directly in typed data. The point of the project is to put molecules into a format whose structure respects chemically meaningful invariances. SMILES is fine at the edge, but it is notation-dependent, folds chemistry into syntax, and becomes awkward exactly where richer molecular structure starts to matter. In the morphine example, the explicit MolADT object keeps the fused sigma graph direct and preserves the parsed SMILES stereo flags beside it.

## What This Repo Contains

- the Python MolADT types, parser, renderer, and pretty-printer
- example molecules including diborane, ferrocene, and morphine
- Bayesian modeling and experiment tooling built around the representation

For the representation itself:

- [MolADT representation](docs/representation.md)
- [Orbitals and theoretical chemistry](docs/orbitals.md)
- [Parsing and rendering](docs/parsing.md)
- [Models and features](docs/models.md)

## Quick Start

```bash
make python-setup
./.venv/bin/python -m moladt.cli parse-smiles "c1ccccc1"
```

`Molecule` is immutable and destructures as:
`atoms, local_bonds, systems, smiles_stereochemistry = molecule`

For probabilistic proposals or local graph surgery, use `MutableMolecule` as a writable scratch state and call `freeze()` to return to canonical `Molecule`.

## Benchmarking

All benchmark commands live here:

```bash
make freesolv
make qm9
make timing
```

- `make freesolv` is the lightest end-to-end run
- `make qm9` runs the focused dipole benchmark
- `make timing` measures SMILES vs MolADT ingest cost

Outputs are written under `results/`.

If a required raw file is too large for GitHub, the repo downloads it on demand. Large transfers and archive extraction show live byte counts, entry counts, throughput, and elapsed time.

## Read More

- [Quickstart](docs/quickstart.md)
- [CLI reference](docs/cli.md)
- [Examples](docs/examples.md)
- [Models and features](docs/models.md)
- [Data sources](docs/data-sources.md)

## Related Repo

- [MolADT-Bayes-Haskell](https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell)
