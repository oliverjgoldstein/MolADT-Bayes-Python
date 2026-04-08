# MolADT-Bayes-Python

MolADT is a chemistry-first molecular ADT for building models over molecules. It keeps atoms, sigma bonds, explicit bonding systems, coordinates, shells, and orbitals visible instead of flattening everything into a serializer-oriented string.

Start here: [Quickstart](docs/quickstart.md)

## Example

Standard SMILES for diborane:
`[BH2]1[H][BH2][H]1`

MolADT can keep the chemistry more directly:

- two boron atoms
- two terminal hydrogens
- two explicit `3c-2e` bridge systems

That is the point of the project: keep SMILES at the boundary, keep the chemistry object in the center, and make that object usable for models over molecules.

## What This Repo Contains

- the Python MolADT types, parser, renderer, and pretty-printer
- example molecules including diborane, ferrocene, and morphine
- model and experiment tooling built around the representation

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
