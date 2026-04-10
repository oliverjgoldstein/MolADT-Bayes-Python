# MolADT-Bayes-Python

MolADT is a typed molecular data format for Bayesian work over molecules. It keeps chemically meaningful structure in the object itself instead of hiding it inside string syntax.

[Quickstart](docs/quickstart.md) · [Representation](docs/representation.md) · [Models](docs/models.md) · [Examples](docs/examples.md)

## Why MolADT

Diborane, ferrocene, and morphine show the point quickly.

- diborane in standard SMILES: `[BH2]1[H][BH2][H]1`
- ferrocene in standard SMILES: `[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]`
- morphine in standard stereochemical SMILES: `CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5`

Those are useful boundary strings, but they are weak central objects.

- diborane wants explicit `3c-2e` bridges
- ferrocene wants shared Cp/metal bonding systems
- morphine pushes fused-ring structure and stereo flags into a linear notation

MolADT keeps that chemistry as typed atoms, local bonds, bonding systems, and stereo annotations. That is the point of the repo: a molecular object that respects meaningful invariances and is easier to sample over, featurize, and compare than a string-first representation.

## Quick Start

```bash
make python-setup
./.venv/bin/python -m moladt.cli parse-smiles "c1ccccc1"
```

If you plan to run benchmarks, install CmdStan once first:

```bash
make python-cmdstan-install
```

`Molecule` is immutable and destructures as:
`atoms, local_bonds, systems, smiles_stereochemistry = molecule`

For probabilistic proposals or local graph surgery, use `MutableMolecule` as a writable scratch state and call `freeze()` to return to canonical `Molecule`.

## What This Repo Contains

- the Python MolADT types, parser, renderer, and pretty-printer
- example molecules including diborane, ferrocene, and morphine
- Stan-oriented feature generation and local benchmark tooling

## Benchmarking

The repo does not ship precomputed benchmark results. `results/` is meant to stay empty in git and only be populated by real non-dry benchmark runs on your machine.

```bash
make python-cmdstan-install
make freesolv
make qm9
make timing
make benchmark-small
```

- `make freesolv` runs the long FreeSolv benchmark from the full local `642`-molecule SDF set and writes a `Training` / `Test` / `Paper` comparison figure
- `make qm9` runs the long QM9 benchmark on the full local download with the paper-sized split and writes the same `Training` / `Test` / `Paper` figure
- `make timing` runs the separate ingest/interoperability timing pass, including raw source reads, manifest CSV field materialization, local SMILES parsing, and MolADT file parsing
- `make benchmark-small` keeps the older lighter 2,000-row QM9 subset path around for a faster local sanity check

Main outputs:

- `results/freesolv/run_.../freesolv_rmse_vs_moleculenet.svg`
- `results/qm9/paper/run_.../qm9_mae_vs_moleculenet.svg`

The predictive comparison is deliberately narrow:

- each figure shows the training score and held-out test score from the validation-selected local Stan MolADT run, then the cited MoleculeNet paper baseline
- FreeSolv compares local MolADT RMSE to the MoleculeNet MPNN RMSE row `1.15`
- QM9 `mu` compares local MolADT MAE to the MoleculeNet DTNN MAE row `2.35`

The default benchmark targets now use the long `paper` inference preset. After CmdStan is built, `make qm9` is expected to take hours, not minutes. Use `make benchmark-small` or override `INFERENCE_PRESET=default QM9_LIMIT=2000 QM9_SPLIT_MODE=subset` when you want the older lighter run.

These are local benchmark artifacts, not committed front-page snapshots. The metric matches the MoleculeNet row, but the Stan model family still differs from the paper, and FreeSolv still uses the repo's own split.

If a required raw file is too large for GitHub, the repo downloads it on demand and shows live progress for large transfers and archive extraction.

## Read More

- [Quickstart](docs/quickstart.md)
- [CLI reference](docs/cli.md)
- [Examples](docs/examples.md)
- [Models and features](docs/models.md)
- [Data sources](docs/data-sources.md)

## Related Repo

- [MolADT-Bayes-Haskell](https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell)
