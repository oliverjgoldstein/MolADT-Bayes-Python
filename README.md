# MolADT-Bayes-Python

MolADT is a typed molecular data format for Bayesian work over molecules. It keeps chemically meaningful structure in the object itself instead of hiding it inside string syntax.

[Quickstart](docs/quickstart.md) · [Representation](docs/representation.md) · [Models](docs/models.md) · [Examples](docs/examples.md)

## Why MolADT

Diborane, ferrocene, and morphine are useful examples.

- diborane in standard SMILES: `[BH2]1[H][BH2][H]1`
- ferrocene in standard SMILES: `[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]`
- morphine in standard stereochemical SMILES: `CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5`

Those are useful boundary strings, but they are weak central objects.

- diborane wants explicit `3c-2e` bridges
- ferrocene wants shared Cp/metal bonding systems
- morphine pushes fused-ring structure and stereo flags into a linear notation

MolADT keeps that chemistry as typed atoms, local bonds, bonding systems, and stereo annotations. The repo is built around that object so that functions over molecules can respect the invariances of the molecular structure, rather than the accidental choices made by a boundary string syntax.

## Quick Start

```bash
make python-setup
./.venv/bin/python -m moladt.cli parse-smiles "c1ccccc1"
# once, before Stan benchmarks
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

Benchmark outputs stay local. `results/` is generated on your machine rather than committed.

```bash
make freesolv
make qm9
make timing
make benchmark-small
```

Current benchmark sizes:

- FreeSolv benchmark path: the SDF-backed `moladt_featurized` export uses all `642` local SDF molecules and splits them into `513` train / `64` validation / `65` test
- QM9 default long run: `110,462` train / `10,000` validation / `10,000` test
- `benchmark-small`: the lighter QM9 subset path uses `1,600` train / `200` validation / `200` test

- `make freesolv` writes `results/freesolv/run_.../freesolv_rmse_vs_moleculenet.svg`
- `make qm9` writes `results/qm9/paper/run_.../qm9_mae_vs_moleculenet.svg`
- `make timing` writes `results/timing/paper/run_.../timing_overview.svg`
- the FreeSolv figure shows `Training`, `Validation`, `Test`, then `Paper`
- the QM9 figure shows `Training`, `Test`, then `Paper`
- `make freesolv` runs one fixed benchmark path: `moladt_featurized` with `bayes_gp_rbf_screened` fit by `laplace`
- FreeSolv compares local MolADT RMSE to the MoleculeNet MPNN RMSE row `1.15`
- QM9 `mu` compares local MolADT MAE to the MoleculeNet DTNN MAE row `2.35`

The default benchmark targets use the long `paper` inference preset. After CmdStan is built, `make qm9` is expected to take hours, not minutes. Use `make benchmark-small` or override `INFERENCE_PRESET=default QM9_LIMIT=2000 QM9_SPLIT_MODE=subset` for a lighter run.

The metric matches the cited MoleculeNet row, but the Stan model family still differs from the paper, and FreeSolv still uses the repo's own split. Large raw files are downloaded on demand when they are not already vendored.

Read the timing overview as a pipeline rather than as one aggregate score:

- `raw_file_read`: how fast the benchmark can pull SMILES text out of the ZINC source file without doing chemistry work
- `smiles_parse_sanitize`: how fast an external chemistry toolkit can parse each SMILES string into a molecular graph
- `smiles_canonicalization`: how fast that toolkit can re-render those molecules into one canonical SMILES form
- `timing_library_prepare`: the one-time cost to build the matched local timing corpus, so later stages use the same molecule count on both sides
- `smiles_csv_string_parse`: the plain-string baseline, where the canonical SMILES field is only materialized from CSV into a Python string
- `smiles_library_parse`: the local MolADT SMILES parser running on those same canonical strings
- `moladt_file_parse`: the local MolADT JSON file reader running on the paired MolADT files

In practice, `smiles_csv_string_parse` is the near-zero "just hand me the string" floor, `smiles_library_parse` is the cost of building a MolADT object from SMILES text, and `moladt_file_parse` is the cost of reading the already-structured MolADT form from disk.

## Read More

- [Quickstart](docs/quickstart.md)
- [CLI reference](docs/cli.md)
- [Examples](docs/examples.md)
- [Models and features](docs/models.md)
- [Data sources](docs/data-sources.md)

## Related Repo

- [MolADT-Bayes-Haskell](https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell)
