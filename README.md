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

## Parsing

Use the CLI when you want to inspect how a boundary format lands inside MolADT.

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
./.venv/bin/python -m moladt.cli parse-smiles "c1ccccc1"
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
```

- `parse` reads one SDF record, validates it, prints the MolADT structure, and preserves SDF title and property fields
- `parse-smiles` reads the conservative SMILES subset and lifts it into the typed MolADT object
- `to-smiles` renders validated classical MolADT structures back into the supported SMILES subset

The SDF reader accepts V2000 and the core V3000 CTAB subset used by common structure exports:

- atom coordinates
- bond tables
- atom-local formal charges
- trailing SDF property blocks

The writer still emits V2000. The parser and renderer are intentionally narrower than the full MDL feature surface.

If one SDF file contains multiple molecules:

- read a small eager slice with `from moladt.io.sdf import read_sdf_records; titles = [record.title for record in read_sdf_records("bundle.sdf", limit=3)]`
- stream lazily with `from moladt.io.sdf import iter_sdf_records; first_titles = [record.title for record in iter_sdf_records("bundle.sdf", limit=3)]`

The local QM9 and vendored FreeSolv raw files in this workspace are still V2000. The downloader and parser prefer V3000 when a future dataset bundle actually provides it, but the current local raws are not being silently relabeled.

## What This Repo Contains

- the Python MolADT types, parser, renderer, and pretty-printer
- example molecules including diborane, ferrocene, and morphine
- Stan-oriented feature generation and local benchmark tooling

## Benchmarking

Benchmark outputs stay local. `results/` is generated on your machine rather than committed.

```bash
make freesolv
make qm9long
make timing
```

- `make freesolv`: FreeSolv RMSE comparison. Fixed path `moladt_featurized + bayes_gp_rbf_screened + laplace`. Writes `results/freesolv/run_.../freesolv_rmse_vs_moleculenet.svg`.
- `make qm9long`: full-data QM9 `mu` MAE comparison over all aligned local QM9 molecules, with `catboost_uncertainty` on the SDF-backed `moladt_featurized` tabular export and `visnet_ensemble` on the geometry exports. The current local bundle yields `107,108 / 13,388 / 13,389` train / validation / test rows under the deterministic `80/10/10` long split. ViSNet runs one member, caps geometry training at `25` epochs, uses seed `102` (the second seed from the old QM9 seed schedule), and prints every epoch when `BENCHMARK_VERBOSE=1`. Writes `results/qm9/long/run_.../qm9_mae_vs_moleculenet.svg`.
- `make timing`: ZINC ingest and runtime comparison. It separates raw I/O, optional external-toolkit stages, a plain string baseline, our SMILES parser, and our MolADT file reader. Writes `results/timing/paper/run_.../timing_overview.svg`.

Results are written under timestamped directories in `results/`, mainly `results/freesolv/run_.../`, `results/qm9/long/run_.../`, and `results/timing/paper/run_.../`.

The FreeSolv figure shows `Training`, `Validation`, `Test`, and `Paper`. The QM9 figure shows `Training`, `Test`, and `Paper`.

For split sizes, the exact benchmark contract, and the detailed timing-stage definitions, see [Inference and benchmarks](docs/inference-and-benchmarks.md) and [Outputs](docs/outputs.md).

## Read More

- [Quickstart](docs/quickstart.md)
- [Inference and benchmarks](docs/inference-and-benchmarks.md)
- [Parsing and rendering](docs/parsing.md)
- [CLI reference](docs/cli.md)
- [Examples](docs/examples.md)
- [Models and features](docs/models.md)
- [Outputs](docs/outputs.md)
- [Data sources](docs/data-sources.md)

## Related Repo

- [MolADT-Bayes-Haskell](https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell)
