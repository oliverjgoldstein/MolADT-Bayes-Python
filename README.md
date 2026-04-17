# MolADT-Bayes-Python

MolADT is a typed molecular data format for Bayesian work over molecules. It keeps chemically meaningful structure in the object itself instead of hiding it inside string syntax.

[Quickstart](docs/quickstart.md) · [Representation](docs/representation.md) · [Models](docs/models.md) · [Examples](docs/examples.md)

## Why MolADT

Diborane, ferrocene, and morphine are useful examples.

- diborane in standard SMILES: `[BH2]1[H][BH2][H]1`
- ferrocene in standard SMILES: `[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]`
- morphine in standard stereochemical SMILES: `CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5`

Those are useful interchange formats, but they are awkward as the main in-memory representation.

- diborane wants explicit `3c-2e` bridges
- ferrocene wants shared Cp/metal bonding systems
- morphine pushes fused-ring structure and stereo flags into a linear notation

MolADT keeps that chemistry as typed atoms, local bonds, bonding systems, and stereo annotations. The code in this repo works over that object directly instead of treating a string format as the primary model.

## Quick Start

```bash
make python-setup
make python-parse
# once, before Stan-backed targets such as FreeSolv
make python-cmdstan-install
```

`make python-setup` installs the Python package and its benchmark dependencies only into `./.venv` inside this repo. `make python-cmdstan-install` installs CmdStan only into `./.cmdstan`. Deleting `./.venv` removes the Python-side local install, deleting `./.cmdstan` removes the local CmdStan toolchain, and neither command touches your system Python, global site-packages, or other local environments.

If your shell creates a Windows-style virtual environment, the Make targets will use `.venv/Scripts/python.exe` automatically. On a fresh machine, the first local setup can take a few minutes and up to about 30 minutes if the larger dependencies still need to be downloaded or built.

`Molecule` is immutable and destructures as:
`atoms, local_bonds, systems, smiles_stereochemistry = molecule`

For probabilistic proposals or local graph surgery, use `MutableMolecule` from `moladt.chem.mutable` as a writable scratch state and call `freeze()` to return to canonical `Molecule`.

## Parsing

Use the CLI when you want to inspect how a boundary format lands inside MolADT.

```bash
make python-parse
make python-pretty-example EXAMPLE=morphine
make python-to-smiles
```

- `parse` reads one SDF record, validates it, prints the MolADT structure, and preserves SDF title and property fields
- `pretty-example` loads the manuscript-facing built-in objects, written as explicit typed molecules with orbital shells intact
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
- feature generation, Stan models, and local benchmark tooling

## Benchmarking

Benchmark outputs stay local. `results/` is generated on your machine rather than committed.

```bash
make freesolv
make qm9long
make timing
```

- `make freesolv`: FreeSolv RMSE comparison. Fixed path `moladt_featurized + bayes_gp_rbf_screened + laplace`. The FreeSolv SVG includes posterior predictive RMSE uncertainty bars from the Stan fit. It writes `results/freesolv/run_.../freesolv_rmse_vs_moleculenet.svg`, `results/freesolv/run_.../freesolv_bayesian_model.txt`, and `results/freesolv/run_.../details/freesolv_train_test_uncertainty.csv`.
- `make qm9long`: full QM9 `mu` MAE comparison over all aligned local QM9 molecules, using `visnet_ensemble` on `moladt_featurized_geom`. That export keeps the atomic numbers and coordinates from the SDF record and adds the full MolADT feature bundle from the same molecule. The current local bundle yields `107,108 / 13,388 / 13,389` train / validation / test rows under the deterministic `80/10/10` long split. ViSNet runs one member for at most `25` epochs with seed `102`, and the verbose run prints every epoch with validation RMSE and MAE. Writes `results/qm9/long/run_.../qm9_mae_vs_moleculenet.svg`.
- `make timing`: ZINC SDF ingest and runtime comparison. It measures raw SDF block reads, local SDF parsing into MolADT, the one-time matched corpus build, and MolADT file loading. Writes `results/timing/paper/run_.../timing_overview.svg`.

Results are written under timestamped directories in `results/`, mainly `results/freesolv/run_.../`, `results/qm9/long/run_.../`, and `results/timing/paper/run_.../`.

The FreeSolv figure shows `Training`, `Validation`, `Test`, and `Paper`. The QM9 figure shows `Training`, `Test`, and `Paper`.

For split sizes, the exact benchmark contract, the published FreeSolv RMSE context table, and the detailed timing-stage definitions, see [Inference and benchmarks](docs/inference-and-benchmarks.md) and [Outputs](docs/outputs.md).

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
