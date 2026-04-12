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

If one SDF file contains multiple molecules, use `from moladt.io.sdf import read_sdf_records; titles = [record.title for record in read_sdf_records("bundle.sdf", limit=3)]` or stream lazily with `from moladt.io.sdf import iter_sdf_records; first_titles = [record.title for record in iter_sdf_records("bundle.sdf", limit=3)]`.

The local QM9 and vendored FreeSolv raw files in this workspace are still V2000. The downloader and parser prefer V3000 when a future dataset bundle actually provides it, but the current local raws are not being silently relabeled.

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
- `make freesolv` runs one fixed benchmark path: `moladt_featurized` with the `bayes_gp_rbf_screened` model fit by Stan `laplace`
- `make qm9` runs one fixed benchmark path: `moladt_featurized` with the `bayes_linear_student_t` model fit by Stan `optimize`
- FreeSolv compares local MolADT RMSE to the MoleculeNet MPNN RMSE row `1.15`
- QM9 `mu` compares local MolADT MAE to the MoleculeNet DTNN MAE row `2.35`

The QM9 path is explicit: representation `moladt_featurized`, model `bayes_linear_student_t`, inference algorithm Stan `optimize` with L-BFGS mode fitting. That choice is deliberate. `mu` is a 3D directional property, so the default path uses the richer MolADT featurized branch with radial, angle, and torsion channels rather than the older compact descriptor set. Exact GP inference is not practical at full QM9 scale, and the old NUTS run is pathological on this benchmark. On the local `QM9_LIMIT=2000` subset sweep, the featurized Student-`t` path with `optimize` beat the other scalable Stan fits on validation MAE, so that is the fixed Stan benchmark path.

The default benchmark targets use the long `paper` inference preset. `make qm9` still runs the full paper-sized split, but it no longer launches the old multi-model multi-algorithm Stan sweep by default. Use `make benchmark-small` or override `INFERENCE_PRESET=default QM9_LIMIT=2000 QM9_SPLIT_MODE=subset` for a lighter run.

The metric matches the cited MoleculeNet row, but the Stan model family still differs from the paper, and FreeSolv still uses the repo's own split. Large raw files are downloaded on demand when they are not already vendored.

Read the timing overview as a pipeline rather than as one score. The SVG uses a log throughput axis, so equal horizontal gaps mean multiplicative speedups.

- `raw_file_read`: I/O baseline only. It reads SMILES text from the normalized ZINC source file and does no chemistry work.
- `smiles_parse_sanitize`: external-toolkit stage. It parses each SMILES string into a molecular graph and applies that toolkit's sanitization rules.
- `smiles_canonicalization`: external-toolkit stage. It rewrites the parsed molecule into one canonical SMILES form.
- `timing_library_prepare`: one-time setup. It builds the matched timing corpus so every later stage runs on the same molecules.
- `smiles_csv_string_parse`: plain-string baseline. It only materializes the canonical SMILES CSV field into a Python string.
- `smiles_library_parse`: our parser. It turns that string into the typed MolADT object through the local SMILES path.
- `moladt_file_parse`: our file reader. It loads the already-structured MolADT JSON form from disk and reconstructs the local typed object.

In practice, compare `smiles_csv_string_parse` against `smiles_library_parse` to see the real incremental cost of building a MolADT object from SMILES text. Compare `smiles_library_parse` against `moladt_file_parse` to see the gap between parsing a boundary string and reading the structured MolADT form directly.

## Read More

- [Quickstart](docs/quickstart.md)
- [Parsing and rendering](docs/parsing.md)
- [CLI reference](docs/cli.md)
- [Examples](docs/examples.md)
- [Models and features](docs/models.md)
- [Data sources](docs/data-sources.md)

## Related Repo

- [MolADT-Bayes-Haskell](https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell)
