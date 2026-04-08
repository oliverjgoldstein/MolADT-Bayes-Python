# MolADT-Bayes-Python

MolADT is a molecular representation that stays explicit where SMILES is compressed.

This repo is the main MolADT benchmark runner. It keeps the representation, parsing, timing, and the focused predictive comparisons for FreeSolv and QM9.

## Start

```bash
make python-setup

make freesolv
make qm9
make timing
```

Those are the front-door commands.

- `make freesolv` runs the focused hydration free energy benchmark.
- `make qm9` runs the focused dipole moment benchmark.
- `make timing` runs the ZINC parsing and file-ingest timing benchmark.

The raw source files those commands use live in this repo under `data/raw/`.

## Representation

MolADT keeps atoms, sigma bonds, Dietz-style bonding systems, coordinates, shells, and orbitals explicit.

In the benchmark, that appears as three MolADT-facing branches:

- `moladt`
- `moladt_typed`, shown as `MolADT+`
- `moladt_typed_geom`, shown as `MolADT+ 3D`

`MolADT+` is the richer descriptor view. It adds typed pair channels, radial distance channels, bond-angle channels, torsion channels, and bonding-system summaries. `MolADT+ 3D` adds the geometry model path on top.

## Why This Exists

SMILES is useful as a boundary format. It is not the right center of this project.

- It hides geometry.
- It flattens richer bonding structure into a renderer-oriented string.
- It is awkward for non-classical chemistry.
- It gives the model a serialization instead of the chemistry object.

MolADT treats the chemistry object as the source of truth and lets SMILES sit at the edge.

## Inference

The front-page commands use the strongest user-facing paths in this repo instead of the old Stan-heavy default.

- FreeSolv uses CatBoost for the fair tabular comparison and DimeNet++ for the geometry-aware branch.
- QM9 `mu` uses CatBoost for the fair tabular comparison and ViSNet for the geometry-aware branch.
- ZINC timing is model-free and measures parse and ingest cost directly.

The older Stan baselines and broader benchmark sweeps still exist, but they are now reference material rather than the landing-page story.

If you want the Stan baselines as well, install CmdStan with:

```bash
make python-cmdstan-install
```

## Outputs

Each run writes a small bundle under `results/`.

- FreeSolv: `results/freesolv/run_<timestamp>/`
- QM9: `results/qm9/run_<timestamp>/`
- Timing: `results/timing/run_<timestamp>/`

The key files are:

- `results.csv`
- `rmse_train_test_vs_literature.svg`
- `inference_sweep_overview.svg`
- `timing_overview.svg` for timing runs
- `figures/metric_comparisons/*.svg`
- `details/`

The comparison charts now separate the fair tabular story from the mixed-family frontier story. The frontier view is where `MolADT+ 3D` appears.

## Read More

- [MolADT representation](docs/representation.md)
- [Parsing and rendering](docs/parsing.md)
- [Models and benchmarks](docs/models.md)
- [Data sources](docs/data-sources.md)
- [Outputs](docs/outputs.md)
- [Quickstart](docs/quickstart.md)
- [Deep benchmark reference](docs/inference-and-benchmarks.md)
- [CLI reference](docs/cli.md)
- [Haskell interop](docs/haskell_interop.md)

## Related Repo

- [MolADT-Bayes-Haskell](https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell)
