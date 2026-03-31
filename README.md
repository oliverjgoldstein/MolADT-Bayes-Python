# MolADT-Bayes-Python

`MolADT-Bayes-Python` is the main benchmark repo for MolADT. It brings together the molecule representation, explicit orbital data, timing benchmark, Stan models, and the aligned train/valid/test exports used by the Haskell baseline.

MolADT is intended as a replacement molecular representation for bringing cheminformatics into the present day: explicit about orbitals and bonding, and less constrained by legacy graph-only assumptions.

Setup expects Python 3.11+ and a POSIX shell. On Windows, the documented path is WSL2.

The intended workflow is local to the directory you are in. The default setup creates a repo-local `.venv`, installs CmdStan into repo-local `.cmdstan`, downloads datasets under `data/`, and writes benchmark outputs under `results/`. In normal use, you do not need a global Python environment for this project beyond having a working Python with `venv` support.

## Molecule Representation

MolADT represents a molecule as atoms, localized sigma bonds, and Dietz-style bonding systems. Classical molecules fit cleanly. Non-classical examples such as diborane and ferrocene still fit the core representation even when they are outside the current SMILES renderer.

## Orbitals

Atoms keep explicit shell and orbital information. The pretty-printer exposes that directly, so the representation is not just a graph with labels.

## Timing

The ZINC benchmark measures:

- raw file read
- SMILES parse and sanitize
- SMILES canonicalization
- optional MolADT parse and render from an RDKit MolBlock through the ADT

## Model

The predictive benchmark fits two Stan models:

- [`bayes_linear_student_t`](stan/bayes_linear_student_t.stan)
- [`bayes_hierarchical_shrinkage`](stan/bayes_hierarchical_shrinkage.stan)

The Python repo also writes aligned `data/processed/` exports. `X_train`, `X_valid`, and `X_test` are standardized from the training split only. `y` stays on the original scale.
Reported representations are `smiles` and `moladt`: the `moladt` branch is built by parsing structure-backed molecules into the ADT and then extracting ADT-native descriptor features from that object.

By default, `make python-setup` installs the CatBoost tabular stack and the PyTorch Geometric geometry stack needed for `make model`, and it does so inside the local repo environment.

Scientific framing:

- `catboost_uncertainty` answers the fair tabular question: same learner, same split, different representation (`smiles` vs `moladt`, and any other tabular export).
- `visnet_ensemble` or `dimenetpp_ensemble` answers the geometry question: what changes when the model family can use coordinates and optional MolADT global descriptors.

## Run Everything

```bash
make python-setup
make python-cmdstan-install
make model
make benchmark-small
make benchmark-paper
```

`make benchmark-small` runs the default 2000-row QM9 subset with MolADT timing enabled.
`make benchmark-paper` runs the paper-sized QM9 split counts `110462 / 10000 / 10000`, uses the paper inference preset, includes MolADT timing, and writes the long-run artifacts under `results/paper/run_<timestamp>/`.
That QM9 paper-style split uses `130,462` molecules in total: `110,462` train, `10,000` validation, and `10,000` test. It is not `100k` in each split.
`make model` runs the predictive model suite only, without the ZINC timing pass. It prepares FreeSolv and QM9, runs the configured predictive models, writes the run under `results/models/run_<timestamp>/` or `results/models/paper/run_<timestamp>/`, and then creates per-model folders under `models/` so you can open one model at a time and see its filtered metrics, predictions, and a short explanation of how to read that model against `smiles` and `MolADT`.
All of those paths are local to this repo directory: `.venv/`, `.cmdstan/`, `data/`, and `results/`.
By default, `make model` runs the two Stan baselines plus the default extra-model set from the `models` subcommand: `catboost_uncertainty` and `visnet_ensemble`. So the default suite is:

- Stan: `bayes_linear_student_t`
- Stan: `bayes_hierarchical_shrinkage`
- Tabular extra model: `catboost_uncertainty`
- Geometry extra model: `visnet_ensemble`

What each part is doing:

- The Stan models fit Bayesian linear baselines on standardized tabular features for `smiles` and `moladt`.
- CatBoost fits the same `smiles` and `moladt` tabular exports under one shared non-linear learner, so this is the main fair representation comparison.
- ViSNet fits the coordinate-aware geometric exports such as `sdf_geom` and `moladt_geom`, so it answers a different question: what changes when the model can use 3D coordinates.

Rough runtime guidance:

- `make model` in the default preset is usually dominated by the Stan fits and is typically in the same general range as the normal benchmark without the ZINC pass: often tens of minutes rather than seconds on a laptop once CmdStan is built.
- `catboost_uncertainty` adds a validation search plus one fitted model per seed for each tabular representation, so it is usually materially slower than the Stan smoke path but still much cheaper than the full paper run.
- `visnet_ensemble` is usually the slowest optional branch because it trains a neural geometry model; in paper mode it can push the run into multi-hour territory.
- `make benchmark-paper` remains the longest path because it combines the paper-sized QM9 split with the long inference preset and the timing benchmark.

If you want to trim the suite, pass explicit extra-model flags such as `--extra-models catboost_uncertainty` or `--skip-geom`.
Each run now lands in its own timestamped folder with a top-level `results.csv`, `rmse_train_test_vs_literature.svg`, `inference_sweep_overview.svg`, `timing_overview.svg`, and a `details/` subfolder for the raw CSVs and Stan output.

In brief, the reported results come from three model layers:

- Stan: `bayes_linear_student_t` and `bayes_hierarchical_shrinkage` are the descriptor-based Bayesian baselines run on standardized `smiles` and `moladt` feature tables.
- CatBoost: `catboost_uncertainty` is the fair non-linear tabular comparison, so it is the main `smiles` vs `MolADT` representation test under one shared learner.
- Geometry models: `visnet_ensemble` or `dimenetpp_ensemble` are only for coordinate-aware rows such as `sdf_geom` and `moladt_geom`, so they answer a different question from the tabular comparison.

Optional comparison commands:

```bash
./.venv/bin/python -m scripts.run_all benchmark --include-moladt-predictive --extra-models catboost_uncertainty
./.venv/bin/python -m scripts.run_all benchmark --include-moladt-predictive --extra-models catboost_uncertainty --geom-model visnet
./.venv/bin/python -m scripts.run_all benchmark --include-moladt-predictive --extra-models catboost_uncertainty,visnet_ensemble --paper-mode
```

These runs extend the default Stan baseline instead of replacing it. Extra outputs include:

- `calibration.csv`
- `literature_baselines.csv`
- `literature_comparison.md`
- `figures/predicted_vs_actual_scatter.svg`
- `figures/residual_vs_uncertainty.svg`
- `figures/coverage_calibration.svg`
- `model_artifacts/`
- `models/<model_name>/README.md`
- `models/<model_name>/predictive_metrics.csv`
- `models/<model_name>/predictions.csv`

Model details: [jump to Model](#model).

## Docs

- [Docs index](docs/README.md)
- [Inference and benchmarks](docs/inference-and-benchmarks.md)
- [Examples](docs/examples.md)
- [CLI](docs/cli.md)
- [SMILES scope and validation](docs/smiles-scope-and-validation.md)
- [Haskell interop](docs/haskell_interop.md)

## Sibling Repo

- [MolADT-Bayes-Haskell](https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell)

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
