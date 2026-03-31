# MolADT-Bayes-Python

`MolADT-Bayes-Python` is the main benchmark repo for MolADT. It brings together the molecule representation, explicit orbital data, timing benchmark, Stan models, and the aligned train/valid/test exports used by the Haskell baseline.

MolADT is intended as a replacement molecular representation for bringing cheminformatics into the present day: explicit about orbitals and bonding, and less constrained by legacy graph-only assumptions.

Setup expects Python 3.11+ and a POSIX shell. On Windows, the documented path is WSL2.

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

Optional extra model families are available behind extras:

- `pip install -e .[ml]` for `catboost_uncertainty`
- `pip install -e .[geom]` for `visnet_ensemble` and `dimenetpp_ensemble`

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
`make model` runs the predictive suite only and writes the run under `results/models/run_<timestamp>/` or `results/models/paper/run_<timestamp>/`.
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
