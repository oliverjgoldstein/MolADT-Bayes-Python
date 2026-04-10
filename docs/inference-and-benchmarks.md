# Inference and Benchmarks

This repo owns the main benchmark run. It prepares datasets, exports aligned MolADT matrices, fits the Stan baselines, and writes the reviewer-facing comparison bundle.

## Main Commands

```bash
make python-cmdstan-install
make freesolv
make qm9
make benchmark-small
make timing
./.venv/bin/python -m scripts.run_all smoke-test --verbose
./.venv/bin/python -m scripts.run_all qm9 --paper-mode --verbose
./.venv/bin/python -m scripts.run_all benchmark --paper-mode --verbose
```

- `make freesolv` runs the long FreeSolv MolADT sweep and compares the validation-selected local Stan run against MoleculeNet Table 3 on RMSE
- `make qm9` runs the long QM9 `mu` MolADT sweep on the full local download with the paper-sized split and compares the validation-selected local Stan run against MoleculeNet Table 3 on MAE
- `make benchmark-small` keeps the older lighter `QM9_LIMIT=2000` subset path available for a faster local check
- `make timing` runs the separate ZINC timing/interoperability pass
- `make python-cmdstan-install` is the one-time local CmdStan install step required before the Stan benchmark targets

## Benchmark Contract

The benchmark contract is deliberately narrow:

- `moladt`
  The same boundary SMILES is parsed into the typed MolADT object and featurized from that object.
- the local run is selected by validation RMSE or MAE, not by test-set peeking
- each reviewer figure shows `Training`, `Test`, then `Paper`
- the paper bar is the matching MoleculeNet row only
- FreeSolv uses RMSE
- QM9 `mu` uses MAE

## What Stan Fits

Stan only consumes numeric `X/y` matrices here, so the benchmark is over MolADT-derived descriptor matrices rather than literal string objects.

The aligned Stan models are:

- [`bayes_linear_student_t`](../stan/bayes_linear_student_t.stan)
- [`bayes_hierarchical_shrinkage`](../stan/bayes_hierarchical_shrinkage.stan)

The benchmark graph keeps the training and held-out test metrics from the validation-selected local MolADT row from that Stan sweep.

## Dataset Meaning

- FreeSolv compares the best local MolADT RMSE against the MoleculeNet Table 3 MPNN RMSE row `1.15`.
- QM9 `mu` compares the best local MolADT MAE against the MoleculeNet Table 3 DTNN MAE row `2.35`.

By default the top-level `make freesolv`, `make qm9`, and `make benchmark` targets use the long `paper` inference preset. `make qm9` is therefore expected to take hours after CmdStan is already built.

## Timing

`make timing` is not the same question as the predictive benchmark.

It writes a ZINC timing bundle under `results/timing/run_<timestamp>/` and reports ingest/runtime stages, including raw file IO, manifest CSV field materialization as plain strings, the local SMILES parser path, and the optional local MolADT-library path. Treat it as an interoperability/runtime benchmark, not as the central representation-quality comparison.

## Outputs

Each predictive run writes a small timestamped folder with:

- `results.csv`
- `freesolv_rmse_vs_moleculenet.svg`
- `qm9_mae_vs_moleculenet.svg`
- `details/`

Important detail files include:

- `details/predictive_metrics.csv`
- `details/aggregated_predictive_metrics.csv`
- `details/predictions.csv`
- `details/model_coefficients.csv`
- `details/moleculenet_comparison.csv`

## Export Contract

The Python side also exports the aligned matrices consumed by the Haskell baseline:

- `*_X_train.csv`, `*_X_valid.csv`, `*_X_test.csv`
- `*_y_train.csv`, `*_y_valid.csv`, `*_y_test.csv`
- `*_metadata.json`

`X_train`, `X_valid`, and `X_test` are standardized from the training split only. `y` stays on the original target scale.

For the Haskell consumer view, see [Haskell interop](haskell_interop.md).
