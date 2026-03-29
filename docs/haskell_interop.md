# Haskell Interop Note

The exported matrices under `data/processed/` are intended to be simple enough to load from Haskell without any Python-specific runtime assumptions.

## Model

- Response: the benchmark keeps the target on its original scale.
- Predictors: every column in `X_train/X_valid/X_test` is standardized with train-set mean and train-set standard deviation only.
- Likelihood: both Stan models use a Student-`t` likelihood with fixed `nu = 4`.
- Linear model priors:
  - intercept `alpha ~ Normal(0, 1.5)`
  - coefficients `beta_k ~ Normal(0, 1)`
  - `sigma ~ HalfNormal(1)` via `normal(0, 1)` with `<lower=0>`
- Hierarchical shrinkage priors:
  - `beta_raw_k ~ Normal(0, 1)`
  - `global_scale ~ HalfNormal(0.5)`
  - one `group_scale_g ~ HalfNormal(1)` per feature family
  - `beta_k = beta_raw_k * global_scale * group_scale[group_id[k]]`

## Splits

- Splits are deterministic random permutations generated from the metadata `seed`.
- Default fractions are `0.8 / 0.1 / 0.1` for train/valid/test.
- The exact row indices and `mol_id` lists for each split are stored in `<dataset>_<repr>_metadata.json`.

## Files

- `*_X_train.csv`, `*_X_valid.csv`, `*_X_test.csv`: standardized predictor matrices with headers.
- `*_y_train.csv`, `*_y_valid.csv`, `*_y_test.csv`: target vectors on the original scale.
- `*_metadata.json`: feature names, feature groups, train means/stds, split indices, split `mol_id`s, and seed.

These files are sufficient for a Haskell Metropolis-Hastings baseline to reuse the exact predictors, target values, standardization constants, and splits used by the Python CmdStanPy runs.
