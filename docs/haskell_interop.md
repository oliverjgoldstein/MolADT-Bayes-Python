# Haskell Interop

The Python repo is the main producer of aligned benchmark exports for the sibling Haskell repo:

- Haskell repo: https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell

The shared contract lives under `data/processed/`. The goal is to let Haskell consume the exact exported train/valid/test matrices without needing to re-run Python feature extraction or guess at scaling rules.

## What Python Writes

For the Haskell-facing export prefix `freesolv_moladt_featurized`, Python writes:

- `*_X_train.csv`, `*_X_valid.csv`, `*_X_test.csv`
- `*_y_train.csv`, `*_y_valid.csv`, `*_y_test.csv`
- `*_metadata.json`
- `*_features.csv`

Example already used by the Haskell side:

- `data/processed/freesolv_moladt_featurized_X_train.csv`

## Standardization Contract

The alignment rules are simple and explicit:

- `X_train`, `X_valid`, and `X_test` are standardized using the train-split mean and standard deviation only.
- `y_train`, `y_valid`, and `y_test` stay on the original target scale.
- split indices and split `mol_id` lists are written into the metadata JSON.
- the metadata also records `split_scheme`, `source_row_count`, and `used_row_count`.
- zero-variance features are left with a safe scale of `1.0` and recorded in metadata.

This is the contract the Haskell baseline consumes today.

## Metadata Contents

Each `*_metadata.json` file includes:

- dataset name
- representation name
- target name
- seed
- feature names
- feature groups
- group names and group ids
- train means and train standard deviations
- zero-variance feature names
- split indices
- split `mol_id` lists
- the relative path to the full feature CSV

## Relation to the Python Models

Python still fits the export-compatible FreeSolv Stan benchmark path:

- FreeSolv: `bayes_gp_rbf_screened`

The Haskell side consumes the same standardized `X/y` export but runs its own local exact GP implementation rather than the Python Stan path.

Shared assumptions that matter for interop:

- the response stays on its original scale
- the predictor matrices are standardized only from the training split

## Practical Workflow

1. Produce or refresh the Python-side exports:

   ```bash
   ./.venv/bin/python -m scripts.run_all freesolv
   ```

2. In the Haskell repo, point the consumer at the processed directory:

   ```bash
   MOLADT_PROCESSED_DATA_DIR=../MolADT-Bayes-Python/data/processed stack run moladtbayes -- infer-benchmark freesolv_moladt_featurized lwis
   ```

## Source Files

- [`scripts/splits.py`](../scripts/splits.py)
- [`scripts/process_freesolv.py`](../scripts/process_freesolv.py)
- [`scripts/run_all.py`](../scripts/run_all.py)
