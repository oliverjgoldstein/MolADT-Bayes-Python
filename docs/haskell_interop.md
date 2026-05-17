# Haskell Interop

The Python repo produces benchmark exports consumed by the sibling Haskell repo:

https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell

The current Python FreeSolv benchmark is the `moladt_full30_rbf_gp` small-feature GP. The Python repo refreshes the parsed FreeSolv exports with `make freesolv`; Haskell interop still consumes the standardized processed CSV exports.

## Shared Files

Exports live under `data/processed/`.

For `freesolv_moladt_featurized`, Python writes:

- `*_X_train.csv`
- `*_X_valid.csv`
- `*_X_test.csv`
- `*_y_train.csv`
- `*_y_valid.csv`
- `*_y_test.csv`
- `*_metadata.json`
- `*_features.csv`

## Scaling Contract

- `X` is standardized using train-split mean and standard deviation only.
- `y` stays on the original target scale.
- split indices and `mol_id` lists are stored in metadata.
- zero-variance features use safe scale `1.0` and are recorded.

## Workflow

Refresh Python exports:

```bash
./.venv/bin/python -m benchmarking.run_all freesolv
```

Run the Haskell consumer:

```bash
stack run moladtbayes -- --help
```

## Source Files

- [`benchmarking/splits.py`](../benchmarking/splits.py)
- [`benchmarking/process_freesolv.py`](../benchmarking/process_freesolv.py)
- [`benchmarking/run_all.py`](../benchmarking/run_all.py)
