# Haskell Interop

The Python repo produces benchmark exports consumed by the sibling Haskell repo:

https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell

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
./.venv/bin/python -m scripts.run_all freesolv
```

Run the Haskell consumer:

```bash
MOLADT_PROCESSED_DATA_DIR=../MolADT-Bayes-Python/data/processed \
  stack run moladtbayes -- infer-benchmark freesolv_moladt_featurized lwis
```

## Source Files

- [`scripts/splits.py`](../scripts/splits.py)
- [`scripts/process_freesolv.py`](../scripts/process_freesolv.py)
- [`scripts/run_all.py`](../scripts/run_all.py)
