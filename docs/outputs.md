# Outputs

Each focused run writes a small timestamped bundle under `results/`.

## Main Locations

- FreeSolv: `results/freesolv/run_<timestamp>/`
- QM9: `results/qm9/paper/run_<timestamp>/` for the default long run, or `results/qm9/run_<timestamp>/` for explicit lighter overrides
- combined benchmark: `results/paper/run_<timestamp>/` for the default long run, or `results/run_<timestamp>/` for explicit lighter overrides
- ZINC timing: `results/timing/run_<timestamp>/`

## What You See First

Each predictive run keeps the top level small:

- `results.csv`
- `freesolv_rmse_vs_moleculenet.svg`
- `qm9_mae_vs_moleculenet.svg`
- `details/`

Timing runs also write:

- `timing_overview.svg`
- `details/zinc_timing.csv`
- `details/zinc_timing_items.csv`
- `details/zinc_timing_library_manifest.csv`

## The Two Comparison Figures

Each predictive run writes only the two MoleculeNet comparison figures:

- `freesolv_rmse_vs_moleculenet.svg`
- `qm9_mae_vs_moleculenet.svg`

The FreeSolv figure shows local `Training`, `Validation`, and `Test` metrics, followed by the cited `Paper` baseline.
The QM9 figure shows local `Training` and `Test` metrics, followed by the cited `Paper` baseline.

## Other Useful Files

- `details/predictive_metrics.csv`
- `details/aggregated_predictive_metrics.csv`
- `details/predictions.csv`
- `details/moleculenet_comparison.csv`

## How To Read The Bundle

Use the files in this order:

1. `results.csv` for the summary row view
2. `freesolv_rmse_vs_moleculenet.svg` and `qm9_mae_vs_moleculenet.svg` for the paper comparison
3. `details/` when you need raw rows for analysis or plotting

For timing runs specifically, `details/zinc_timing.csv` separates the manifest CSV field-to-string baseline (`smiles_csv_string_parse`) from the local SMILES parser and MolADT file parser.
