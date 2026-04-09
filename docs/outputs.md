# Outputs

Each focused run writes a small timestamped bundle under `results/`.

## Main Locations

- FreeSolv: `results/freesolv/run_<timestamp>/`
- QM9: `results/qm9/run_<timestamp>/`
- combined benchmark: `results/run_<timestamp>/` or `results/paper/run_<timestamp>/`
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

## The Two Comparison Figures

Each predictive run writes only the two reviewer-facing MoleculeNet comparisons:

- `freesolv_rmse_vs_moleculenet.svg`
- `qm9_mae_vs_moleculenet.svg`

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
