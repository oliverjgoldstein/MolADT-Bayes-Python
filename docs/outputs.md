# Outputs

Each focused run writes a small timestamped bundle under `results/`.

## Main Locations

- FreeSolv: `results/freesolv/run_<timestamp>/`
- QM9: `results/qm9/run_<timestamp>/`
- ZINC timing: `results/timing/run_<timestamp>/`

The older broad benchmark targets still write to the existing `results/run_<timestamp>/` and `results/paper/run_<timestamp>/` layout.

## What You See First

Each predictive run keeps the top level small:

- `results.csv`
- `rmse_train_test_vs_literature.svg`
- `inference_sweep_overview.svg`
- `figures/`
- `details/`

Timing runs also write:

- `timing_overview.svg`
- `details/zinc_timing.csv`

## The Two Comparison Views

Inside `figures/metric_comparisons/` the repo now writes two kinds of chart per metric.

- `*_comparison.svg`
  The fair tabular view.
  This is the fixed-learner comparison for `smiles`, `moladt`, and `moladt_typed`.
- `*_frontier_comparison.svg`
  The mixed-family frontier.
  This adds `moladt_typed_geom` as `MolADT+ 3D` and makes it explicit that the geometry row may come from a different model family.

## Other Useful Files

- `figures/predicted_vs_actual_scatter.svg`
- `figures/residual_vs_uncertainty.svg`
- `figures/coverage_calibration.svg`
- `details/predictive_metrics.csv`
- `details/aggregated_predictive_metrics.csv`
- `details/predictions.csv`
- `details/model_artifacts.csv`

## How To Read The Bundle

Use the files in this order:

1. `results.csv` for the summary row view
2. `figures/metric_comparisons/` for representation-level comparisons
3. `figures/predicted_vs_actual_scatter.svg` for molecule-level sanity checking
4. `details/` when you need raw rows for analysis or plotting
