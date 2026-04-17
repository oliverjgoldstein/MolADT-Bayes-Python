# Outputs

Each focused run writes a small timestamped bundle under `results/`.

## Main Locations

- FreeSolv: `results/freesolv/run_<timestamp>/`
- QM9: `results/qm9/long/run_<timestamp>/` for the full-data `qm9long` run
- combined benchmark: `results/paper/run_<timestamp>/` for the default long run, or `results/run_<timestamp>/` for explicit lighter overrides
- ZINC timing: `results/timing/paper/run_<timestamp>/` by default, or `results/timing/run_<timestamp>/` when you override `INFERENCE_PRESET`

## What You See First

Each predictive run keeps the top level small:

- `results.csv`
- `freesolv_rmse_vs_moleculenet.svg`
- `freesolv_bayesian_model.txt`
- `qm9_mae_vs_moleculenet.svg`
- `details/`

Timing runs also write:

- `timing_overview.svg`
- `caption.txt`
- `timing_result_files.txt`
- `details/zinc_timing.csv`
- `details/zinc_timing_items.csv`
- `details/zinc_timing_corpus_manifest.csv`

## The Two Comparison Figures

Each predictive run writes only the two MoleculeNet comparison figures:

- `freesolv_rmse_vs_moleculenet.svg`
- `qm9_mae_vs_moleculenet.svg`
- `caption.txt` in the run directory for the paper-facing prose caption

The FreeSolv figure shows local `Training`, `Validation`, and `Test` metrics, followed by the cited `Paper` baseline.
The QM9 figure shows local `Training` and `Test` metrics, followed by the cited `Paper` baseline for the `qm9long` ViSNet run on `moladt_featurized_geom`.

## Other Useful Files

- `details/predictive_metrics.csv`
- `details/aggregated_predictive_metrics.csv`
- `details/predictions.csv`
- `details/freesolv_train_test_uncertainty.csv`
- `details/moleculenet_comparison.csv`

## How To Read The Bundle

Use the files in this order:

1. `results.csv` for the summary row view
2. `freesolv_rmse_vs_moleculenet.svg` and `qm9_mae_vs_moleculenet.svg` for the paper comparison
3. `details/` when you need raw rows for analysis or plotting

For timing runs specifically, `details/zinc_timing.csv` separates:

- the matched SMILES CSV read (`smiles_csv_to_string`)
- SMILES parsing plus JSON serialization (`smiles_to_json`)
- cached SDF parsing into MolADT (`sdf_to_moladt`)
- cached SDF rendering back to SMILES (`sdf_to_smiles`)
- MolADT JSON serialization (`moladt_to_json`)
- JSON decoding back into MolADT (`json_to_moladt`)

The top-level `timing_overview.svg` is the clean paper figure. The explanatory prose lives in `caption.txt`.
