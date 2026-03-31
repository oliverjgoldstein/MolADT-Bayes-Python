# Inference and Benchmarks

This repo owns the main benchmark run. It prepares the datasets, exports aligned matrices, fits the Stan models, writes a compact CSV-first result bundle, and runs the timing pass.

## Main Commands

```bash
make timing
make catboost-geom-model
make catboost-geom-model-paper
make benchmark-small
make benchmark-paper
./.venv/bin/python -m scripts.run_all benchmark --include-moladt-predictive --extra-models catboost_uncertainty
./.venv/bin/python -m scripts.run_all benchmark --include-moladt-predictive --extra-models catboost_uncertainty --geom-model visnet
./.venv/bin/python -m scripts.run_all benchmark --include-moladt-predictive --extra-models catboost_uncertainty,visnet_ensemble --paper-mode
```

`make timing` is the local timing-only command:

- ZINC timing benchmark only
- builds a matched local corpus under `data/processed/zinc_timing/...`
- writes one MolADT JSON file per matched molecule plus one canonical SMILES corpus file
- measures RDKit baseline parsing plus local MolADT SMILES parsing and local MolADT file parsing
- writes results under `results/timing/run_<timestamp>/`

`make benchmark-small` is the default benchmark:

- FreeSolv predictive benchmark
- QM9 `mu` benchmark on the deterministic 2000-row subset configuration
- ZINC timing benchmark
- MolADT timing included
- results written under `results/run_<timestamp>/`

`make benchmark-paper` is the full long run:

- FreeSolv predictive benchmark
- QM9 dipole benchmark for `mu`
- ZINC timing benchmark
- MolADT timing included
- paper-scale inference budget
- deterministic local split counts matching the paper-sized QM9 setup: `110462 / 10000 / 10000`
- results written under `results/paper/run_<timestamp>/`
- live Stan and timing output shown in the terminal by default

That paper-style QM9 split uses `130,462` molecules in total: `110,462` train, `10,000` validation, and `10,000` test. It is not `100k` per split.

It is the hours-long benchmark command. If you want a quieter run, use `BENCHMARK_VERBOSE=0 make benchmark-paper`.

Model details: [jump to Model](#model).

## Timing

The ZINC timing benchmark measures:

- `raw_file_read`
- `smiles_parse_sanitize`
- `smiles_canonicalization`
- `timing_library_prepare` when MolADT timing is enabled
  This builds the matched local timing corpus: one MolADT JSON file per molecule and one canonical SMILES library with the same molecule count.
- `smiles_library_parse` when MolADT timing is enabled
  This parses each matched canonical SMILES entry with the local MolADT SMILES parser.
- `moladt_file_parse` when MolADT timing is enabled
  This reads each MolADT JSON file and parses it back into the local Molecule ADT.

Current defaults from the code:

- dataset size: `250K`
- dataset dimension: `2D`
- QM9 small-run mode: `QM9_LIMIT=2000`, `QM9_SPLIT_MODE=subset`
- QM9 paper-run mode: `QM9_LIMIT=` with `QM9_SPLIT_MODE=paper`
- inference methods: `sample,variational,pathfinder,optimize,laplace`
- models: `bayes_linear_student_t,bayes_hierarchical_shrinkage`

## Model

The predictive benchmark fits:

- [`bayes_linear_student_t`](../stan/bayes_linear_student_t.stan)
- [`bayes_hierarchical_shrinkage`](../stan/bayes_hierarchical_shrinkage.stan)
- optional `catboost_uncertainty` for fair tabular representation comparisons
- optional `visnet_ensemble` or `dimenetpp_ensemble` for geometry-aware rows

Reported representations are:

- `smiles`
- `moladt`
  The `moladt` branch is not a raw SDF descriptor path. Structure-backed molecules are parsed into the MolADT object first, then ADT-native descriptors are computed from that object.
- `sdf_geom` and `moladt_geom` behind the optional geometry extras

The MolADT file parse stage uses `orjson` when it is present in the local environment because this is runtime data parsing, not source-code parsing.

The Python side also exports the aligned matrices used by the Haskell baseline:

- `*_X_train.csv`, `*_X_valid.csv`, `*_X_test.csv`
- `*_y_train.csv`, `*_y_valid.csv`, `*_y_test.csv`
- `*_metadata.json`

`X_train`, `X_valid`, and `X_test` use training-split mean and standard deviation only. `y` stays on the original scale.

## Outputs

Each run writes a timestamped folder. The top level is intentionally small:

- `results.csv`
- `rmse_train_test_vs_literature.svg`
- `inference_sweep_overview.svg`
- `timing_overview.svg` when timing ran
- `details/`

`details/` holds the raw CSVs and Stan outputs:

- `details/predictive_metrics.csv`
- `details/aggregated_predictive_metrics.csv`
- `details/generalization_metrics.csv`
- `details/predictions.csv`
- `details/model_coefficients.csv`
- `details/training_curves.csv`
- `details/model_artifacts.csv`
- `details/zinc_timing_items.csv`
- `details/zinc_timing_library_manifest.csv`
- `literature_baselines.csv`
- `literature_comparison.md`
- `calibration.csv`
- `models/README.md`
- `models/<model_name>/README.md`
- `figures/predicted_vs_actual_scatter.svg`
- `figures/residual_vs_uncertainty.svg`
- `figures/coverage_calibration.svg`
- `figures/metric_comparisons/*.svg`
- `details/zinc_timing.csv`
- `details/stan_output/`

For the paper-scale make run, outputs go under `results/paper/run_<timestamp>/`.

The metric comparison pack writes one SVG per metric. Each chart compares the matched local `smiles` and `moladt` rows from a shared model family when available, plus a literature bar if the repo has a numeric paper-context value for that metric.

## Other Entrypoints

```bash
make timing
make benchmark-small
make benchmark-paper
make benchmark-bg
./.venv/bin/python -m scripts.run_all --help
```

`make benchmark-bg` mirrors live output to the active results log while still running in the foreground.
`make catboost-geom-model` runs the predictive model suite only on the default QM9 subset and writes a per-model browser under `results/models/...`.
`make catboost-geom-model-paper` runs the predictive model suite only on the paper-sized QM9 split and writes under `results/models/paper/...`.

For the Haskell consumer view, see [Haskell interop](haskell_interop.md).
