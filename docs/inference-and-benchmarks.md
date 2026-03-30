# Inference and Benchmarks

This repo owns the main benchmark run. It prepares the datasets, exports aligned matrices, fits the Stan models, writes a compact CSV-first result bundle, and runs the timing pass.

## The One Big Command

```bash
make benchmark INFERENCE_PRESET=paper INCLUDE_MOLADT=1
```

This is the full run:

- FreeSolv predictive benchmark
- QM9 dipole benchmark for `mu`
- ZINC timing benchmark
- MolADT timing included
- paper-scale inference budget
- results written under `results/paper/run_<timestamp>/`
- live Stan and timing output shown in the terminal by default

It is the hours-long benchmark command.

Model details: [jump to Model](#model).

If you want a quieter run, use `BENCHMARK_VERBOSE=0 make benchmark ...`.

## Timing

The ZINC timing benchmark measures:

- `raw_file_read`
- `smiles_parse_sanitize`
- `smiles_canonicalization`
- `moladt_parse_render` when `--include-moladt` or `INCLUDE_MOLADT=1` is used

Current defaults from the code:

- dataset size: `250K`
- dataset dimension: `2D`
- QM9 limit: `2000`
- inference methods: `sample,variational,pathfinder,optimize,laplace`
- models: `bayes_linear_student_t,bayes_hierarchical_shrinkage`

## Model

The predictive benchmark fits:

- [`bayes_linear_student_t`](../stan/bayes_linear_student_t.stan)
- [`bayes_hierarchical_shrinkage`](../stan/bayes_hierarchical_shrinkage.stan)

The Python side also exports the aligned matrices used by the Haskell baseline:

- `*_X_train.csv`, `*_X_valid.csv`, `*_X_test.csv`
- `*_y_train.csv`, `*_y_valid.csv`, `*_y_test.csv`
- `*_metadata.json`

`X_train`, `X_valid`, and `X_test` use training-split mean and standard deviation only. `y` stays on the original scale.

## Outputs

Each run writes a timestamped folder. The top level is intentionally small:

- `results.csv`
- `rmse_train_test_vs_literature.svg`
- `timing_overview.svg` when timing ran
- `details/`

`details/` holds the raw CSVs and Stan outputs:

- `details/predictive_metrics.csv`
- `details/generalization_metrics.csv`
- `details/predictions.csv`
- `details/model_coefficients.csv`
- `details/literature_context.csv`
- `details/zinc_timing.csv`
- `details/stan_output/`

For the paper-scale make run, outputs go under `results/paper/run_<timestamp>/`.

## Other Entrypoints

```bash
make benchmark
make benchmark-bg
./.venv/bin/python -m scripts.run_all --help
```

`make benchmark-bg` mirrors live output to the active results log while still running in the foreground.

For the Haskell consumer view, see [Haskell interop](haskell_interop.md).
