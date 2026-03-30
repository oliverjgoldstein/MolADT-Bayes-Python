# Inference and Benchmarks

This repo owns the main benchmark run. It prepares the datasets, exports aligned matrices, fits the Stan models, writes the reports, and runs the timing pass.

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
- results written under `results/paper/`
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

The main outputs are:

- `results/summary.md`
- `results/predictive_metrics.csv`
- `results/predictions.csv`
- `results/model_report.md`
- `results/model_coefficients.csv`
- `results/zinc_timing.csv`
- `results/zinc_timing.md`
- `results/generalization_metrics.csv`
- `results/generalization_report.md`
- `results/split_rmse_overview.svg`
- `results/predicted_vs_actual_overview.svg`
- `results/literature_context.md`

For the paper-scale make run, outputs go under `results/paper/`.

## Other Entrypoints

```bash
make benchmark
make benchmark-bg
./.venv/bin/python -m scripts.run_all --help
```

`make benchmark-bg` mirrors live output to the active results log while still running in the foreground.

For the Haskell consumer view, see [Haskell interop](haskell_interop.md).
