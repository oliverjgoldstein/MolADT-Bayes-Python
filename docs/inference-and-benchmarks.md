# Inference and Benchmarks

The Python repo owns the main empirical benchmark pipeline. It prepares aligned datasets, exports standardized train/valid/test matrices to `data/processed/`, fits Stan models, writes reviewer-facing summaries under `results/`, and optionally runs the ZINC timing comparison with MolADT included.

## Main Entrypoints

```bash
make benchmark
make benchmark-bg
./.venv/bin/python -m scripts.run_all --help
```

`make benchmark-bg` currently runs in the foreground and mirrors live output to `results/benchmark.out` or the active results directory.

## `scripts.run_all` Subcommands

### `smoke-test`

```bash
./.venv/bin/python -m scripts.run_all smoke-test
```

Runs the FreeSolv smoke benchmark. By default it prepares the SMILES export and also the aligned SDF export unless `--skip-sdf` is passed.

### `qm9`

```bash
./.venv/bin/python -m scripts.run_all qm9
```

Runs the QM9 dipole benchmark for target `mu`. The current default deterministic subset size is `2000` rows.

### `zinc-timing`

```bash
./.venv/bin/python -m scripts.run_all zinc-timing
./.venv/bin/python -m scripts.run_all zinc-timing --include-moladt
```

Runs the ZINC SMILES timing benchmark. Current defaults from the code are:

- dataset size: `250K`
- dataset dimension: `2D`
- limit: no explicit row cap unless `--limit` is passed
- stages always included: `raw_file_read`, `smiles_parse_sanitize`, `smiles_canonicalization`
- MolADT stage: `moladt_parse_render` only when `--include-moladt` is passed

The default path does not include MolADT timing unless you request it explicitly.

### `benchmark`

```bash
./.venv/bin/python -m scripts.run_all benchmark
./.venv/bin/python -m scripts.run_all benchmark --include-moladt
```

Runs FreeSolv, QM9, and ZINC in order. Current defaults from the code are:

- methods: `sample,variational,pathfinder,optimize,laplace`
- models: `bayes_linear_student_t,bayes_hierarchical_shrinkage`
- QM9 limit: `2000`
- ZINC dataset size: `250K`
- ZINC dataset dimension: `2D`
- ZINC limit: unset
- MolADT timing: off unless `--include-moladt` is passed

`make benchmark` currently maps to this subcommand without `--include-moladt`. If you want `moladt_parse_render` in `results/zinc_timing.csv`, run:

```bash
./.venv/bin/python -m scripts.run_all benchmark --include-moladt
```

## Important Options

### `--methods`

Comma-separated inference methods to run for each selected model. The current set is:

- `sample`
- `variational`
- `pathfinder`
- `optimize`
- `laplace`

Example:

```bash
./.venv/bin/python -m scripts.run_all smoke-test --methods optimize,pathfinder
```

### `--models`

Comma-separated Stan models to run. Current defaults:

- `bayes_linear_student_t`
- `bayes_hierarchical_shrinkage`

Example:

```bash
./.venv/bin/python -m scripts.run_all qm9 --models bayes_linear_student_t
```

### `--include-moladt`

Relevant to `zinc-timing` and `benchmark`. It adds the `moladt_parse_render` timing stage. Without it, the benchmark stops after RDKit file read, parse/sanitize, and canonicalization stages.

### `--qm9-limit`

Used by `benchmark` to choose the QM9 subset size. The current default is `2000`.

### `--zinc-dataset-size`

Used by `benchmark` and `zinc-timing`. The current default is `250K`.

### `--zinc-dataset-dimension`

Used by `benchmark` and `zinc-timing`. The current default is `2D`.

### `--zinc-limit`

Optional row cap for the ZINC pass. If omitted, the timing run uses the full selected dataset file.

## What `make benchmark` Runs

From the current `Makefile`, the default benchmark helper sets:

- `--qm9-limit 2000`
- `--zinc-dataset-size 250K`
- `--zinc-dataset-dimension 2D`
- `--methods sample,variational,pathfinder,optimize,laplace`
- `--models bayes_linear_student_t,bayes_hierarchical_shrinkage`

It also injects preset-dependent sampling and optimization budgets. The `Makefile` exposes `INFERENCE_PRESET=quick` and `INFERENCE_PRESET=paper` to change those budgets, the expected runtime, and the output location.

## Outputs

The main benchmark outputs go under `results/` by default:

- `results/summary.md`
- `results/predictive_metrics.csv`
- `results/predictions.csv`
- `results/model_report.md`
- `results/model_coefficients.csv`
- `results/zinc_timing.csv`
- `results/zinc_timing.md`

Current reporting also writes:

- `results/generalization_metrics.csv`
- `results/generalization_report.md`
- `results/split_rmse_overview.svg`
- `results/predicted_vs_actual_overview.svg`
- `results/literature_context.md`

If `INFERENCE_PRESET=paper` is used through `make`, outputs go under `results/paper/`. If `MOLADT_RESULTS_DIR` is set, that path becomes the results root.

## Processed Exports for Haskell

The Python side is the producer of the aligned benchmark exports used by the Haskell repo. Those files live under `data/processed/` and include:

- `*_X_train.csv`, `*_X_valid.csv`, `*_X_test.csv`
- `*_y_train.csv`, `*_y_valid.csv`, `*_y_test.csv`
- `*_metadata.json`

The exported predictors are standardized with train-split mean and standard deviation only. The targets stay on their original scale.

For the Haskell consumer view, see [Haskell interop](haskell_interop.md).

## Related Files

- [`scripts/run_all.py`](../scripts/run_all.py)
- [`scripts/benchmark_zinc.py`](../scripts/benchmark_zinc.py)
- [`scripts/process_freesolv.py`](../scripts/process_freesolv.py)
- [`scripts/process_qm9.py`](../scripts/process_qm9.py)
- [`scripts/stan_runner.py`](../scripts/stan_runner.py)
- [`stan/`](../stan/)
