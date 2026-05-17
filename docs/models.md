# Benchmarking Models

Benchmarking models consume small numeric feature tables. The current FreeSolv benchmark keeps the SMILES graph baseline separate from the MolADT-only descriptor additions.

## Main Commands

```bash
make freesolv
make freesolv-20split
make freesolv-ablation
make qm9long
make timing
```

## FreeSolv

Current path:

- representation: `moladt_small_descriptors`
- model: `moladt_full30_rbf_gp`
- method: empirical-Bayes exact GP
- target: hydration free energy, `expt`

The GP uses 30 named features: 20 features from a SMILES-decoded adjacency graph and 10 MolADT descriptor extensions. It fits an RBF-kernel exact GP on the deterministic seed-18 FreeSolv split.

The feature manifest is written to `feature_manifest.csv` for every repeated-split run.

The fitted kernel is:

```text
k(x, x') =
  signal_variance * exp(-||z(x) - z(x')||^2 / (2 * lengthscale^2))
```

`z(x)` is the standardized small-feature vector.

The benchmark writes predictive metrics, predictions, fitted kernel weights, target scaling, and noise variance under `results/freesolv/run_<timestamp>/details/`.

Inverse design always loads the latest `results/freesolv/run_*` artifact.

For repeated-split uncertainty, run:

```bash
make freesolv-20split
```

That target writes split-level metrics and the exact molecule assignments under
`results/freesolv_small_feature_gp/run_<timestamp>/`.

For the representation ablation, run:

```bash
make freesolv-ablation
```

That target writes `predictive_metrics.csv`, `summary.csv`,
`paired_against_full_moladt.csv`, `feature_manifest.csv`, and
`freesolv_small_feature_ablation.svg` under
`results/freesolv_ablation/run_<timestamp>/`. The ladder is:

- atom bag: 10 SMILES atom-count features
- SMILES adjacency graph: the atom bag plus bond counts, component count, cycle rank, and degree summaries
- full MolADT: the SMILES graph features plus 10 MolADT descriptor additions

| Label | Variant | Meaning | Test RMSE |
| --- | --- | --- | ---: |
| A | atom bag | 10 SMILES atom-count features | `1.971 +/- 0.567` |
| B | SMILES adjacency graph | 20 graph-only features | `1.791 +/- 0.505` |
| C | full MolADT | 30 features: graph baseline plus MolADT descriptors | `1.308 +/- 0.461` |

This table is preferred over the legacy RBF descriptor comparison because it
isolates the representation question directly: ordinary graph structure versus
explicit Dietz bonding systems.

Feature example:

```python
from benchmarking.features import compute_moladt_featurized_descriptors
from moladt.examples import ferrocene_pretty

features = compute_moladt_featurized_descriptors(ferrocene_pretty)

print(features["weight"])
print(features["bonding_system_count"])
print(features["system_shared_electrons_sum"])
```

Prediction example after `make freesolv`:

```python
from experiments.freesolv_inverse_design import FreeSolvBayesianPredictor
from moladt.examples import water

model = FreeSolvBayesianPredictor.load()
prediction = model.predict(water)

print(prediction.mean, prediction.sd)
```

## QM9

Current path:

- task: `mu`
- geometry representation: `moladt_featurized_geom`
- model: `visnet_ensemble`
- split: deterministic `80/10/10`
- training cap: `25` epochs

`make qm9long` runs the geometry path, not the Stan shortcut.

## Feature Families

`moladt_featurized` includes:

- element counts
- charge summaries
- local bond summaries
- Dietz bonding-system summaries
- effective bond-order features
- radial, angle, and torsion channels

The key point: all features are computed from the typed `Molecule`, not from a lossy string proxy.

## Registered Model Names

- `moladt_full30_rbf_gp`
- `bayes_linear_student_t`
- `bayes_hierarchical_shrinkage`
- `catboost_uncertainty`
- `visnet_ensemble`
- `dimenetpp_ensemble`

See [Inference and benchmarks](inference-and-benchmarks.md) for run details and [Outputs](outputs.md) for generated files.
