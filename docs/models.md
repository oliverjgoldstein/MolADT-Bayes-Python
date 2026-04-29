# Models

Models do not consume SMILES strings directly. They consume numeric features exported from typed MolADT molecules.

## Main Commands

```bash
make freesolv
make qm9long
make timing
```

## FreeSolv

Current path:

- representation: `moladt_featurized`
- model: `bayes_gp_rbf_screened`
- method: Stan `laplace`
- target: hydration free energy, `expt`

The GP uses screened MolADT features and writes coefficients plus posterior draws under `results/freesolv/run_<timestamp>/details/`.

Inverse design always loads the latest `results/freesolv/run_*` artifact.

Feature example:

```python
from scripts.features import compute_moladt_featurized_descriptors
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

- `bayes_linear_student_t`
- `bayes_hierarchical_shrinkage`
- `bayes_gp_rbf_screened`
- `catboost_uncertainty`
- `visnet_ensemble`
- `dimenetpp_ensemble`

See [Inference and benchmarks](inference-and-benchmarks.md) for run details and [Outputs](outputs.md) for generated files.
