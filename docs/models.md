# Models

Models do not consume SMILES strings directly. They consume numeric features exported from typed MolADT molecules.

## Main Commands

```bash
make freesolv
make freesolv-20split
make qm9long
make timing
```

## FreeSolv

Current path:

- representation: `moladt`
- model: `moladt_wl_system_gp`
- method: empirical-Bayes exact GP
- target: hydration free energy, `expt`

The GP uses only the parsed MolADT representation. It vectorizes Weisfeiler-Lehman graph tokens and Dietz bonding-system tokens, then fits a weighted Tanimoto-kernel exact GP on the deterministic seed-18 FreeSolv split.

The token features include:

- element
- formal charge
- shell and orbital occupancy
- shared electrons
- effective bond order
- bonding-system overlap counts
- bonding-system kind, including covalent, ionic, bridge, pi, and coordination tags

The fitted kernel is:

```text
k(x, x') =
  w_all    * Tanimoto(WL + bonding-system tokens)
  + w_sys  * Tanimoto(bonding-system tokens)
  + w_wl   * Tanimoto(WL graph tokens)
```

Tanimoto is used because these are sparse non-negative token counts. It measures
shared token support relative to the active-token union, which is a better match
to the representation than Euclidean distance over a standardized descriptor
table.

The benchmark writes predictive metrics, predictions, fitted kernel weights, target scaling, and noise variance under `results/freesolv/run_<timestamp>/details/`.

Inverse design always loads the latest `results/freesolv/run_*` artifact.

For repeated-split uncertainty, run:

```bash
make freesolv-20split
```

That target writes split-level metrics and the exact molecule assignments under
`results/freesolv_20split/run_<timestamp>/`.

Difference from the previous GP:

- previous path: screened `moladt_featurized` scalar descriptor columns,
  standardized Euclidean distances, and an RBF kernel
- current default: sparse MolADT token counts from atoms, formal charges,
  orbitals, edges, and explicit bonding systems, compared with Tanimoto kernels
- no molecular fingerprints are used in the current default
- for a representation-led result, the current default is the primary model;
  the previous RBF GP is best treated as a baseline or ablation

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

- `moladt_wl_system_gp`
- `bayes_linear_student_t`
- `bayes_hierarchical_shrinkage`
- `catboost_uncertainty`
- `visnet_ensemble`
- `dimenetpp_ensemble`

See [Inference and benchmarks](inference-and-benchmarks.md) for run details and [Outputs](outputs.md) for generated files.
