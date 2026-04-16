# Models and Features

This repo benchmarks MolADT by exporting descriptor matrices from the typed molecule object and then fitting either Stan or geometry-aware neural models to those exports.

## Benchmark Paths

The two user-facing benchmark commands are intentionally different:

- FreeSolv keeps the fixed Stan GP path:
  - representation: `moladt_featurized`
  - model: `bayes_gp_rbf_screened`
  - algorithm: Stan `laplace`
- QM9 uses one geometry path:
  - geometry representation: `moladt_featurized_geom`
  - geometry model: `visnet_ensemble`
  - long split: deterministic `80/10/10` over all aligned local QM9 rows
  - current local counts: `107,108 / 13,388 / 13,389`
  - geometry epochs: `25`
  - default seed: `102`
  - geometry members: `1`

`make qm9long` does not use the Stan path. It runs a full-data ViSNet benchmark on `moladt_featurized_geom` and plots the local result against the MoleculeNet DTNN MAE row.

## Representation Contract

The FreeSolv and QM9 benchmark paths both rely on `moladt_featurized`, the richer SDF-backed branch with pair, radial, angle, torsion, and bonding-system channels.

Boundary SMILES still matter because many datasets ship with SMILES strings, but the benchmark object is the typed `Molecule` ADT derived from that boundary string or aligned SDF record.

## Why `moladt` Is The Typed Model

There is no separate benchmark meaning of "typed" beyond the normal MolADT object.

The typed chemistry object is already the ordinary `Molecule` class:

- atoms carry typed attributes
- sigma edges are explicit
- Dietz bonding systems are explicit
- shell and orbital data remain attached to atoms

The benchmark already uses the ordinary typed `Molecule` class.

## What The Features Mean

### `moladt`

This is the typed molecule-object baseline.

The exported descriptors come from the MolADT object and include:

- element counts
- formal-charge summaries
- sigma-edge and bonding-system counts
- effective bond-order summaries
- aromatic-ring and ring-edge summaries
- donor/acceptor and other compact topology summaries

For the published benchmark comparison, these descriptors are computed from the typed molecule object and then fed into Stan.

### `moladt_featurized`

This is the feature-rich MolADT branch used by the current FreeSolv benchmark and as the global descriptor source for the current QM9 geometry benchmark.

It keeps the compact MolADT descriptors and adds:

- typed atom-pair count and interaction channels
- bonding-system summaries
- effective-bond-order bucket counts
- APRDF-like radial channels
- bond-angle channels
- torsion channels

This branch is exported from the aligned SDF-backed MolADT molecules. It keeps all `642` local FreeSolv structures in the benchmark split and is also the default QM9 representation because the `mu` task depends on 3D geometry and directional structure.

## Model Families

The repo contains these predictive model families:

- `bayes_linear_student_t`
- `bayes_hierarchical_shrinkage`
- `bayes_gp_rbf_screened`
- `visnet_ensemble`
- `dimenetpp_ensemble`

## Stan Boundary

Stan does not fit directly over string objects. The Stan models in this repo consume numeric matrices exported from MolADT descriptors.

`bayes_gp_rbf_screened` is the FreeSolv benchmark model. It screens the richer featurized descriptor matrix down to the strongest training-only channels and then fits an exact RBF Gaussian process in Stan. It was chosen because it performed better than the linear baselines on local validation for the `642`-molecule FreeSolv task.

For FreeSolv, the benchmark contract is:

- representation: `moladt_featurized`
- model: `bayes_gp_rbf_screened`
- algorithm: `laplace`

For QM9, the benchmark contract is:

- geometry export: `moladt_featurized_geom`
- geometry model: `visnet_ensemble`
- split mode: `long`
- run size: all aligned local QM9 rows under the deterministic `80/10/10` split
- current local counts: `107,108 / 13,388 / 13,389`
- geometry training cap: `25` epochs with one member

That QM9 choice is deliberate. `mu` is a directional 3D target, so the benchmark uses a geometry model with the full MolADT feature bundle derived from the aligned SDF molecules. The long run is a geometry benchmark, not the older Stan shortcut.

Reports compare the fixed FreeSolv benchmark run and the full-data QM9 predictive run to MoleculeNet only:

- FreeSolv: local MolADT RMSE versus MoleculeNet MPNN RMSE `1.15`
- QM9 `mu`: local MolADT MAE versus MoleculeNet DTNN MAE `2.35`

## Example Code

This is the smallest programmatic example of the current QM9 geometry path:

```python
from scripts.process_qm9 import process_qm9_dataset
from scripts.geometry_runner import GeometryRunConfig, run_geometry_ensemble

artifacts = process_qm9_dataset(split_mode="long", include_legacy_tabular=False, verbose=True)
bundle = artifacts.geometric_exports["moladt_featurized_geom"]
metrics_rows, prediction_rows, training_curve_rows, artifact_rows = run_geometry_ensemble(
    bundle,
    config=GeometryRunConfig(model_name="visnet_ensemble", seeds=(102,), verbose=True),
)
```

`make qm9long` runs exactly that ViSNet path on `moladt_featurized_geom`, then writes the local QM9 row for the final paper-comparison figure.

## Benchmark Commands

```bash
make freesolv
make qm9long
make timing
```

- `make freesolv` writes the `Training` / `Validation` / `Test` / `Paper` FreeSolv figure
- `make qm9long` writes the `Training` / `Test` / `Paper` QM9 figure for the full-data ViSNet path
- `make timing` is the separate four-stage SMILES-vs-MolADT timing path

For the broader protocol and result bundle, see [Inference and benchmarks](inference-and-benchmarks.md).
