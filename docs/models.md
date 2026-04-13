# Models and Features

This repo benchmarks MolADT by exporting descriptor matrices from the typed molecule object and then fitting either Stan, CatBoost, or geometry-aware neural models to those exports.

## Front-Door Benchmark Paths

The two user-facing benchmark commands are intentionally different:

- FreeSolv keeps the fixed Stan GP path:
  - representation: `moladt_featurized`
  - model: `bayes_gp_rbf_screened`
  - algorithm: Stan `laplace`
- QM9 keeps the recovered predictive path that was producing the strong `mu` results:
  - tabular representation: `moladt_featurized`
  - tabular model: `catboost_uncertainty`
  - geometry models: `visnet_ensemble` on the SDF-backed geometry exports
  - long split: deterministic `80/10/10` over all aligned local QM9 rows
  - current local counts: `107,108 / 13,388 / 13,389`
  - geometry epochs: `25`
  - default seed: `102`
  - geometry members: `1`

`make qm9long` does not use the fixed Stan path anymore. It runs the full-data CatBoost + ViSNet comparison and then plots the validation-selected local result against the MoleculeNet DTNN MAE row.

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

This is the feature-rich MolADT branch used by the current FreeSolv benchmark and the tabular side of the current QM9 benchmark.

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
- `catboost_uncertainty`
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

- tabular export: `moladt_featurized`
- tabular model: `catboost_uncertainty`
- geometry exports: `sdf_geom`, `moladt_geom`
- geometry model: `visnet_ensemble`
- split mode: `long`
- run size: all aligned local QM9 rows under the deterministic `80/10/10` split
- current local counts: `107,108 / 13,388 / 13,389`
- geometry training cap: `25` epochs with one member

That QM9 choice is deliberate. `mu` is a directional 3D target, so the front-door path uses a strong non-linear tabular baseline for the featurized MolADT descriptors and a geometry-aware ViSNet path for the coordinate-bearing export. The long run is still dominated by the geometry model path, not the older Stan `optimize` shortcut.

Reports compare the fixed FreeSolv benchmark run and the full-data QM9 predictive run to MoleculeNet only:

- FreeSolv: local MolADT RMSE versus MoleculeNet MPNN RMSE `1.15`
- QM9 `mu`: local MolADT MAE versus MoleculeNet DTNN MAE `2.35`

## Example Code

This is the smallest programmatic example of the recovered QM9 tabular path:

```python
from scripts.process_qm9 import process_qm9_dataset
from scripts.tabular_runner import CatBoostRunConfig, run_catboost_uncertainty

artifacts = process_qm9_dataset(split_mode="long", include_legacy_tabular=False, verbose=True)
bundle = artifacts.moladt_featurized_export
assert bundle is not None

config = CatBoostRunConfig(
    seeds=(1,),
    search_hyperparameters=True,
)

summary_rows, prediction_rows, artifact_rows = run_catboost_uncertainty(
    bundle,
    config=config,
)
```

`make qm9long` also runs `visnet_ensemble` on the geometry export, then keeps the best local validation-selected QM9 row for the final paper-comparison figure.

## Benchmark Commands

```bash
make freesolv
make qm9long
make timing
```

- `make freesolv` writes the `Training` / `Validation` / `Test` / `Paper` FreeSolv figure
- `make qm9long` writes the `Training` / `Test` / `Paper` QM9 figure for the full-data CatBoost + ViSNet path
- `make timing` is the separate ingest/interoperability timing path

For the broader protocol and result bundle, see [Inference and benchmarks](inference-and-benchmarks.md).
