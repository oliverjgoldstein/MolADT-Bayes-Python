# Models and Features

This repo benchmarks MolADT by exporting descriptor matrices from the typed molecule object and fitting Stan models to those matrices.

## Benchmark Shape

- build a typed MolADT `Molecule`
- derive a numeric descriptor matrix from that object
- fit the benchmark model in Stan
- compare the local result to the matching MoleculeNet baseline

The benchmark input is the ordinary typed `Molecule` object. Both reviewer-facing predictive paths use the richer `moladt_featurized` branch.

## Representation Contract

The FreeSolv and QM9 benchmark paths use `moladt_featurized`, the richer SDF-backed branch with pair, radial, angle, torsion, and bonding-system channels.

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

This is the feature-rich MolADT branch used by the current FreeSolv and QM9 benchmark defaults.

It keeps the compact MolADT descriptors and adds:

- typed atom-pair count and interaction channels
- bonding-system summaries
- effective-bond-order bucket counts
- APRDF-like radial channels
- bond-angle channels
- torsion channels

This branch is exported from the aligned SDF-backed MolADT molecules. It keeps all `642` local FreeSolv structures in the benchmark split and is also the default QM9 representation because the `mu` task depends on 3D geometry and directional structure.

## Stan Boundary

Stan does not fit directly over string objects. The Stan models in this repo consume numeric matrices exported from MolADT descriptors.

## Model Families

The repo contains these Stan model families:

- `bayes_linear_student_t`
- `bayes_hierarchical_shrinkage`
- `bayes_gp_rbf_screened`

`bayes_gp_rbf_screened` is the FreeSolv benchmark model. It screens the richer featurized descriptor matrix down to the strongest training-only channels and then fits an exact RBF Gaussian process in Stan. It was chosen because it performed better than the linear baselines on local validation for the `642`-molecule FreeSolv task.

For FreeSolv, the benchmark contract is:

- representation: `moladt_featurized`
- model: `bayes_gp_rbf_screened`
- algorithm: `laplace`

For QM9, the benchmark contract is:

- representation: `moladt_featurized`
- model: `bayes_linear_student_t`
- algorithm: `optimize`

That QM9 choice is based on two constraints. First, the literature on QM9 `mu` consistently rewards geometry-aware and direction-aware models, so the compact descriptor set is the wrong default. Second, exact GP inference is not practical at full QM9 scale, and the pasted long NUTS run is a clear example of why full sampling is the wrong default here. On the local `QM9_LIMIT=2000` subset, the featurized Student-`t` path outperformed the compact and grouped-shrinkage Stan alternatives on validation MAE, and `optimize` slightly beat the other scalable Stan fits on validation MAE while being much faster than `pathfinder`, so that is the fixed reviewer-facing Stan path.

Reports compare the fixed FreeSolv benchmark run and the fixed QM9 Stan benchmark run to MoleculeNet only:

- FreeSolv: local MolADT RMSE versus MoleculeNet MPNN RMSE `1.15`
- QM9 `mu`: local MolADT MAE versus MoleculeNet DTNN MAE `2.35`

## Example Code

This is the smallest programmatic example of the fixed QM9 Stan path:

```python
from scripts.process_qm9 import process_qm9_dataset
from scripts.stan_runner import StanRunConfig, run_model_suite

artifacts = process_qm9_dataset(limit=2000, split_mode="subset", verbose=True)
bundle = artifacts.moladt_featurized_export
assert bundle is not None

config = StanRunConfig(
    methods=("optimize",),
    optimize_iterations=2000,
    predictive_draws=500,
)

summary_rows, prediction_rows, coefficient_rows = run_model_suite(
    bundle,
    model_name="bayes_linear_student_t",
    config=config,
)
```

That call path uses the typed MolADT object only indirectly: Python first featurizes the MolADT molecules into the exported `X/y` bundle, and then Stan fits the model from that numeric matrix. The corresponding Stan sources are [`stan/bayes_linear_student_t.stan`](../stan/bayes_linear_student_t.stan) and [`stan/bayes_gp_rbf_screened.stan`](../stan/bayes_gp_rbf_screened.stan).

## Benchmark Commands

```bash
make freesolv
make qm9
make benchmark-small
make timing
```

- `make freesolv` writes the `Training` / `Validation` / `Test` / `Paper` FreeSolv figure
- `make qm9` writes the long-run `Training` / `Test` / `Paper` QM9 figure for the fixed `moladt_featurized + bayes_linear_student_t + optimize` path
- `make benchmark-small` keeps the lighter subset benchmark path available
- `make timing` is the separate ingest/interoperability timing path

For the broader protocol and result bundle, see [Inference and benchmarks](inference-and-benchmarks.md).
