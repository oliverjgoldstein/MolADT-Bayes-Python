# Models and Features

This repo benchmarks MolADT by exporting descriptor matrices from the typed molecule object and fitting Stan models to those matrices.

## Benchmark Shape

- build a typed MolADT `Molecule`
- derive a numeric descriptor matrix from that object
- fit the benchmark model in Stan
- compare the local result to the matching MoleculeNet baseline

The benchmark input is the ordinary typed `Molecule` object. FreeSolv uses the richer `moladt_featurized` branch, while QM9 uses the compact `moladt` branch.

## Representation Contract

The FreeSolv benchmark uses `moladt_featurized`, the richer SDF-backed branch with pair, radial, angle, torsion, and bonding-system channels.

QM9 uses the compact `moladt` branch for the Stan comparison.

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

This is the FreeSolv feature-rich branch.

It keeps the compact MolADT descriptors and adds:

- typed atom-pair count and interaction channels
- bonding-system summaries
- effective-bond-order bucket counts
- APRDF-like radial channels
- bond-angle channels
- torsion channels

This branch is exported from the aligned SDF-backed MolADT molecules, so it keeps all `642` local FreeSolv structures in the benchmark split.

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

The linear models remain in the repo for QM9 and side experiments.

Reports compare the fixed FreeSolv benchmark run and the selected QM9 local run to MoleculeNet only:

- FreeSolv: local MolADT RMSE versus MoleculeNet MPNN RMSE `1.15`
- QM9 `mu`: local MolADT MAE versus MoleculeNet DTNN MAE `2.35`

## Benchmark Commands

```bash
make freesolv
make qm9
make benchmark-small
make timing
```

- `make freesolv` writes the `Training` / `Validation` / `Test` / `Paper` FreeSolv figure
- `make qm9` writes the long-run `Training` / `Test` / `Paper` QM9 figure
- `make benchmark-small` keeps the lighter subset benchmark path available
- `make timing` is the separate ingest/interoperability timing path

For the broader protocol and result bundle, see [Inference and benchmarks](inference-and-benchmarks.md).
