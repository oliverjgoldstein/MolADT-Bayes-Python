# Models and Features

This page explains how MolADT is turned into model inputs.

The short version is:

- MolADT keeps more chemistry visible than a reduced graph or a SMILES string
- the repo then derives feature tables or geometry model inputs from that molecule object
- those features are what the predictive models consume

## Representation Branches

The main Python benchmark uses three MolADT-facing branches:

- `moladt`
- `moladt_typed`, shown as `MolADT+`
- `moladt_typed_geom`, shown as `MolADT+ 3D`

The comparison also includes `smiles` and `sdf_geom` baselines where appropriate.

## What The Features Mean

### `smiles`

This is the string-first baseline. Features come from the conventional SMILES path instead of the explicit MolADT object.

### `moladt`

This is the first molecule-object path. It uses descriptors derived from the explicit ADT, such as:

- atom and element counts
- bond and connectivity summaries
- bonding-system summaries
- geometry summaries when coordinates are available

The point is that the features are extracted from the chemistry object, not reverse-engineered from a string.

### `moladt_typed` / `MolADT+`

This is the richer descriptor branch. It adds more structured channels over the molecule, including:

- typed pair counts and interactions
- radial distance channels
- bond-angle channels
- torsion channels
- bonding-system summaries

This is where the representation starts to act less like a flat descriptor table and more like a typed chemistry object being projected into model space.

### `moladt_typed_geom` / `MolADT+ 3D`

This keeps the richer MolADT view and adds the geometry-aware model path on top.

That branch is for models that can exploit 3D structure instead of only tabular summaries.

## Model Families

The repo contains five predictive model families:

- `bayes_linear_student_t`
- `bayes_hierarchical_shrinkage`
- `catboost_uncertainty`
- `visnet_ensemble`
- `dimenetpp_ensemble`

They do not all consume the exact same input form.

- the Bayesian and CatBoost paths consume exported feature tables
- the geometry model families consume 3D molecular structure through their own model-specific pipelines

## Why This Matters

The representation is not there just to print prettier molecules.

It is there so models can be built over:

- explicit bonding systems
- typed local structure
- shell and orbital context
- geometry when available

That is the claim of the repo: richer chemistry objects can support richer model inputs.

## Benchmark Commands

The benchmark commands remain:

```bash
make freesolv
make qm9
make timing
```

- `make freesolv` is the lightest end-to-end run
- `make qm9` runs the focused dipole benchmark
- `make timing` measures SMILES vs MolADT ingest cost

For the broader protocol and benchmark details, see [Inference and benchmarks](inference-and-benchmarks.md).
