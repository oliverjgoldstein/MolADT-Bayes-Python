# Models and Features

This page explains how the repo turns molecular data into model inputs.

## Our Approach

The approach is simple:

- build a typed MolADT `Molecule`
- turn that object into a numeric descriptor matrix
- run the local Stan sweep on that matrix
- keep the best local MolADT run
- compare that one result to MoleculeNet for FreeSolv and QM9

The short version is:

- the benchmark model input is the normal typed `Molecule` object
- the benchmark export is a MolADT descriptor matrix derived from that object
- Stan sweeps several inference methods and model variants, then the reporting layer keeps the single best MolADT row per dataset

## Representation Contract

The benchmark path now uses `moladt` only.

Boundary SMILES still matter because many datasets ship with SMILES strings, but the benchmark object is the typed `Molecule` ADT derived from that boundary string or aligned SDF record.

## Why `moladt` Is The Typed Model

There is no separate benchmark meaning of "typed" beyond the normal MolADT object.

The typed chemistry object is already the ordinary `Molecule` class:

- atoms carry typed attributes
- sigma edges are explicit
- Dietz bonding systems are explicit
- shell and orbital data remain attached to atoms

So the meaningful question is not "plain MolADT versus typed MolADT". The benchmark model already uses the ordinary typed `Molecule` class.

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

## Stan Boundary

Stan does not fit directly over string objects. The Stan models in this repo consume numeric matrices exported from MolADT descriptors.

## Model Families

The benchmark sweep contains:

- `bayes_linear_student_t`
- `bayes_hierarchical_shrinkage`

The reporting layer keeps the best local MolADT run from that Stan sweep and compares it to MoleculeNet only:

- FreeSolv: local MolADT RMSE versus MoleculeNet MPNN RMSE `1.15`
- QM9 `mu`: local MolADT MAE versus MoleculeNet DTNN MAE `2.35`

## Benchmark Commands

```bash
make freesolv
make qm9
make timing
```

- `make freesolv` writes `freesolv_rmse_vs_moleculenet.svg`
- `make qm9` writes `qm9_mae_vs_moleculenet.svg`
- `make timing` is the separate ingest/interoperability timing path

For the broader protocol and result bundle, see [Inference and benchmarks](inference-and-benchmarks.md).
