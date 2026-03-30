# Repo Map

This page describes the current layout of the Python repo as it exists today.

## Top-Level Areas

### [`moladt/chem/`](../moladt/chem/)

Core MolADT chemistry types and helpers:

- constants and element tables
- coordinates and units
- Dietz-style bonding-system machinery
- molecule helpers and pretty printing
- structural validation

### [`moladt/examples/`](../moladt/examples/)

Built-in example molecules and manuscript-facing render wrappers:

- benzene
- water and other small sample molecules
- diborane
- ferrocene

### [`moladt/inference/`](../moladt/inference/)

Lightweight descriptor code over the MolADT representation. This is the repo-local inference-oriented feature layer inside the package.

### [`moladt/io/`](../moladt/io/)

File and text I/O:

- SDF parsing and rendering
- conservative SMILES parsing and rendering

### [`moladt/stan/`](../moladt/stan/)

Current package namespace for Stan-related Python code. The actual Stan model files live in the top-level [`stan/`](../stan/) directory, and benchmark execution lives in [`scripts/stan_runner.py`](../scripts/stan_runner.py).

### [`scripts/`](../scripts/)

Benchmark orchestration and data-processing entrypoints:

- dataset download
- feature extraction
- deterministic train/valid/test export
- Stan fitting
- summary/report generation
- ZINC timing

### [`molecules/`](../molecules/)

Small file-backed examples used for CLI demos and parser tests:

- `benzene.sdf`
- `water.sdf`

### [`tests/`](../tests/)

Pytest coverage for:

- benchmark/export plumbing
- example validity
- parser and SMILES round-trips
- pretty rendering
- reporting artifacts
- validation invariants

### [`docs/`](./)

GitHub-native documentation for setup, CLI use, benchmarks, SMILES scope, interop, and repo structure.

## Related Benchmark Files

- [`stan/`](../stan/)
- [`results/`](../results/)
- [`data/processed/`](../data/processed/)
