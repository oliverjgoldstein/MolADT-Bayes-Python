# Quickstart

This is the shortest path from a fresh clone to a working Python CLI and a first benchmark run.

## 1. Install

From the repo root:

```bash
make python-setup
```

That creates `.venv` and installs the package plus benchmark dependencies. Use Python 3.11+.

- macOS, Linux, WSL: use the command as-is.
- Windows: use WSL2 for the benchmark stack.

If your shell cannot find the venv Python later, run commands with `./.venv/bin/python ...` or print the activation command with:

```bash
make python-activate
```

## 2. First Successful CLI Run

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
./.venv/bin/python -m moladt.cli pretty-example morphine
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
```

If those three commands work, the local install is in good shape and the example path is reading the checked-in SDF files correctly.

## 3. First Benchmark Run

If CmdStan is not installed yet, do this once first:

```bash
make python-cmdstan-install
```

Then start with the faster benchmark:

```bash
make freesolv
```

Then, when that works:

```bash
make qm9long
make timing
```

- `make freesolv` runs the long FreeSolv comparison and writes `freesolv_rmse_vs_moleculenet.svg`.
- `make qm9long` runs the full-data QM9 path over all aligned local QM9 molecules, using `visnet_ensemble` only on the SDF-backed `moladt_featurized_geom` export. The geometry path caps at `25` epochs and logs every epoch with validation RMSE and MAE.
- `make timing` runs the four-stage ZINC SMILES-vs-MolADT timing comparison: source SMILES reads, SMILES parsing, MolADT JSON reads, and MolADT JSON decoding.

If a required raw dataset file is too large for GitHub, the repo fetches it on demand. Large downloads and archive extractions show live byte counts, entry counts, throughput, and elapsed time.

## 4. Optional

Run these when you want extra confidence:

```bash
make python-test
make python-typecheck
```

## 5. If Setup Fails

- Missing `ensurepip` on Ubuntu, Debian, or WSL:

```bash
sudo apt update
sudo apt install -y python3-venv
make python-setup
```

- Wrong Python or missing activation:
  use `./.venv/bin/python ...` directly.
- Unsupported SMILES:
  see [SMILES scope and validation](smiles-scope-and-validation.md).

For deeper details, use:

```bash
./.venv/bin/python -m scripts.run_all --help
```
