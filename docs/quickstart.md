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
./.venv/bin/python -m moladt.cli parse-smiles "c1ccccc1"
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
```

If those three commands work, the local install is in good shape.

## 3. First Benchmark Run

If CmdStan is not installed yet, do this once first:

```bash
make python-cmdstan-install
```

Then start with the lightest path:

```bash
make benchmark-small
```

Then, when that works:

```bash
make qm9
make freesolv
make timing
```

- `make benchmark-small` is the quickest end-to-end benchmark check and keeps the older 2,000-row QM9 subset path.
- `make qm9` now runs the long full-data QM9 benchmark and writes `qm9_mae_vs_moleculenet.svg`.
- `make freesolv` runs the long FreeSolv comparison and writes `freesolv_rmse_vs_moleculenet.svg`.
- `make timing` builds the matched ADT/SMILES timing corpus and reports the CSV-string baseline alongside SMILES and MolADT parse stages.

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
