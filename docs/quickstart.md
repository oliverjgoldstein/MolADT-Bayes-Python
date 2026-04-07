# Quickstart

This page is the shortest reliable path from a fresh clone to a working Python CLI and a first benchmark run.

## 1. Create the Environment

From the repo root:

```bash
make python-setup
```

That creates `.venv` and installs the package plus benchmark dependencies. The Makefile looks for `python3` first, then `python`, and requires Python 3.11+.

If you prefer direct commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev,ml,geom]"
```

Windows users should use WSL2 for the benchmark stack. The Makefile also recognizes Windows-style `.venv/Scripts` layouts when it is run from a POSIX shell, but WSL2 remains the documented Windows path.

If WSL or another apt-based Linux reports that `ensurepip` is unavailable, `make python-setup` now offers to fix it for you. Type `y` and it will run `apt-get update` plus the relevant `venv` package install. That may still prompt for your sudo password.

If that automatic retry still fails, run the install manually:

```bash
sudo apt update
sudo apt install -y python3-venv
make python-setup
```

If your distro uses a versioned package name instead, install the package that matches `python3 --version`, for example `python3.10-venv` or `python3.12-venv`.

## 2. Install CmdStan If You Want The Stan Baselines

From the repo root:

```bash
make python-cmdstan-install
```

The focused front-page commands do not need this step. The local install target writes CmdStan under `.cmdstan/`.

## 3. Verify the Install

Run the test suite and typecheck:

```bash
make python-test
make python-typecheck
```

If you want plain `python` in your shell, activate the local virtual environment first:

```bash
source .venv/bin/activate
```

If you want the exact activation command for the venv that was created, run:

```bash
make python-activate
```

Otherwise use `./.venv/bin/python` explicitly in every command below on macOS, Linux, or WSL.

## 4. First Successful CLI Run

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
./.venv/bin/python -m moladt.cli parse-smiles "c1ccccc1"
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
```

The first command reads an SDF record, validates it, prints the record title, and pretty-prints the MolADT structure. The second parses and validates a conservative SMILES string. The third validates an SDF-backed molecule and renders it back to SMILES.

## 5. First Successful Benchmark Run

The focused user-facing commands are:

```bash
make freesolv
make qm9
make timing
```

Use them in that order the first time through.

- `make freesolv` is the lightest predictive path and the best first end-to-end check.
- `make qm9` is the focused dipole benchmark on the default local QM9 subset.
- `make timing` is the timing-only path that builds the matched ADT/SMILES corpus first.

The older broad benchmark wrappers still exist, but they are no longer the shortest recommended path:

```bash
make benchmark
make benchmark-paper
```

## Verify and Troubleshoot

Use these checks when something looks wrong:

```bash
./.venv/bin/python -m scripts.run_all --help
```

- Wrong Python path or no venv activation:
  use `./.venv/bin/python ...` directly on macOS, Linux, or WSL, or run `make python-activate` to print the right activation command for the venv layout that was created.
- Missing CmdStan:
  `scripts.run_all` will fail before fitting Stan models; run `make python-cmdstan-install` or point CmdStanPy at an existing install.
- Benchmark outputs not where you expect:
  focused runs now write to `results/freesolv/run_<timestamp>/`, `results/qm9/run_<timestamp>/`, or `results/timing/run_<timestamp>/`; the older broad wrappers still use `results/run_<timestamp>/` and `results/paper/run_<timestamp>/`; `MOLADT_RESULTS_DIR` overrides the root entirely.
- Unsupported SMILES:
  the parser and renderer only support the conservative subset documented in [SMILES scope and validation](smiles-scope-and-validation.md).

Relevant tests live under [tests/](../tests/). The parser/round-trip and example coverage are especially useful when changing user-facing behavior.
