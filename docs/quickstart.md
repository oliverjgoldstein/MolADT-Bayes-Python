# Quickstart

Use these commands from the repo root.

## Install

```bash
make python-setup
```

This creates `./.venv` and installs the package locally. It does not touch your system Python.

For Stan-backed FreeSolv runs, install CmdStan once:

```bash
make python-cmdstan-install
```

This creates `./.cmdstan`.

## Check The CLI

```bash
make python-parse
make python-pretty-example EXAMPLE=morphine
make view
make molecule-viewer VIEWER_INPUT=molecules/benzene.sdf
```

Those commands check SDF parsing, built-in examples, validation, and the standalone molecule viewer. `make view` opens six example molecules in one browser page. In the viewer, clicking an atom shows the shell and orbital data stored on that atom.

`make test-molecule-viewer` runs the viewer tests and then opens the configured viewer HTML in the default browser.

Use `OPEN_VIEWER=1` when you want the generated viewer HTML to open in the default browser:

```bash
OPEN_VIEWER=1 make python-pretty-example EXAMPLE=morphine
OPEN_VIEWER=1 make molecule-viewer VIEWER_INPUT=molecules/benzene.sdf
```

## Run Benchmarks

```bash
make freesolv
make inverse-design TARGET=-5.0
make qm9long
make timing
```

- `make freesolv` runs the FreeSolv Bayesian GP benchmark.
- `make inverse-design TARGET=-5.0` generates 1,000 valid FreeSolv candidates and writes the top 10 by the model's Bayesian credible score.
- `make qm9long` runs the full local QM9 `mu` path.
- `make timing` runs the ZINC representation timing comparison.

Run `make freesolv` before inverse design when you want inverse design paired with the latest FreeSolv benchmark artifact.

## Optional Checks

```bash
make python-test
make python-typecheck
```

## Common Fixes

If Ubuntu, Debian, or WSL is missing `ensurepip`:

```bash
sudo apt update
sudo apt install -y python3-venv
make python-setup
```

If the shell cannot find the venv Python, call it directly:

```bash
./.venv/bin/python -m moladt.cli --help
```

On Windows-style venvs, use:

```bash
./.venv/Scripts/python.exe -m moladt.cli --help
```
