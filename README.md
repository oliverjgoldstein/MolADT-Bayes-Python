# MolADT-Bayes-Python

`MolADT-Bayes-Python` is the Python implementation of the MolADT chemistry model. The core stays typed and structural: frozen dataclasses, explicit atoms and coordinates, Dietz-style bonding systems, local validation, a lightweight SDF parser, and a conservative SMILES boundary.

This repository also contains the main empirical benchmark pipeline used in the workspace:

- FreeSolv smoke benchmark
- QM9 dipole-moment benchmark
- ZINC SMILES timing benchmark
- exported standardized feature matrices for the aligned Haskell baseline

## Default Path

From the workspace root, use these commands exactly as written first. They already use the default settings.

- `make python-setup`  
  Create the virtual environment and install the package plus benchmark dependencies.
- `make python-cmdstan-install`  
  Install CmdStan once for the benchmark pipeline.
- `make python-test`  
  Run the Python test suite.
- `make benchmark`  
  Run the default Python benchmark bundle.
- `make benchmark-bg`  
  Run the same default benchmark in the background and write logs to `MolADT-Bayes-Python/results/benchmark.out`.
- `make showcase`  
  Run the shared workspace bundle, including the Python benchmark outputs.
- `make help`  
  Show the smaller set of optional commands and overrides.

Most users only need `make benchmark`.

## Install

Repository-local install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Windows users should use WSL2 for the benchmark stack.

## CmdStanPy

Preferred:

```bash
conda create -n stan -c conda-forge cmdstanpy
conda activate stan
```

Alternative:

```bash
python -m pip install --upgrade cmdstanpy
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

To point CmdStanPy at an existing CmdStan installation:

```python
from cmdstanpy import set_cmdstan_path

set_cmdstan_path("/path/to/cmdstan")
```

## Simple CLI

After installation, the simplest direct commands are:

```bash
python -m moladt.cli parse molecules/benzene.sdf
python -m moladt.cli parse-smiles "c1ccccc1"
python -m moladt.cli to-smiles molecules/benzene.sdf
python -m moladt.cli pretty-example ferrocene
python -m moladt.cli pretty-example diborane
```

The manuscript-facing rendering layer lives in `moladt/chem/pretty.py`, and the named examples live in `moladt/examples/manuscript.py`.

## Benchmark

The default benchmark run is:

```bash
make benchmark
```

It runs:

- the FreeSolv smoke property benchmark
- the default QM9 dipole benchmark subset
- the default ZINC timing benchmark

The main outputs are written to:

- `results/summary.md`
- `results/predictive_metrics.csv`
- `results/predictions.csv`
- `results/model_report.md`
- `results/model_coefficients.csv`
- `results/zinc_timing.csv`
- `results/zinc_timing.md`

For a long run in the background:

```bash
make benchmark-bg
tail -f MolADT-Bayes-Python/results/benchmark.out
```

If you later need non-default settings, use:

```bash
make help
python -m scripts.run_all --help
```

## SMILES Scope

The SMILES boundary is intentionally conservative. It supports:

- atoms and bracket atoms
- bracket hydrogens and formal charges
- branches and ring digits `1-9`
- single, double, and triple bonds
- benzene-style aromatic input such as `c1ccccc1`

It does not try to encode non-classical multicenter systems like diborane or ferrocene as SMILES. Those remain representable in the MolADT core, but `to-smiles` rejects structures outside the supported classical subset.

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
