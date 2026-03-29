# MolADT-Bayes-Python

`MolADT-Bayes-Python` is the Python implementation of the MolADT chemistry model. The core stays typed and structural: frozen dataclasses, explicit atoms and coordinates, Dietz-style bonding systems, local validation, a lightweight SDF parser, and a conservative SMILES boundary.

This repository also contains the main empirical benchmark pipeline used in the workspace:

- FreeSolv smoke benchmark
- QM9 dipole-moment benchmark
- ZINC SMILES timing benchmark
- exported standardized feature matrices for the aligned Haskell baseline

## Step By Step

### 1. Create the environment

From the workspace root:

```bash
make python-setup
```

That creates `MolADT-Bayes-Python/.venv` and installs the Python package plus the benchmark dependencies.

If you prefer to run the commands inside this repository directly:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Windows users should use WSL2 for the benchmark stack.

### 2. Install CmdStan

From the workspace root:

```bash
make python-cmdstan-install
```

This uses `cmdstanpy.install_cmdstan()` and does not require conda.

If you already have a CmdStan installation, point CmdStanPy at it with:

```python
from cmdstanpy import set_cmdstan_path

set_cmdstan_path("/path/to/cmdstan")
```

### 3. Check that the install works

From the workspace root:

```bash
make python-test
```

If you also want the typecheck:

```bash
make python-typecheck
```

### 4. Try the CLI on small examples

From this repository:

```bash
python -m moladt.cli parse molecules/benzene.sdf
python -m moladt.cli parse-smiles "c1ccccc1"
python -m moladt.cli to-smiles molecules/benzene.sdf
python -m moladt.cli pretty-example ferrocene
python -m moladt.cli pretty-example diborane
```

The manuscript-facing rendering layer lives in `moladt/chem/pretty.py`, and the named examples live in `moladt/examples/manuscript.py`.

### 5. Run the default benchmark

From the workspace root:

```bash
make benchmark
```

This runs, in order:

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

### 6. Run the long job in the background

From the workspace root:

```bash
make benchmark-bg
tail -f MolADT-Bayes-Python/results/benchmark.out
```

### 7. Run the full workspace bundle

If you want the shared Python + Haskell run from the workspace root:

```bash
make showcase
```

### 8. Need non-default settings?

Use:

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
