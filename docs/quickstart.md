# Quickstart

This page is the shortest reliable path from a fresh clone to a working Python CLI and a first benchmark run.

## 1. Create the Environment

From the repo root:

```bash
make python-setup
```

That creates `.venv` and installs the package plus benchmark dependencies.

If you prefer direct commands:

```bash
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev]"
```

Windows users should use WSL2 for the benchmark stack.

## 2. Install CmdStan

From the repo root:

```bash
make python-cmdstan-install
```

The local install target writes CmdStan under `.cmdstan/`.

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

Otherwise use `./.venv/bin/python` explicitly in every command below.

## 4. First Successful CLI Run

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
./.venv/bin/python -m moladt.cli parse-smiles "c1ccccc1"
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
```

The first command reads an SDF record, validates it, prints the record title, and pretty-prints the MolADT structure. The second parses and validates a conservative SMILES string. The third validates an SDF-backed molecule and renders it back to SMILES.

## 5. First Successful Benchmark Run

For a lighter first pass, start with the FreeSolv smoke benchmark instead of the full default benchmark:

```bash
./.venv/bin/python -m scripts.run_all smoke-test --methods optimize --models bayes_linear_student_t
```

After that works, the default full benchmark entrypoint is:

```bash
make benchmark
```

## Verify and Troubleshoot

Use these checks when something looks wrong:

```bash
./.venv/bin/python -m scripts.run_all --help
```

- Wrong Python path or no venv activation:
  use `./.venv/bin/python ...` directly, or run `source .venv/bin/activate`.
- Missing CmdStan:
  `scripts.run_all` will fail before fitting Stan models; run `make python-cmdstan-install` or point CmdStanPy at an existing install.
- Benchmark outputs not where you expect:
  default runs write to `results/`; `INFERENCE_PRESET=paper` writes to `results/paper/`; `MOLADT_RESULTS_DIR` overrides the root entirely.
- Unsupported SMILES:
  the parser and renderer only support the conservative subset documented in [SMILES scope and validation](smiles-scope-and-validation.md).

Relevant tests live under [tests/](../tests/). The parser/round-trip and example coverage are especially useful when changing user-facing behavior.
