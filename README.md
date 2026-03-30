# MolADT-Bayes-Python

`MolADT-Bayes-Python` is the Python implementation of the MolADT chemistry model and the main benchmark producer for this project. It contains the typed MolADT core, conservative SDF/SMILES I/O, built-in manuscript examples, the CmdStanPy benchmark pipeline, and the aligned `data/processed/` exports consumed by the Haskell baseline.

This repo is for readers who want to inspect the Python model, run the CLI on small molecules, generate benchmark outputs under `results/`, or export aligned train/valid/test matrices for the sibling Haskell repo.

## Start Here

```bash
make python-setup
make python-cmdstan-install
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
./.venv/bin/python -m scripts.run_all smoke-test --methods optimize --models bayes_linear_student_t
```

## Docs

- [Docs index](docs/README.md)
- [Quickstart](docs/quickstart.md)
- [Examples](docs/examples.md)
- [CLI](docs/cli.md)
- [Inference and benchmarks](docs/inference-and-benchmarks.md)
- [SMILES scope and validation](docs/smiles-scope-and-validation.md)
- [Repo map](docs/repo-map.md)
- [Haskell interop](docs/haskell_interop.md)

## Sibling Repo

- [MolADT-Bayes-Haskell](https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell)

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
