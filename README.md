# MolADT-Bayes-Python

`MolADT-Bayes-Python` is the main benchmark repo for MolADT. It brings together the molecule representation, explicit orbital data, timing benchmark, Stan models, and the aligned train/valid/test exports used by the Haskell baseline.

MolADT is intended as a replacement molecular representation for bringing cheminformatics into the present day: explicit about orbitals and bonding, and less constrained by legacy graph-only assumptions.

Setup expects Python 3.11+ and a POSIX shell. On Windows, the documented path is WSL2.

## Molecule Representation

MolADT represents a molecule as atoms, localized sigma bonds, and Dietz-style bonding systems. Classical molecules fit cleanly. Non-classical examples such as diborane and ferrocene still fit the core representation even when they are outside the current SMILES renderer.

## Orbitals

Atoms keep explicit shell and orbital information. The pretty-printer exposes that directly, so the representation is not just a graph with labels.

## Timing

The ZINC benchmark measures:

- raw file read
- SMILES parse and sanitize
- SMILES canonicalization
- optional MolADT parse and render

## Model

The predictive benchmark fits two Stan models:

- `bayes_linear_student_t`
- `bayes_hierarchical_shrinkage`

The Python repo also writes aligned `data/processed/` exports. `X_train`, `X_valid`, and `X_test` are standardized from the training split only. `y` stays on the original scale.

## Run Everything

```bash
make python-setup
make python-cmdstan-install
make benchmark INFERENCE_PRESET=paper INCLUDE_MOLADT=1
```

That is the full hours-long run. It includes MolADT timing and writes the long-run artifacts under `results/paper/`.

## Docs

- [Docs index](docs/README.md)
- [Inference and benchmarks](docs/inference-and-benchmarks.md)
- [Examples](docs/examples.md)
- [CLI](docs/cli.md)
- [SMILES scope and validation](docs/smiles-scope-and-validation.md)
- [Haskell interop](docs/haskell_interop.md)

## Sibling Repo

- [MolADT-Bayes-Haskell](https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell)

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
