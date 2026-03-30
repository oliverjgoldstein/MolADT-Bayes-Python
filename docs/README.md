# Python Docs

This repo is the main execution path for MolADT. If you want the representation, the orbitals, the timing story, the Stan models, and the full benchmark run, start here.

## Molecule Representation

The core molecule type keeps atoms, sigma bonds, and Dietz-style bonding systems together. It is meant to stay readable, typed, and inspectable.

## Orbitals

Atoms keep explicit shell and orbital data. The pretty-printer shows that directly for small molecules and built-in examples.

## Timing

The timing story lives in the ZINC benchmark:

- `raw_file_read`
- `smiles_parse_sanitize`
- `smiles_canonicalization`
- `moladt_parse_render` when MolADT timing is enabled

## Model

The predictive benchmark fits the Stan models, writes `results/`, and exports aligned matrices under `data/processed/` for the Haskell repo.

## Run Everything

```bash
make python-setup
make python-cmdstan-install
make benchmark INFERENCE_PRESET=paper INCLUDE_MOLADT=1
```

That is the full hours-long run and it keeps the benchmark output visible in the terminal.

## Pages

- [Inference and benchmarks](inference-and-benchmarks.md)
- [Examples](examples.md)
- [CLI](cli.md)
- [SMILES scope and validation](smiles-scope-and-validation.md)
- [Haskell interop](haskell_interop.md)
- [Quickstart](quickstart.md)
- [Repo map](repo-map.md)

## Related Repo

- [MolADT-Bayes-Haskell](https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell)
