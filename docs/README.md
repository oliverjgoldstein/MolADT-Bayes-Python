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
- `timing_library_prepare` when MolADT timing is enabled
- `smiles_library_parse` for matched MolADT-compatible SMILES entries
- `moladt_file_parse` for the local ADT JSON file library

## Model

The predictive benchmark fits the Stan models, writes a timestamped results bundle with one top-level `results.csv` plus three SVG graphs, and exports aligned matrices under `data/processed/` for the Haskell repo.

Core model files:

- [`stan/bayes_linear_student_t.stan`](../stan/bayes_linear_student_t.stan)
- [`stan/bayes_hierarchical_shrinkage.stan`](../stan/bayes_hierarchical_shrinkage.stan)

## Run Everything

```bash
make python-setup
make python-cmdstan-install
make timing
make benchmark-small
make benchmark-paper
```

`make timing` builds the local timing corpus and compares SMILES-entry parsing against MolADT-file parsing.
`make benchmark-small` is the default 2000-row QM9 subset run.
`make benchmark-paper` is the paper-sized QM9 count-matched run.

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
