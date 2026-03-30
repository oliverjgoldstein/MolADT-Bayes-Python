# Python Docs

This repo is the Python side of MolADT. It owns the main benchmark pipeline, writes the reviewer-facing files under `results/`, and exports aligned matrices under `data/processed/` for the Haskell baseline.

## Most Common Tasks

| Task | Page | Command |
| --- | --- | --- |
| Set up the environment | [Quickstart](quickstart.md) | `make python-setup` |
| Install CmdStan | [Quickstart](quickstart.md) | `make python-cmdstan-install` |
| Parse an SDF or SMILES string | [CLI](cli.md) | `./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf` |
| Inspect built-in non-classical examples | [Examples](examples.md) | `./.venv/bin/python -m moladt.cli pretty-example ferrocene` |
| Run the benchmark pipeline | [Inference and benchmarks](inference-and-benchmarks.md) | `make benchmark` |
| Add MolADT to ZINC timing | [Inference and benchmarks](inference-and-benchmarks.md) | `./.venv/bin/python -m scripts.run_all benchmark --include-moladt` |
| Check the conservative SMILES boundary | [SMILES scope and validation](smiles-scope-and-validation.md) | `./.venv/bin/python -m moladt.cli parse-smiles "c1ccccc1"` |
| Understand the repo layout | [Repo map](repo-map.md) | `rg --files moladt scripts tests` |
| Run the Haskell consumer against Python exports | [Haskell interop](haskell_interop.md) | `MOLADT_PROCESSED_DATA_DIR=... stack run moladtbayes -- infer-benchmark freesolv_smiles lwis` |

## Pages

- [Quickstart](quickstart.md)
- [Examples](examples.md)
- [CLI](cli.md)
- [Inference and benchmarks](inference-and-benchmarks.md)
- [SMILES scope and validation](smiles-scope-and-validation.md)
- [Repo map](repo-map.md)
- [Haskell interop](haskell_interop.md)

## Related Repo

- [MolADT-Bayes-Haskell](https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell)
