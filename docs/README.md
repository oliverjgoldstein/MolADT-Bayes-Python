# Docs

Start here when you need more than the root [README](../README.md).

## Fast Path

| Need | Read |
| --- | --- |
| Install and run the first commands | [Quickstart](quickstart.md) |
| Understand the representation | [Representation](representation.md) |
| Inspect molecules | [Examples](examples.md), [Parsing](parsing.md), [CLI](cli.md) |
| Compare reordered molecules | [Molecule equality](molecule-equality.md) |
| Run models | [Models](models.md), [Inference and benchmarks](inference-and-benchmarks.md) |
| Find result files | [Outputs](outputs.md), [results README](../results/README.md) |

## Reference

- [Orbitals](orbitals.md): local shell and orbital records on atoms.
- [Molecule equality](molecule-equality.md): compare MolADT values modulo container ordering.
- [SMILES scope](smiles-scope-and-validation.md): what the conservative SMILES parser and renderer support.
- [Haskell interop](haskell_interop.md): shared JSON and processed matrix contracts.
- [Data sources](data-sources.md): FreeSolv, QM9, and ZINC inputs.
- [Repo map](repo-map.md): where the main code lives.

MolADT's core idea is simple: keep molecule structure as typed data, then parse, validate, featurize, score, and export that same object.
