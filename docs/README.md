# Documentation Index

The root [README](../README.md) gives the project overview and the longer motivating examples. This directory holds the task-specific reference docs.

MolADT has three main pieces:

- a typed molecule representation for chemistry-aware modeling and inverse design
- parsers, renderers, and JSON boundaries for moving molecules between tools and the sibling Haskell repo
- benchmark pipelines for FreeSolv, QM9, and timing comparisons

## Start Here

- [Quickstart](quickstart.md): install the local Python environment, run the first CLI checks, and start the main benchmarks.
- [MolADT representation](representation.md): explains why the representation is more than SMILES or a plain graph.
- [Examples](examples.md): lists the checked-in molecules and what each one demonstrates.
- [Parsing and rendering](parsing.md): covers SDF, SMILES, and MolADT JSON boundaries.

## Modeling And Benchmarks

- [Models and features](models.md): describes the descriptor bundles and model families, including the fixed FreeSolv and QM9 paths.
- [Inference and benchmarks](inference-and-benchmarks.md): gives the exact benchmark contract, split notes, and literature comparison context.
- [FreeSolv inverse design](inference-and-benchmarks.md#freesolv-inverse-design): documents the deterministic MolADT/Dietz growth experiment, its `--target` and `--seed-molecule` flags, and the tracked reference molecule files.
- [Outputs](outputs.md): explains generated result directories, figures, captions, CSVs, and summary files.
- [Results README](../results/README.md): explains the committed FreeSolv artifacts plus the checked-in QM9/timing graphs and captions.

## Representation Reference

- [ADT model](data-model.md): documents the Python record shape and its relationship to the Haskell ADT.
- [Orbitals and theoretical chemistry](orbitals.md): explains the orbital and shell records stored on atoms.
- [SMILES scope and validation](smiles-scope-and-validation.md): documents the conservative SMILES subset and validation behavior.

## Interop And Maintenance

- [Haskell interop](haskell_interop.md): describes the shared JSON and standardized `data/processed/` matrix contracts.
- [Data sources](data-sources.md): lists the vendored and downloadable FreeSolv, QM9, and ZINC sources.
- [CLI reference](cli.md): documents every `python -m moladt.cli` command.
- [Repo map](repo-map.md): gives the file-level project layout.
