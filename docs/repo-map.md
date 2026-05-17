# Repo Map

Main folders:

| Path | Purpose |
| --- | --- |
| `moladt/chem/` | Core molecule, Dietz bonding, coordinates, validation, pretty printing. |
| `moladt/io/` | SDF, SMILES, and MolADT JSON. |
| `moladt/examples/` | Built-in molecules and manuscript examples. |
| `moladt/inference/` | Descriptor code over MolADT molecules. |
| `benchmarking/` | Data processing, benchmark orchestration, reports, timing, and model runners. |
| `stan/` | Stan model files. |
| `molecules/` | Small SDF examples. |
| `data/` | Raw and processed benchmark data. |
| `results/` | Local and reference outputs. |
| `tests/` | Pytest coverage. |
| `docs/` | Project documentation. |
| `docs/freesolv-gp-feature-list.md` | Current 30-feature FreeSolv GP list. |

Useful entrypoints:

- `moladt/cli.py`
- `experiments/freesolv_inverse_design.py`
- `benchmarking/run_all.py`
- `Makefile`
