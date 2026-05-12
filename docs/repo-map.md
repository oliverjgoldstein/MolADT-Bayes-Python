# Repo Map

Main folders:

| Path | Purpose |
| --- | --- |
| `moladt/chem/` | Core molecule, Dietz bonding, coordinates, validation, pretty printing. |
| `moladt/io/` | SDF, SMILES, and MolADT JSON. |
| `moladt/examples/` | Built-in molecules and manuscript examples. |
| `moladt/inference/` | Descriptor code over MolADT molecules. |
| `scripts/` | Data processing, benchmark orchestration, reports, timing. |
| `stan/` | Stan model files. |
| `molecules/` | Small SDF examples. |
| `data/` | Raw and processed benchmark data. |
| `results/` | Local and reference outputs. |
| `tests/` | Pytest coverage. |
| `docs/` | Project documentation. |

Useful entrypoints:

- `moladt/cli.py`
- `experiments/freesolv_inverse_design.py`
- `scripts/run_all.py`
- `Makefile`
- [`docs/freesolv-gp-feature-list.md`](freesolv-gp-feature-list.md): exact sparse token vocabulary for the default FreeSolv GP.
- [`docs/freesolv-gp-feature-layman.md`](freesolv-gp-feature-layman.md): plain-English descriptions for each FreeSolv GP feature index.
