# Outputs

Runs write timestamped folders under `results/`.

## Main Folders

| Command | Folder |
| --- | --- |
| `make freesolv` | `results/freesolv/run_<timestamp>/` |
| `make inverse-design` | `results/inverse_design/run_<timestamp>/` and `results/inverse_design/reference/` |
| `make qm9long` | `results/qm9/long/run_<timestamp>/` |
| `make timing` | `results/timing/paper/run_<timestamp>/` |

## Predictive Runs

Common files:

- `results.csv`
- `caption.txt`
- `freesolv_rmse_vs_moleculenet.svg`
- `qm9_mae_vs_moleculenet.svg`
- `details/`

Useful detail files:

- `details/predictive_metrics.csv`
- `details/aggregated_predictive_metrics.csv`
- `details/predictions.csv`
- `details/model_coefficients.csv`
- `details/moleculenet_comparison.csv`

## Inverse Design

The reference folder is Git-trackable:

- `top_01_molecule.py` through `top_10_molecule.py`
- `generated_molecules.csv`
- `generated_molecules.jsonl`

The top files are the 10 generated molecules with the highest model-side Bayesian credible score percentage.

Read the generated pool:

```python
import json
from pathlib import Path

path = Path("results/inverse_design/reference/generated_molecules.jsonl")

for line in path.open():
    record = json.loads(line)
    print(record["rank"], record["formula"], record["bayesian_credible_score_percent"])
    break
```

## Timing

Timing runs write:

- `timing_overview.svg`
- `caption.txt`
- `details/zinc_timing.csv`
- `details/zinc_timing_items.csv`
- `details/zinc_timing_corpus_manifest.csv`

The timing comparison covers SMILES reads, SDF parsing, MolADT JSON, and round trips back to MolADT or SMILES.
