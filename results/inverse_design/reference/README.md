# FreeSolv Inverse-Design Reference

This folder stores the Git-trackable reference output from:

```bash
make inverse-design TARGET=-5.0
```

The run:

- starts from water by default
- uses the latest `results/freesolv/run_*` Bayesian GP artifact
- generates 1,000 valid molecules
- keeps the top 10 by the model's Bayesian credible score percentage

Tracked files:

- `top_01_molecule.py` through `top_10_molecule.py`
- `generated_molecules.csv`
- `generated_molecules.jsonl`

Run `make freesolv` first when these molecules should be paired with a fresh FreeSolv benchmark.
