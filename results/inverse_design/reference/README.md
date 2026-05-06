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

The molecule files are explicit MolADT values: atoms plus bonding systems. A
generated single, double, triple, or quadruple edge is stored as a one-edge
system sharing 2, 4, 6, or 8 electrons, so pretty-printing the files shows
electron sharing over the edge rather than a separate bond-order field.
Ionic contacts use the same system layer: the edge is a one-edge `ionic` system
with 0 shared electrons, and the formal charges live on the atoms.

Run `make freesolv` first when these molecules should be paired with a fresh FreeSolv benchmark.

Open the saved top 10 in the browser:

```bash
make inverse-design-view
```
