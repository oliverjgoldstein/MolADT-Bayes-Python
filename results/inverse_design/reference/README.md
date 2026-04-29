# FreeSolv inverse-design reference molecules

This directory stores the Git-tracked reference molecule files produced by:

```bash
make inverse-design TARGET=-5.0
```

The default run starts all deterministic search chains from water. To regenerate the files from another supported seed molecule, run:

```bash
make inverse-design TARGET=-5.0 SEED_MOLECULE=methane
```

Each `top_*.py` or `dietz_*.py` file defines an importable, validated MolADT `molecule` object plus the seed molecule, fixed random seed, target, prediction, score, and formula metadata.
