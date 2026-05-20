# Results

Local runs write timestamped folders here.

Most generated files are ignored. Small reference artifacts can be checked in when they help review or reproduce a result.

Molecule references use the current MolADT shape: atoms plus bonding systems,
with each generated edge represented by a shared-electron system rather than an
edge-only bond order. Conventional generated covalent edges are one-edge
2/4/6/8e systems; ionic contacts, when present, are one-edge 0e systems with
formal charge on the atoms.

## Main Folders

| Folder | Produced by |
| --- | --- |
| `freesolv_ablation/` | `make freesolv-ablation` |
| `inverse_design/` | `make inverse-design TARGET=-5.0` |
| `qm9/` | `make qm9long` |
| `timing/` | `make timing` |

## Checked-In References

- FreeSolv keeps compact benchmark outputs and model artifacts.
- QM9 and timing keep paper-facing SVGs and captions.
- Inverse design keeps reference molecule exports under `inverse_design/reference/`.

Historical FreeSolv reference:

```text
results/freesolv_ablation/run_20260512_small_feature_ablation/
```

That run is the 20-split A/B/C small-feature ablation before the multigraph
feature-contract redo. The old full MolADT `moladt_full30_rbf_gp` row reports
`1.308 +/- 0.461` kcal/mol test RMSE. Re-run `make freesolv-ablation` before
citing updated RMSE for the current 30-feature MolADT multigraph panel.

For file names, see [docs/outputs.md](../docs/outputs.md). For benchmark meaning, see [docs/inference-and-benchmarks.md](../docs/inference-and-benchmarks.md).
