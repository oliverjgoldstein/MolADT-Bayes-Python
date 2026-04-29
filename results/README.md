# Results

Local runs write timestamped folders here.

Most generated files are ignored. Small reference artifacts can be checked in when they help review or reproduce a result.

## Main Folders

| Folder | Produced by |
| --- | --- |
| `freesolv/` | `make freesolv` |
| `inverse_design/` | `make inverse-design TARGET=-5.0` |
| `qm9/` | `make qm9long` |
| `timing/` | `make timing` |

## Checked-In References

- FreeSolv keeps compact benchmark outputs and model artifacts.
- QM9 and timing keep paper-facing SVGs and captions.
- Inverse design keeps reference molecule exports under `inverse_design/reference/`.

For file names, see [docs/outputs.md](../docs/outputs.md). For benchmark meaning, see [docs/inference-and-benchmarks.md](../docs/inference-and-benchmarks.md).
