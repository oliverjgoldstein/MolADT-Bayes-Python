# Inference And Benchmarks

This page is the run contract.

## Commands

```bash
make freesolv
make freesolv-20split
make freesolv-ablation
make inverse-design TARGET=-5.0
OPEN_VIEWER=1 make inverse-design TARGET=-5.0
make inverse-design-view
make qm9long
make timing
```

Direct inverse-design command:

```bash
python -m experiments.freesolv_inverse_design --target -5.0
python -m experiments.freesolv_inverse_design --target -5.0 --open-viewer
python -m experiments.freesolv_inverse_design --view-results results/inverse_design/reference --open-viewer
```

When viewer output is requested, the command prints a portable `file://` URL and
uses that same URL as the manual fallback if the operating system does not open a
browser automatically.

## FreeSolv Benchmark

`make freesolv` runs:

- dataset: FreeSolv, `642` SDF-backed rows
- split: deterministic `513 / 64 / 65`, seed `18`
- final fit: train plus validation rows, tested on the held-out 65 rows
- representation: `moladt_small_descriptors`
- model: `moladt_full30_rbf_gp`
- method: empirical-Bayes exact GP
- kernel: RBF over 30 standardized small features
- features: 30 MolADT-native composition, bonding-system, effective-order, and
  radial descriptors
- metric: RMSE

The result is a local benchmark artifact, not a universal leaderboard claim.

The comparison figure uses the MoleculeNet MPNN RMSE row `1.15` as the paper bar.

`make freesolv-20split` runs the same 30-feature full MolADT GP on 20
deterministic random splits, using seeds `0..19` by default. It writes:

- `predictive_metrics.csv`
- `predictions.csv`
- `model_coefficients.csv`
- `model_artifacts.csv`
- `split_assignments.csv`
- `summary.csv`

The target is for repeated-split uncertainty checks. It is separate from the
single seed-18 artifact used by inverse design.

`make freesolv-ablation` runs the representation ladder over the same 20
split seeds:

- atom bag: 10 SMILES atom-count features
- SMILES adjacency graph: 20 graph-only features
- full MolADT: 30 MolADT-native multigraph descriptors

It writes:

- `predictive_metrics.csv`
- `summary.csv`
- `paired_against_full_moladt.csv`
- `feature_manifest.csv`
- `freesolv_small_feature_ablation.svg`

The SMILES adjacency graph variant is deliberately limited to information a
SMILES-decoded graph provides: atom counts, bond-order counts, component count,
cycle rank, and heavy-atom degree summaries. It has no MolADT bonding-system
features, 3D geometry, shared-electron counts, or hashed/tokenized WL features.
The latest committed 20-split result before the multigraph feature redo had
full MolADT at `1.308 +/- 0.461` kcal/mol test RMSE. Re-run
`make freesolv-ablation` before citing an RMSE for the current C-row feature
contract.

## FreeSolv Inverse Design

`make inverse-design TARGET=-5.0` does this:

1. Loads the latest FreeSolv `moladt_full30_rbf_gp` artifact.
2. Samples initial molecules from the valid MolADT FreeSolv prior by default, reweighted by the unchanged GP target likelihood.
3. Generates at least `1,000` unique valid molecules.
4. Prints progress and ETA while loading/scoring the FreeSolv prior, periodic proposal-attempt progress, then one progress line per generated molecule with count and elapsed time.
5. Sorts the 1,000 generated molecules by the model's Bayesian credible score percentage.
6. Writes the top 10 as importable `top_*.py` files.
7. Writes geometry audit columns for bond lengths, van der Waals non-bonded clearance, and bond angles.
8. With `--open-viewer`, writes one combined viewer HTML page for the top 10 molecules by default.

Pass `SEED_MOLECULE=water` when you want the old fixed water start instead of the FreeSolv prior.

`make inverse-design-view` opens the saved top 10 molecules from `results/inverse_design/reference/` without rerunning inverse design. To inspect a timestamped run, pass `INVERSE_DESIGN_VIEW_DIR=results/inverse_design/run_...`.

The credible score is from the model's posterior predictive perspective. It is a bounded `0..100` value derived from the target log credible score. It is not a frequentist confidence or coverage statement.

Python API:

```python
from experiments.freesolv_inverse_design import molecular_formula, run_inverse_design

result = run_inverse_design(
    target=-5.0,
    min_unique_valid_molecules=1_000,
    top_k=10,
)

for candidate in result.top_candidates:
    print(
        molecular_formula(candidate.molecule),
        round(candidate.predicted_mean, 3),
        round(candidate.bayesian_credible_score_percent, 2),
    )
```

The generator builds molecules under conservative rules:

- connected
- neutral
- CHONFCl only
- no H-H edges
- closed valence shells
- terminal H/F/Cl rules
- valid Dietz bonding systems

The move rules avoid impossible local valence states during generation. Geometry values are still audited and written to the result files, but the inverse-design sampler is not conditioned on whether a candidate is physically plausible.

Run `make freesolv` first when inverse design should use a fresh FreeSolv model.

## QM9

`make qm9long` runs:

- task: QM9 `mu`
- representation: `moladt_featurized_geom`
- model: `visnet_ensemble`
- split: deterministic `80/10/10`
- metric: MAE
- paper bar: MoleculeNet DTNN MAE `2.35`

This is the full local geometry path.

## Timing

`make timing` runs a ZINC interoperability benchmark. It compares:

- raw SMILES CSV reads
- MolADT CSV decode
- SMILES to JSON
- SDF to MolADT
- SDF to SMILES
- MolADT to JSON
- JSON to MolADT
- JSON to SMILES

Treat timing as a representation/runtime comparison, not as a predictive benchmark.

## Outputs

Predictive runs write:

- `results.csv`
- comparison SVGs
- `caption.txt`
- `details/`

Inverse design writes:

- `results/inverse_design/run_<timestamp>/`
- `results/inverse_design/reference/`

See [Outputs](outputs.md) for file names.

## Haskell Export

The Python side also writes standardized matrices for the Haskell repo:

- `*_X_train.csv`, `*_X_valid.csv`, `*_X_test.csv`
- `*_y_train.csv`, `*_y_valid.csv`, `*_y_test.csv`
- `*_metadata.json`

`X` is standardized from the training split only. `y` stays on the original target scale.
