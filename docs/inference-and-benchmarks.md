# Inference and Benchmarks

This repo prepares the benchmark datasets, exports aligned MolADT matrices, fits the configured models, and writes the comparison bundle.

## Main Commands

```bash
make python-cmdstan-install
make freesolv
make inverse-design TARGET=-5.0
make inverse-design TARGET=-5.0 SEED_MOLECULE=methane
make qm9long
make timing
python -m experiments.freesolv_inverse_design --target -5.0
python -m experiments.freesolv_inverse_design --target -5.0 --seed-molecule methane
./.venv/bin/python -m scripts.run_all freesolv --verbose
./.venv/bin/python -m scripts.run_all qm9 --split-mode long --include-moladt-predictive --models "" --extra-models visnet_ensemble --preferred-qm9-geometry-representation moladt_featurized_geom --verbose
./.venv/bin/python -m scripts.run_all benchmark --paper-mode --verbose
```

- `make freesolv` runs the long FreeSolv benchmark path, imports all `642` vendored SDF structures as the molecule source, and compares the fixed local Stan run against MoleculeNet Table 3 on RMSE
- `make inverse-design TARGET=-5.0` runs the deterministic FreeSolv MolADT inverse-design proof of concept from water by default and prints generated molecules to stdout
- `make qm9long` runs the full-data QM9 `mu` predictive path with `visnet_ensemble` on the SDF-backed `moladt_featurized_geom` export
- `make timing` runs the separate ZINC SDF timing/interoperability pass
- `make python-cmdstan-install` is the one-time local CmdStan install step required before the Stan benchmark targets

## Benchmark Contract

The benchmark contract is narrow on purpose:

- `moladt_featurized`
  The same boundary SMILES or aligned SDF record is parsed into the typed MolADT object and featurized from that object with pair, radial, angle, torsion, and bonding-system channels.
- the FreeSolv figure shows `Training`, `Validation`, `Test`, then `Paper`
- the QM9 figure shows `Training`, `Test`, then `Paper`
- FreeSolv uses one fixed benchmark path: `moladt_featurized` with the `bayes_gp_rbf_screened` model fit by Stan `laplace`
- QM9 uses one fixed path: `visnet_ensemble` on `moladt_featurized_geom`
- QM9 long uses all aligned local QM9 molecules under a deterministic `80/10/10` split. With the current `133,885`-row local bundle, that is `107,108 / 13,388 / 13,389`.
- QM9 long uses seed `102`, matching the second seed from the old QM9 optional-model seed schedule
- QM9 geometry training runs one ViSNet member for at most `25` epochs and logs every epoch in verbose mode with validation RMSE and MAE
- the paper bar is the matching MoleculeNet row only
- FreeSolv uses RMSE
- QM9 `mu` uses MAE

## What Stan Fits

Stan only consumes numeric `X/y` matrices here, so the benchmark is over MolADT-derived descriptor matrices rather than literal string objects.

The aligned Stan models are:

- [`bayes_linear_student_t`](../stan/bayes_linear_student_t.stan)
- [`bayes_hierarchical_shrinkage`](../stan/bayes_hierarchical_shrinkage.stan)
- [`bayes_gp_rbf_screened`](../stan/bayes_gp_rbf_screened.stan)

The benchmark graph uses the fixed FreeSolv Stan run and the fixed `qm9long` ViSNet run.

## Dataset Meaning

- FreeSolv compares the local MolADT RMSE against the MoleculeNet Table 3 MPNN RMSE row `1.15`.
- QM9 `mu` compares the local MolADT MAE against the MoleculeNet Table 3 DTNN MAE row `2.35`.

`make qm9long` does not use the old subset split. It runs on the full local QM9 alignment with seed `102` and a single ViSNet member.

## FreeSolv Context

The fixed local FreeSolv run is a single deterministic `513 / 64 / 65` split over the `642` SDF-backed rows, so it should be read as a repo-local reference point rather than a universal leaderboard claim. Recent local runs land around `0.74` test RMSE on that split. Published FreeSolv numbers worth keeping in view are:

| Source | FreeSolv RMSE | Uncertainty | Split note |
| --- | --- | --- | --- |
| This repo, fixed local MolADT path | `~0.74` | point estimate only | one deterministic `513 / 64 / 65` split |
| MoleculeNet MPNN baseline | `1.15` | not reported in the benchmark row | MoleculeNet random-split benchmark row |
| MolFCL | `1.045` | `± 0.160` | scaffold-split regression table |
| KANO as quoted in MolFCL | `1.142` | `± 0.258` | scaffold-split regression table |
| SCAGE | `0.802` | `± 0.033` | scaffold-split comparison table |
| MolProphecy | `0.796` | `± 0.09` | explicit `8:1:1` random split |

Those rows are useful for scale, but they are not all the same protocol. The local MolADT number is strongest when read as: better than the original MoleculeNet MPNN bar, in the same numerical range as stronger recent FreeSolv models, but still not a strict apples-to-apples SOTA claim.

Source links:

- MoleculeNet benchmark paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/
- MolFCL: https://academic.oup.com/bioinformatics/article/41/2/btaf061/8005854
- SCAGE: https://pmc.ncbi.nlm.nih.gov/articles/PMC12069555/
- MolProphecy: https://www.sciencedirect.com/science/article/pii/S2090123225008306?dgcid=rss_sd_all

## FreeSolv Inverse Design

The inverse-design runner is deliberately small:

```bash
python -m experiments.freesolv_inverse_design --target -5.0
```

`--target` and `--seed-molecule` are the only command-line flags. If `--target` is omitted, the runner uses the median experimental FreeSolv value from the processed dataset and prints that choice. The default `--seed-molecule` is `water`; all five deterministic search chains start from that same seed molecule with fixed RNG offsets. The number of steps, retained candidates, molecule-size limits, temperature, and random seed are fixed constants in `experiments/freesolv_inverse_design.py` so the run is reproducible.

The search reuses the checked-in FreeSolv `moladt_featurized + bayes_gp_rbf_screened + laplace` artifact. It loads the GP parameters from `results/freesolv/run_20260417_162536/details/model_coefficients.csv`, filtered to `freesolv / moladt_featurized / expt / bayes_gp_rbf_screened / laplace`, and fails fast if those rows or the matching processed metadata are not present. It proposes local MolADT edits, conservatively completes terminal hydrogens, validates the candidate, scores the Bayesian GP prediction against the target hydration free energy, and accepts by a compact Metropolis rule.

The implemented moves are:

- add a terminal atom to an atom with available valence
- add a local sigma edge for ring closure
- mutate a non-hydrogen atom while preserving its ID and incident structure
- remove a terminal atom that is not part of a multi-edge Dietz bonding system
- add a six-electron pi bonding system over an existing carbon six-ring

Every accepted candidate is checked with the repo validator, which accounts for local sigma edges plus Dietz bonding systems. The output prints diagnostics and a readable MolADT/Dietz listing for the top candidates, including atoms, local bonds, bonding systems, formula, predicted FreeSolv value, target error, and score.

The generator also applies conservative chemistry constraints before scoring: generated atoms are limited to CHONFCl, H-H local bonds are rejected, generated valence is capped at `H=1`, `C=4`, `N=3`, `O=2`, `F=1`, and `Cl=1`, duplicate bonding systems are rejected, and a `pi_ring` must be a simple carbon six-ring over existing local bonds. Candidate IDs are canonicalized after terminal-hydrogen completion so the printed and exported molecules are stable and readable.

`make inverse-design` sets `MOLADT_RESULTS_DIR=results/inverse_design/run_<timestamp>` and writes `top_01_molecule.py` through `top_10_molecule.py` for the best target-matching candidates. It also refreshes the Git-tracked reference copies in `results/inverse_design/reference/`. When valid candidates with Dietz bonding systems were seen, it writes `dietz_01_molecule.py` through `dietz_05_molecule.py`, making the representation-specific part of the experiment visible in the artifact. Each file defines an importable, validated `molecule` object plus the seed molecule, fixed random seed, candidate rank, and prediction metadata.

This is a manuscript-supporting proof of concept for property-conditioned growth, not a synthesizability claim and not a state-of-the-art generator.

## Timing

`make timing` is not the same question as the predictive benchmark.

With the default `paper` preset, it writes a ZINC timing bundle under `results/timing/paper/run_<timestamp>/`. If you override the preset away from `paper`, the path becomes `results/timing/run_<timestamp>/`. The timing bundle reports the fixed eight-stage SMILES-vs-MolADT comparison:

- raw SMILES CSV row reads
- cached `MolADT CSV -> MolADT`
- `SMILES -> JSON`
- cached `SDF -> MolADT`
- cached `SDF -> SMILES`
- local MolADT JSON serialization
- local MolADT JSON decoding
- local JSON-to-SMILES rendering

Treat it as an interoperability and runtime benchmark, not as the main model comparison. The timing SVG uses a log throughput axis so large stage gaps remain readable without overlapping labels.

## Outputs

Each predictive run writes a small timestamped folder with:

- `results.csv`
- `freesolv_rmse_vs_moleculenet.svg`
- `qm9_mae_vs_moleculenet.svg`
- `details/`

Important detail files include:

- `details/predictive_metrics.csv`
- `details/aggregated_predictive_metrics.csv`
- `details/predictions.csv`
- `details/model_coefficients.csv`
- `details/moleculenet_comparison.csv`

## Export Contract

The Python side also exports the aligned matrices consumed by the Haskell baseline:

- `*_X_train.csv`, `*_X_valid.csv`, `*_X_test.csv`
- `*_y_train.csv`, `*_y_valid.csv`, `*_y_test.csv`
- `*_metadata.json`

`X_train`, `X_valid`, and `X_test` are standardized from the training split only. `y` stays on the original target scale.

For the Haskell consumer view, see [Haskell interop](haskell_interop.md).
