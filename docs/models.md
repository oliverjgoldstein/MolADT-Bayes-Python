# Models and Benchmarks

The main README now points to three focused commands:

```bash
make freesolv
make qm9
make timing
```

These are the user-facing paths. They are intentionally narrower than the older full benchmark sweep.

## FreeSolv

```bash
make freesolv
```

This runs the hydration free energy benchmark on `expt`.

It uses:

- `catboost_uncertainty` for the fair tabular `smiles` vs `moladt` vs `moladt_typed` comparison
- `dimenetpp_ensemble` for the geometry-aware `sdf_geom`, `moladt_geom`, and `moladt_typed_geom` rows

The repo vendors the raw FreeSolv source files for this path: `data/raw/freesolv/SAMPL.csv` plus `data/raw/freesolv/sdffiles/*.sdf`. Provenance and upstream links are listed in [Data sources](data-sources.md).

The point is to keep the strong shared tabular baseline while letting the geometry branch use the model family that currently best fits this task in the repo.

## QM9

```bash
make qm9
```

This runs the QM9 dipole moment benchmark on `mu`.

It uses:

- `catboost_uncertainty` for the fair tabular comparison
- `visnet_ensemble` for the geometry-aware rows

This matches the tensorial and geometry-heavy character of the dipole task better than reusing the same geometry preference as FreeSolv.

The repo vendors the normalized QM9 source files for this path: `data/raw/qm9/qm9.sdf` and `data/raw/qm9/qm9.sdf.csv`.

If you want the paper-sized split instead of the local default subset:

```bash
INFERENCE_PRESET=paper QM9_LIMIT= QM9_SPLIT_MODE=paper make qm9
```

That paper-style QM9 split is `110462 / 10000 / 10000`, for `130,462` molecules total.

## ZINC Timing

```bash
make timing
```

This builds the local matched timing corpus and measures:

- raw file IO
- RDKit SMILES parse and sanitize
- RDKit canonicalization
- local timing-library build
- MolADT SMILES parsing
- MolADT file parsing

The repo vendors the normalized ZINC source file for this path: `data/raw/zinc/zinc15_250K_2D.csv`.

## Why The README No Longer Starts With Stan

The old Stan baselines are still in the repo, but they are no longer the front-door story.

- They are useful for manuscript context and ablation.
- They are not the best user-facing demonstration of the representation.
- The focused commands keep the comparison centered on the strongest non-degenerate paths now in the repo.

## All Model Families Still Present

The repo still contains five predictive model families:

- `bayes_linear_student_t`
- `bayes_hierarchical_shrinkage`
- `catboost_uncertainty`
- `visnet_ensemble`
- `dimenetpp_ensemble`

The front-page commands just stop making the Stan baselines the default experience.

## Deep Reference

For the full protocol, legacy benchmark commands, and lower-level options, see [Inference and benchmarks](inference-and-benchmarks.md).
