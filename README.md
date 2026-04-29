# MolADT-Bayes-Python

Molecular models often start from strings or plain graphs, then bolt on the chemistry that the representation could not express directly. That is awkward for Bayesian modelling and inverse design: the code wants to edit atoms, bonds, charges, geometry, stereochemistry, and delocalised or multicentre bonding, then validate the candidate before scoring it.

MolADT is the typed molecule object for that job. It keeps chemically meaningful structure in the data model itself, so candidate molecules can be mutated, validated, featurised, serialized, and shared with the Haskell repo without flattening the candidate back into a string-only boundary format.

[Quickstart](docs/quickstart.md) · [Docs index](docs/README.md) · [Representation](docs/representation.md) · [ADT model](docs/data-model.md) · [Models](docs/models.md) · [Examples](docs/examples.md) · [Outputs](docs/outputs.md)

## What This Repo Does

- Builds the Python MolADT toolkit used for parsing, validation, feature extraction, examples, and JSON interchange.
- Runs the paper-facing FreeSolv, QM9, and timing experiments.
- Demonstrates FreeSolv inverse design with deterministic, validity-checked MolADT/Dietz growth moves.
- Exports processed matrices and molecule payloads that the Haskell repo can consume.

Use this repo when you want the benchmark pipeline, Python-side modelling, generated result artifacts, or a practical molecule-growth experiment. Use the Haskell repo when you want the smaller typed reference implementation and aligned benchmark consumer.

## Why The Representation Matters

MolADT is designed for cases where ordinary graph or string encodings make the important chemistry hard to address directly:

- diborane wants explicit `3c-2e` bridge systems
- ferrocene wants shared Cp/metal bonding systems
- morphine wants fused-ring structure and stereochemical annotations available as data

That matters for inverse design because proposal operators can work on the object being optimised. A search step can add or remove a candidate bonding system, alter local connectivity, preserve stereo metadata, validate valence, and serialize the result for another model without routing the candidate through a lossy boundary string.

## Quick Start

For the complete setup walkthrough, see [docs/quickstart.md](docs/quickstart.md). The shortest local smoke test is:

```bash
make python-setup
make python-parse
# once, before Stan-backed targets such as FreeSolv
make python-cmdstan-install
```

`make python-setup` installs the Python package and its benchmark dependencies only into `./.venv` inside this repo. `make python-cmdstan-install` installs CmdStan only into `./.cmdstan`. Deleting `./.venv` removes the Python-side local install, deleting `./.cmdstan` removes the local CmdStan toolchain, and neither command touches your system Python, global site-packages, or other local environments.

If your shell creates a Windows-style virtual environment, the Make targets will use `.venv/Scripts/python.exe` automatically. On a fresh machine, the first local setup can take a few minutes and up to about 30 minutes if the larger dependencies still need to be downloaded or built.

`Molecule` is immutable. Access its fields directly as `molecule.atoms`, `molecule.local_bonds`, `molecule.systems`, and `molecule.smiles_stereochemistry`, or use `molecule_fields(molecule)` from `moladt.chem.molecule` when you want the tuple form.

For probabilistic proposals or local graph surgery, use `MutableMolecule` from `moladt.chem.mutable` as a writable scratch state and call `freeze()` to return to canonical `Molecule`.

## Examples

For the longer example catalogue, see [docs/examples.md](docs/examples.md). This README keeps the examples that explain why the representation exists.

The examples below use molecules that exercise the representation directly: morphine for fused rings and stereo, diborane for multicenter bonding, and ferrocene for organometallic bonding systems.

### 1. Inspect An Explicit Molecule

```bash
./.venv/bin/python -m moladt.cli pretty-example morphine
```

The built-in morphine object starts from the same fused skeleton as the standard stereochemical SMILES, but it keeps the graph and the extra chemical annotations as fields on the object:

```text
Morphine (explicit Dietz skeleton)
Dietz-style ADT that turns the five classic SMILES ring closures into sigma edges, keeps the phenyl ring as an explicit pi system, and preserves the five atom-centered stereochemistry flags from the standard boundary string.

Molecule Report
===============
atoms            21
heavy atoms      21
sigma bonds      25
bonding systems  2
net charge       +0
composition      C17 N O3
stereo flags     5 atom

Atoms
-----
atom    Z  chg  sigma  used  xyz (Angstrom)  sigma neighbors  systems
...

Electron Shells
---------------
...

Bonding Systems
---------------
[#1] alkene_bridge
  shared electrons: 2
  member atoms:     C#5, C#6
  edge bonus:       +1.00 to each listed edge

[#2] phenyl_pi_ring
  shared electrons: 6
  member atoms:     C#10, C#11, C#12, C#14, C#15, C#16
  edge bonus:       +0.50 to each listed edge

SMILES Stereochemistry
----------------------
atom-centered:
  center #2: TH1 from token @
  center #3: TH2 from token @@
  center #7: TH1 from token @
  center #8: TH1 from token @
  center #18: TH1 from token @
```

A plain SDF parse still works:

```bash
./.venv/bin/python -m moladt.cli parse molecules/morphine.sdf
```

That command reports the SDF sigma network. The manuscript example then adds the explicit `alkene_bridge`, `phenyl_pi_ring`, and stereo annotations that make the object useful for chemistry-aware modeling.

### 2. Round Trip A Rich Object Through Python And Haskell

MolADT JSON is the boundary format shared by this Python repo and the sibling Haskell repo. That makes it a practical candidate format for inverse design loops: Python can generate, mutate, featurize, and benchmark candidates while Haskell can consume the same structured molecule or the same standardized benchmark export.

```bash
./.venv/bin/python - <<'PY' > morphine.moladt.json
from moladt.examples import morphine_pretty
from moladt.io import molecule_to_json

print(molecule_to_json(morphine_pretty))
PY

./.venv/bin/python -m moladt.cli from-json morphine.moladt.json
```

The same JSON payload is accepted by the Haskell repo:

```bash
stack run moladtbayes -- from-json ../MolADT-Bayes-Python/morphine.moladt.json
```

The full benchmark export path is also shared. Python writes standardized `X/y` matrices under `data/processed/`, and Haskell can consume them without re-running Python feature extraction:

```bash
./.venv/bin/python -m scripts.run_all freesolv
MOLADT_PROCESSED_DATA_DIR=../MolADT-Bayes-Python/data/processed \
  stack run moladtbayes -- infer-benchmark freesolv_moladt_featurized lwis
```

For inverse design, the important point is the contract: candidates and benchmark rows can cross the repo boundary with the chemistry still explicit. The ADT keeps the atom table, local bonds, delocalized or multicenter bonding systems, stereo layer, and coordinates available to downstream code.

### 3. Use A Mutable Candidate During Search

`Molecule` is immutable on purpose. In a design loop, make a mutable scratch candidate, apply a proposal, validate it, then freeze it back to a canonical `Molecule`.

```python
from moladt import MutableMolecule
from moladt.chem.validate import validate_molecule
from moladt.examples import ferrocene_pretty
from moladt.io import molecule_from_json, molecule_to_json

def remove_bonding_system_by_tag(molecule, tag):
    candidate = MutableMolecule.from_molecule(molecule)
    candidate.systems = [
        (system_id, system)
        for system_id, system in candidate.systems
        if system.tag != tag
    ]
    return validate_molecule(candidate.freeze())

proposal = remove_bonding_system_by_tag(ferrocene_pretty, "fe_backdonation")

print(f"candidate atoms: {len(proposal.atoms)}")
print(f"candidate sigma bonds: {len(proposal.local_bonds)}")
print(f"candidate bonding systems: {[system.tag for _, system in proposal.systems]}")

payload = molecule_to_json(proposal)
round_tripped = validate_molecule(molecule_from_json(payload))
assert round_tripped == proposal
```

This is intentionally shaped like a proposal operator. A real inverse design run would replace the simple `remove_bonding_system_by_tag` step with a move set, score the frozen candidate with a predictive model, and keep or reject the proposal. The representation supports that style because the editable state and the canonical state have the same fields.

### 4. Compare String And ADT Views

Diborane and ferrocene are useful because compact line notations hide the structure that a design algorithm may need to manipulate.

```bash
./.venv/bin/python -m moladt.cli pretty-example diborane
./.venv/bin/python -m moladt.cli pretty-example ferrocene
```

In MolADT, diborane has two explicit `3c-2e` bridging hydrogen systems. Ferrocene has two cyclopentadienyl `pi` systems plus an Fe back-donation-style pool. Those systems are separate typed objects, so a proposal can preserve, remove, replace, or score them directly.

## Parsing Scope

For detailed parser and renderer behavior, see [docs/parsing.md](docs/parsing.md), [docs/cli.md](docs/cli.md), and [docs/smiles-scope-and-validation.md](docs/smiles-scope-and-validation.md).

Use the CLI when you want to inspect or serialize how a boundary format lands inside MolADT.

```bash
./.venv/bin/python -m moladt.cli parse molecules/morphine.sdf
./.venv/bin/python -m moladt.cli parse-smiles 'CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5'
./.venv/bin/python -m moladt.cli to-json molecules/morphine.sdf > morphine.moladt.json
./.venv/bin/python -m moladt.cli from-json morphine.moladt.json
./.venv/bin/python -m moladt.cli view-html morphine.moladt.json --format json --output morphine.viewer.html
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
```

- `parse` reads one SDF record, validates it, and prints a MolADT report plus the SDF title; add `--properties` when you want the raw SDF property fields too
- `parse-smiles` reads a supported SMILES string into the same typed molecule shape
- `to-json` reads one SDF record, validates it, and writes the shared MolADT JSON boundary format used across the Python and Haskell repos
- `from-json` reads that MolADT JSON back into the typed `Molecule` object and prints the usual MolADT report
- `view-html` exports a standalone beta 3D browser viewer with drag rotation, wheel zoom, JSON drop loading, atom picking, and colored Dietz bonding-system annotations
- `pretty-example` loads the manuscript-facing built-in objects, written as explicit typed molecules with orbital shells intact
- `to-smiles` renders validated classical MolADT structures back into the supported SMILES subset

The SDF reader accepts V2000 and the core V3000 CTAB subset used by common structure exports:

- atom coordinates
- bond tables
- atom-local formal charges
- trailing SDF property blocks

The writer still emits V2000. The parser and renderer are intentionally narrower than the full MDL feature surface.

If one SDF file contains multiple molecules:

- read a small eager slice with `from moladt.io.sdf import read_sdf_records`
- stream lazily with `from moladt.io.sdf import iter_sdf_records`

The local QM9 and vendored FreeSolv raw files in this workspace are still V2000. The downloader and parser prefer V3000 when a future dataset bundle actually provides it, but the current local raws are not being silently relabeled.

## What This Repo Contains

- the Python MolADT types, parser, renderer, and pretty-printer
- example molecules including diborane, ferrocene, and morphine
- feature generation, Stan models, and local benchmark tooling
- the shared JSON and processed-matrix contracts used by the sibling Haskell repo

The documentation is split by task:

- first run: [Quickstart](docs/quickstart.md)
- representation: [MolADT representation](docs/representation.md), [ADT model](docs/data-model.md), and [orbitals](docs/orbitals.md)
- examples and parsing: [Examples](docs/examples.md), [Parsing and rendering](docs/parsing.md), and [CLI reference](docs/cli.md)
- benchmarks and outputs: [Inference and benchmarks](docs/inference-and-benchmarks.md), [Models and features](docs/models.md), [Outputs](docs/outputs.md), and [results README](results/README.md)
- interop and data: [Haskell interop](docs/haskell_interop.md), [Data sources](docs/data-sources.md), and [Repo map](docs/repo-map.md)

## Inverse Design

The FreeSolv inverse-design experiment grows molecules toward a target hydration free energy using the checked-in FreeSolv Bayesian GP. It starts from water by default, uses fixed random seeds, validates every accepted proposal, and writes the top generated molecules as importable Python files.

Run the default water-seeded experiment with:

```bash
make inverse-design TARGET=-5.0
```

Here `-5.0` means “try to find molecules predicted near `-5.0` kcal/mol hydration free energy.” Water is the default seed molecule, so no seed argument is needed for the paper-facing run.

To start from another supported seed, for example methane:

```bash
make inverse-design TARGET=-5.0 SEED_MOLECULE=methane
```

The equivalent direct Python commands are:

```bash
python -m experiments.freesolv_inverse_design --target -5.0
python -m experiments.freesolv_inverse_design --target -5.0 --seed-molecule methane
```

`--target` and `--seed-molecule` are the only user-facing flags. If no target is supplied, the script uses the median experimental FreeSolv value from the processed dataset. Other settings are fixed constants in `experiments/freesolv_inverse_design.py` for reproducibility: `N_STEPS=2000`, `N_SEEDS=5`, `TOP_K=10`, `MAX_HEAVY_ATOMS=12`, `TEMPERATURE=1.0`, `RANDOM_SEED=0`, and `CONFIDENCE_NOISE_FLOOR=1.0`.

The predictor loads the committed FreeSolv coefficient summary at `results/freesolv/run_20260417_162536/details/model_coefficients.csv`, filtered to `freesolv / moladt_featurized / expt / bayes_gp_rbf_screened / laplace`, and the matching Laplace posterior draws at `results/freesolv/run_20260417_162536/details/stan_output/freesolv/moladt_featurized/bayes_gp_rbf_screened/laplace/`. Candidate scores use the posterior-draw averaged GP predictive mean and predictive standard deviation. Lower-uncertainty molecules are preferred through a Gaussian log score for the target, so the retained molecules are both target-matching and comparatively confident.

The generator uses small MolADT/Dietz-aware moves: add a terminal atom, close a sigma edge, mutate a non-hydrogen atom, remove a terminal atom, or add a six-electron pi system over an existing carbon six-ring. Every proposal is validated against local sigma edges plus Dietz bonding systems before it can be accepted, so delocalized and multicenter bonding are not ignored as plain graph decoration.

At the end, the run prints diagnostics and the top molecules to stdout. It writes `results/inverse_design/run_.../top_01_molecule.py` through `top_10_molecule.py` for the timestamped run and refreshes the Git-tracked reference copies in `results/inverse_design/reference/`. It also writes `dietz_01_molecule.py` through `dietz_05_molecule.py` when valid candidates with Dietz bonding systems were seen. Each file defines an importable, validated `molecule` plus the seed molecule, fixed random seed, candidate rank, predicted FreeSolv value, target error, score, and formula.

This is a proof of concept for property-conditioned MolADT inverse design on FreeSolv, not a synthesizability filter or a state-of-the-art molecular generator.

## Benchmarking

FreeSolv has a committed reference bundle so the fitted GP summary, posterior draws, predictions, and processed feature matrices are available in the repo. QM9 and timing keep their paper-facing graphs and captions in Git, while their heavier run details stay local.

```bash
make freesolv
make inverse-design TARGET=-5.0
make inverse-design TARGET=-5.0 SEED_MOLECULE=methane
make qm9long
make timing
```

- `make freesolv`: FreeSolv RMSE comparison. Fixed path `moladt_featurized + bayes_gp_rbf_screened + laplace`. The paper SVG is a clean bar comparison, and the prose caption is written separately to `caption.txt`. It writes `results/freesolv/run_.../freesolv_rmse_vs_moleculenet.svg`, `results/freesolv/run_.../caption.txt`, `results/freesolv/run_.../freesolv_bayesian_model.txt`, and `results/freesolv/run_.../details/freesolv_train_test_uncertainty.csv`.
- `make inverse-design TARGET=-5.0`: FreeSolv inverse-design proof of concept. It reuses the fixed FreeSolv Bayesian GP artifact, starts deterministic search chains from water by default, applies MolADT/Dietz-aware local growth moves, validates every accepted molecule, prints the top generated molecules plus the best Dietz-system molecules to stdout, and writes importable Python files under `results/inverse_design/run_.../` and `results/inverse_design/reference/`. If `TARGET` is omitted, the script uses the median experimental FreeSolv value. Set `SEED_MOLECULE=methane` to start all chains from methane instead.
- `make qm9long`: full QM9 `mu` MAE comparison over all aligned local QM9 molecules, using `visnet_ensemble` on `moladt_featurized_geom`. That export keeps the atomic numbers and coordinates from the SDF record and adds the full MolADT feature bundle from the same molecule. The current local bundle yields `107,108 / 13,388 / 13,389` train / validation / test rows under the deterministic `80/10/10` long split. ViSNet runs one member for at most `25` epochs with seed `102`, and the verbose run prints every epoch with validation RMSE and MAE. It writes the clean paper SVG plus `caption.txt`.
- `make timing`: ZINC timing comparison on the fixed eight-stage paper path: `SMILES CSV -> string`, `MolADT CSV -> MolADT`, `SMILES -> JSON`, `SDF -> MolADT`, `SDF -> SMILES`, `MolADT -> JSON`, `JSON -> MolADT`, and `JSON -> SMILES`. It writes `results/timing/paper/run_.../timing_overview.svg`, `results/timing/paper/run_.../caption.txt`, and `results/timing/paper/run_.../timing_result_files.txt`.

Results are written under timestamped directories in `results/`, mainly `results/freesolv/run_.../`, `results/qm9/long/run_.../`, and `results/timing/paper/run_.../`. The committed artifacts are the FreeSolv reference bundle plus the QM9/timing SVG graphs and captions.

The FreeSolv figure shows `Training`, `Validation`, `Test`, and `Paper`. The QM9 figure shows `Training`, `Test`, and `Paper`.

For split sizes, the exact benchmark contract, the published FreeSolv RMSE context table, and the detailed timing-stage definitions, see [Inference and benchmarks](docs/inference-and-benchmarks.md), [Outputs](docs/outputs.md), and the checked-in [results README](results/README.md).

## Read More

- [Quickstart](docs/quickstart.md)
- [Docs index](docs/README.md)
- [ADT representation](docs/data-model.md)
- [Inference and benchmarks](docs/inference-and-benchmarks.md)
- [Parsing and rendering](docs/parsing.md)
- [CLI reference](docs/cli.md)
- [Examples](docs/examples.md)
- [Models and features](docs/models.md)
- [Outputs](docs/outputs.md)
- [Data sources](docs/data-sources.md)
- [Results README](results/README.md)
- [Haskell interop](docs/haskell_interop.md)

## Related Repo

- [MolADT-Bayes-Haskell](https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell)
