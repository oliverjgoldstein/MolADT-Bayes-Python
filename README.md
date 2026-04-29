# MolADT-Bayes-Python

Design molecules as data.

MolADT is a typed molecule representation for Bayesian modelling, feature generation, validation, and inverse design. It keeps the chemistry in the object: atoms, coordinates, sigma bonds, delocalised or multicentre bonding systems, formal charges, and stereochemistry all live as structured fields instead of being compressed into a string.

Not just a SMILES string. Not just a graph. A molecule object you can inspect, mutate, validate, score, serialize, and share.

[Quickstart](docs/quickstart.md) · [Representation](docs/representation.md) · [ADT model](docs/data-model.md) · [Examples](docs/examples.md) · [CLI](docs/cli.md) · [Models](docs/models.md) · [Benchmarks](docs/inference-and-benchmarks.md) · [Outputs](docs/outputs.md)

## Why MolADT

Molecular ML pipelines often begin with a compact boundary format, then add chemistry back later. MolADT starts with the chemistry.

| What you need | What MolADT gives you |
| --- | --- |
| Explicit structure | `atoms`, `local_bonds`, `systems`, and `smiles_stereochemistry` as first-class fields |
| Better proposal moves | edit molecules directly with `MutableMolecule`, then freeze and validate |
| Unusual bonding | diborane `3c-2e` bridges, ferrocene Cp/metal systems, aromatic pi systems |
| Safe scoring | validate valence, connectivity, bonding systems, and FreeSolv-specific candidate rules before prediction |
| Interop | JSON payloads and processed matrices shared with the Haskell repo |
| Benchmarks | FreeSolv, QM9, ZINC timing, and committed result artifacts |

Read the deeper case for the representation in [MolADT representation](docs/representation.md), [ADT model](docs/data-model.md), and [orbitals](docs/orbitals.md).

## Bayesian Generative Tasks

Bayesian molecular tasks need a clean support: the model should know what kind of object it is allowed to put probability mass on. MolADT gives that support as an explicit typed generator, not as an after-the-fact string filter.

That matters because proposal moves, priors, validators, feature maps, posterior predictive scores, and exported candidates all talk about the same object. A FreeSolv inverse-design run can grow a molecule, enforce neutral closed-valence CHONFCl chemistry, score it with a Bayesian uncertainty model, and write the exact generated structure back to disk without changing representation halfway through the pipeline.

For Bayesian work, that gives you:

| Advantage | Why it helps |
| --- | --- |
| Typed support | priors and proposal kernels operate on valid molecule fields instead of loosely parsed text |
| Explicit constraints | chemistry rules can be part of generation, not just reject-sampling after scoring |
| Inspectable uncertainty | posterior scores stay attached to the molecule, formula, bonds, and bonding systems that produced them |
| Reproducible artifacts | generated candidates can be validated, serialized, imported, and benchmarked again |
| Generality | the same ADT supports parsing, featurization, timing, FreeSolv, QM9, and inverse design |

See [Models and features](docs/models.md), [Inference and benchmarks](docs/inference-and-benchmarks.md), and [Outputs](docs/outputs.md) for the Bayesian tasks built on top of the representation.

## Start

```bash
make python-setup
make python-parse
```

For Stan-backed FreeSolv runs:

```bash
make python-cmdstan-install
```

`make python-setup` creates `./.venv` inside this repo. `make python-cmdstan-install` creates `./.cmdstan`. Both are local to the checkout.

Full setup notes live in [Quickstart](docs/quickstart.md).

## See A Molecule

```bash
make python-pretty-example EXAMPLE=morphine
make molecule-viewer VIEWER_INPUT=molecules/benzene.sdf
make test-molecule-viewer
```

The viewer exports a standalone HTML file under `results/viewer/` by default. Use `VIEWER_OUTPUT`, `VIEWER_FORMAT`, and `VIEWER_TITLE` when you need a custom export.

`make test-molecule-viewer` now runs the viewer tests, writes the configured viewer HTML, and opens it in your default browser automatically.

Turn on browser auto-open with `OPEN_VIEWER=1`:

```bash
OPEN_VIEWER=1 make python-pretty-example EXAMPLE=ferrocene
OPEN_VIEWER=1 make molecule-viewer VIEWER_INPUT=molecules/diborane.sdf
OPEN_VIEWER=1 VIEWER_COUNT=3 make inverse-design TARGET=-5.0
```

The first command still prints the pretty molecule report, then writes and opens `results/viewer/ferrocene.viewer.html`. The inverse-design command writes viewer files for the top generated molecules under `results/inverse_design/run_.../` and opens the first `VIEWER_COUNT` of them.

Direct CLI equivalents:

```bash
./.venv/bin/python -m moladt.cli pretty-example diborane --open-viewer
./.venv/bin/python -m moladt.cli view-html molecules/benzene.sdf --output results/viewer/benzene.viewer.html --open-viewer
./.venv/bin/python -m experiments.freesolv_inverse_design --target -5.0 --open-viewer --viewer-count 3
```

More examples: [Examples](docs/examples.md), [Parsing and rendering](docs/parsing.md), and [CLI reference](docs/cli.md).

## Pretty Representation Examples

The built-in examples are written as explicit typed molecules. They are intentionally not just display fixtures: the same values can be validated, serialized, opened in the viewer, featurized, and used as proposal states.

### Morphine

```bash
./.venv/bin/python -m moladt.cli pretty-example morphine
```

The morphine example keeps the fused sigma skeleton, an `alkene_bridge`, a `phenyl_pi_ring`, and the atom-centered stereochemistry flags from the standard boundary string:

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

### Diborane And Ferrocene

```bash
./.venv/bin/python -m moladt.cli pretty-example diborane
./.venv/bin/python -m moladt.cli pretty-example ferrocene
```

Diborane has two explicit `3c-2e` bridging hydrogen systems. Ferrocene has two cyclopentadienyl `pi` systems plus an Fe back-donation-style pool. Those systems are separate typed objects, so a proposal can preserve, remove, replace, or score them directly.

```text
Diborane (B2H6)
Dietz-style ADT with two explicit 3c-2e bridging hydrogen bonding systems.

Molecule Report
===============
atoms            8
heavy atoms      2
sigma bonds      5
bonding systems  2
net charge       +0
composition      B2 H6

Bonding Systems
---------------
[#1] bridge_h3_3c2e
  shared electrons: 2
  member atoms:     B#1, B#2, H#3
  edge bonus:       +0.50 to each listed edge

[#2] bridge_h4_3c2e
  shared electrons: 2
  member atoms:     B#1, B#2, H#4
  edge bonus:       +0.50 to each listed edge
```

## Use The ADT

```python
from moladt import MutableMolecule
from moladt.chem.validate import validate_molecule
from moladt.examples import ferrocene_pretty
from moladt.io import molecule_from_json, molecule_to_json

candidate = MutableMolecule.from_molecule(ferrocene_pretty)
candidate.systems = [
    (system_id, system)
    for system_id, system in candidate.systems
    if system.tag != "fe_backdonation"
]

molecule = validate_molecule(candidate.freeze())
print(len(molecule.atoms), len(molecule.local_bonds), len(molecule.systems))

payload = molecule_to_json(molecule)
round_tripped = validate_molecule(molecule_from_json(payload))
assert round_tripped == molecule
```

`Molecule` is immutable. `MutableMolecule` is the scratchpad for proposal moves. That split keeps generated candidates easy to edit and easy to validate.

See [ADT model](docs/data-model.md) for the field layout and [Examples](docs/examples.md) for diborane, ferrocene, morphine, benzene, and file-backed molecules.

## Python And Haskell

MolADT JSON is the boundary format shared by this Python repo and the sibling Haskell repo. Python can generate, mutate, featurize, and benchmark candidates while Haskell can consume the same structured molecule.

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

Python also writes standardized benchmark matrices that Haskell can consume without re-running Python feature extraction:

```bash
./.venv/bin/python -m scripts.run_all freesolv
MOLADT_PROCESSED_DATA_DIR=../MolADT-Bayes-Python/data/processed \
  stack run moladtbayes -- infer-benchmark freesolv_moladt_featurized lwis
```

## Benchmarks

```bash
make freesolv
make inverse-design TARGET=-5.0
make qm9long
make timing
```

| Workflow | Command | What it produces |
| --- | --- | --- |
| FreeSolv benchmark | `make freesolv` | FreeSolv RMSE comparison for the fixed `moladt_featurized + bayes_gp_rbf_screened + laplace` path |
| FreeSolv inverse design | `make inverse-design TARGET=-5.0` | At least 1,000 unique valid generated candidates, with the top 10 chosen by the model's Bayesian credible score percentage |
| QM9 benchmark | `make qm9long` | Full-data QM9 `mu` comparison using `visnet_ensemble` on `moladt_featurized_geom` |
| Timing benchmark | `make timing` | ZINC representation timing comparison across SMILES, SDF, JSON, and MolADT paths |

FreeSolv should usually be read as a pair:

```bash
make freesolv
make inverse-design TARGET=-5.0
```

The first command checks the predictive model on the FreeSolv split. The second command uses the latest `results/freesolv/run_*` FreeSolv GP artifact to search for molecules near the requested hydration free energy. If the newest FreeSolv run is incomplete, inverse design fails instead of falling back to an older model. The inverse-design run writes top generated molecule files under `results/inverse_design/run_.../` and refreshes the reference files under `results/inverse_design/reference/`.

The FreeSolv generator uses MolADT/Dietz-aware edits whose move rules choose only locally feasible atoms, elements, and rings. Candidates are then checked by the repo validator plus FreeSolv-specific chemistry invariants before scoring: connected, neutral, CHONFCl-only, closed-valence, and sound bonding systems. That enforces the relevant graph and electron-count rules for this generator; it is not a substitute for quantum relaxation, thermodynamic stability analysis, or a synthesizability filter.

Benchmark details are in [Inference and benchmarks](docs/inference-and-benchmarks.md), [Models and features](docs/models.md), [Outputs](docs/outputs.md), and [results README](results/README.md).

## Modules

| Area | Start here |
| --- | --- |
| First run | [Quickstart](docs/quickstart.md) |
| Representation | [MolADT representation](docs/representation.md), [ADT model](docs/data-model.md), [orbitals](docs/orbitals.md) |
| Examples | [Examples](docs/examples.md) |
| Parsing and rendering | [Parsing and rendering](docs/parsing.md), [SMILES scope](docs/smiles-scope-and-validation.md), [CLI](docs/cli.md) |
| Models and features | [Models and features](docs/models.md) |
| Benchmarks and artifacts | [Inference and benchmarks](docs/inference-and-benchmarks.md), [Outputs](docs/outputs.md), [results README](results/README.md) |
| Data | [Data sources](docs/data-sources.md) |
| Interop | [Haskell interop](docs/haskell_interop.md) |
| Repo tour | [Repo map](docs/repo-map.md) |

## What Lives Here

- `moladt/`: typed molecule objects, validation, parsers, renderers, viewer, examples
- `experiments/`: FreeSolv inverse design
- `scripts/`: data processing, feature generation, model runs, reporting
- `stan/`: Bayesian model definitions
- `data/`: vendored and processed benchmark data
- `results/`: committed reference outputs and local run artifacts
- `docs/`: modular documentation

## Related

- [MolADT-Bayes-Haskell](https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell)
