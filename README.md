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

More examples: [Examples](docs/examples.md), [Parsing and rendering](docs/parsing.md), and [CLI reference](docs/cli.md).

## Use The ADT

```python
from moladt import MutableMolecule
from moladt.chem.validate import validate_molecule
from moladt.examples import ferrocene_pretty

candidate = MutableMolecule.from_molecule(ferrocene_pretty)
candidate.systems = [
    (system_id, system)
    for system_id, system in candidate.systems
    if system.tag != "fe_backdonation"
]

molecule = validate_molecule(candidate.freeze())
print(len(molecule.atoms), len(molecule.local_bonds), len(molecule.systems))
```

`Molecule` is immutable. `MutableMolecule` is the scratchpad for proposal moves. That split keeps generated candidates easy to edit and easy to validate.

See [ADT model](docs/data-model.md) for the field layout and [Examples](docs/examples.md) for diborane, ferrocene, morphine, benzene, and file-backed molecules.

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
