# MolADT-Bayes-Python

MolADT represents molecules as typed data for Bayesian modelling, feature generation, inverse design, validation, and viewing.

The core object is not just a string and not just a graph. It keeps atoms, coordinates, local bonds, electron-sharing systems, formal charge, and shell/orbital data in explicit fields that can be inspected, mutated, scored, serialized, and shared with the Haskell repo.

[Quickstart](docs/quickstart.md) · [Representation](docs/representation.md) · [Examples](docs/examples.md) · [CLI](docs/cli.md) · [Models](docs/models.md) · [Benchmarks](docs/inference-and-benchmarks.md) · [Outputs](docs/outputs.md)

## What It Does

| Task | Command | Output |
| --- | --- | --- |
| View examples | `make view` | Browser viewer for built-in MolADT examples |
| FreeSolv benchmark | `make freesolv` | Bayesian GP benchmark for hydration free energy |
| FreeSolv inverse design | `make inverse-design TARGET=-5.0` | 1,000 generated candidates, ranked by Bayesian credible score |
| QM9 benchmark | `make qm9long` | QM9 dipole moment `mu` benchmark using geometry features |
| Timing benchmark | `make timing` | ZINC representation timing comparison |

## Start

```bash
make python-setup
make python-parse
make view
```

For Stan-backed FreeSolv runs:

```bash
make python-cmdstan-install
make freesolv
make inverse-design TARGET=-5.0
```

`make python-setup` creates `./.venv`. `make python-cmdstan-install` creates `./.cmdstan`. Both stay inside this checkout.

## Why The ADT Matters

MolADT gives Bayesian molecular work a typed support: priors, proposal kernels, validators, feature maps, posterior predictive scores, and exported candidates all talk about the same molecule object.

That matters for inverse design. The FreeSolv task can start from a FreeSolv-derived molecule prior, grow typed candidates, score each molecule with the unchanged GP posterior predictive distribution, and write the exact generated ADT back to disk.

The FreeSolv model is posterior predictive: it gives a distribution over hydration free energy for a given molecule. It is not itself a molecule-generating prior.

## The Representation

The core molecule shape mirrors the sibling Haskell repo:

```haskell
data Molecule = Molecule
  { atoms      :: Map AtomId Atom
  , localBonds :: Set Edge
  , systems    :: [(SystemId, BondingSystem)]
  }
```

```python
@dataclass(frozen=True, slots=True)
class Molecule:
    atoms: Mapping[AtomId, Atom]
    local_bonds: frozenset[Edge]
    systems: tuple[tuple[SystemId, BondingSystem], ...]
```

An atom carries element data, position, formal charge, and shell data:

```python
@dataclass(frozen=True, slots=True)
class Atom:
    atom_id: AtomId
    attributes: ElementAttributes
    coordinate: Coordinate
    shells: Shells
    formal_charge: int = 0
```

Delocalised and multicentre bonding is represented explicitly:

```python
@dataclass(frozen=True, slots=True)
class BondingSystem:
    shared_electrons: NonNegative
    member_atoms: frozenset[AtomId]
    member_edges: frozenset[Edge]
    tag: str | None = None
```

Examples:

| Molecule | What MolADT stores |
| --- | --- |
| Benzene | local ring bonds plus a six-electron `pi_ring` system |
| Diborane | two explicit `3c-2e` bridge systems |
| Ferrocene | two Cp pi systems plus an Fe/Cp electron pool |
| Morphine | fused sigma skeleton plus named delocalised systems |

## Orbitals

Shells and orbitals are typed too. A heavier atom such as iodine is represented by occupancy data rather than an opaque label:

```python
IODINE: Shells = (
    _shell(1, s_counts=(2,)),
    _shell(2, s_counts=(2,), p_counts=(2, 2, 2)),
    _shell(3, s_counts=(2,), p_counts=(2, 2, 2), d_counts=(2, 2, 2, 2, 2)),
    _shell(4, s_counts=(2,), p_counts=(2, 2, 2), d_counts=(2, 2, 2, 2, 2)),
    _shell(5, s_counts=(2,), p_counts=(2, 2, 1)),
)
```

See [Orbitals](docs/orbitals.md) for the fuller model.

## View Molecules

```bash
make view
make molecule-viewer VIEWER_EXAMPLES="benzene diborane ferrocene"
make python-pretty-example EXAMPLE=morphine
```

`make view` opens six built-in examples in one browser page. Click an atom to see stored shell/orbital glyphs, coordinates, 3D edge lengths, and bond angles from the molecule coordinates.

Use `OPEN_VIEWER=1` to open generated viewer pages automatically:

```bash
OPEN_VIEWER=1 make molecule-viewer VIEWER_EXAMPLES=diborane
OPEN_VIEWER=1 make inverse-design TARGET=-5.0
```

## FreeSolv Inverse Design

Run FreeSolv and inverse design together:

```bash
make freesolv
make inverse-design TARGET=-5.0
make inverse-design-view
```

`make inverse-design` uses the latest `results/freesolv/run_*` Bayesian GP artifact. It samples initial chains from the MolADT FreeSolv prior, reweights that prior with the unchanged GP target likelihood, generates 1,000 candidates, and writes the top 10 by Bayesian credible score.

Geometry values are audited in the outputs. The sampler is not conditioned on physical plausibility.

Generated reference outputs live in `results/inverse_design/reference/`.

## Benchmarks

| Benchmark | Target | Main command |
| --- | --- | --- |
| FreeSolv | hydration free energy | `make freesolv` |
| FreeSolv inverse design | target hydration free energy | `make inverse-design TARGET=-5.0` |
| QM9 | dipole moment `mu` | `make qm9long` |
| ZINC timing | representation throughput | `make timing` |

Benchmark details are in [Inference and benchmarks](docs/inference-and-benchmarks.md), [Models and features](docs/models.md), [Outputs](docs/outputs.md), and [results README](results/README.md).

## Python And Haskell

MolADT JSON is the boundary shared by the Python and Haskell repos.

```bash
./.venv/bin/python - <<'PY' > morphine.moladt.json
from moladt.examples import morphine_pretty
from moladt.io import molecule_to_json

print(molecule_to_json(morphine_pretty))
PY

stack run moladtbayes -- from-json ../MolADT-Bayes-Python/morphine.moladt.json
```

Python also writes standardized benchmark matrices that Haskell can consume:

```bash
./.venv/bin/python -m scripts.run_all freesolv
MOLADT_PROCESSED_DATA_DIR=../MolADT-Bayes-Python/data/processed \
  stack run moladtbayes -- infer-benchmark freesolv_moladt_featurized lwis
```

## Repo Map

| Path | Purpose |
| --- | --- |
| `moladt/` | molecule ADT, validation, parsers, renderers, viewer, examples |
| `experiments/` | FreeSolv inverse design |
| `scripts/` | data processing, feature generation, model runs, reporting |
| `stan/` | Bayesian model definitions |
| `data/` | vendored and processed benchmark data |
| `results/` | committed reference outputs and local run artifacts |
| `docs/` | modular documentation |
