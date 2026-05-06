# MolADT-Bayes-Python

MolADT represents molecules as typed data for Bayesian modelling, feature generation, inverse design, validation, and viewing.

The core object is not just a string and not just a graph. It keeps atoms, coordinates, bonding systems, formal charge, and optional shell/orbital data in explicit fields that can be inspected, mutated, scored, serialized, and shared with the Haskell repo. The edge set is derived from bonding systems and kept as a compatibility index.

[Quickstart](docs/quickstart.md) · [Representation](docs/representation.md) · [Examples](docs/examples.md) · [Equality](docs/molecule-equality.md) · [CLI](docs/cli.md) · [Models](docs/models.md) · [Benchmarks](docs/inference-and-benchmarks.md) · [Outputs](docs/outputs.md)

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
    local_bonds: frozenset[Edge] = frozenset()
    systems: tuple[tuple[SystemId, BondingSystem], ...] = ()
```

The canonical bonding layer is `systems`: every edge is an instance of a
`BondingSystem`. A conventional single, double, or triple bond is a one-edge
system with `2`, `4`, or `6` shared electrons respectively, tagged `single`,
`double`, or `triple`. `local_bonds` is retained for older callers and graph
traversal; constructing a molecule from bare edges automatically lifts them
into two-electron `single` systems.

Pretty printers derive their edge rows from the bonding systems. They show the
total electrons shared over each edge, then the effective order. For example, a
benzene C-C edge displays as `shared=3e` and `order=1.50`: `2e` from its
one-edge `single` system plus `1e/edge` from the six-electron `pi_ring`
system. The viewer lists the same explicit bonding systems.

Use [`same_molecule`](docs/molecule-equality.md) when you want equality modulo
container ordering, such as atom maps, edge sets, system tuples, and annotation
tuples. It keeps atom and system identifiers meaningful.

An atom carries element data, position, formal charge, and shell data:

```python
@dataclass(frozen=True, slots=True)
class Atom:
    atom_id: AtomId
    attributes: ElementAttributes
    coordinate: Coordinate
    shells: Shells | None = None
    formal_charge: int = 0
```

`ElementAttributes` also carries the default shell data, so simple atom
builders can use `element_attributes(symbol)` and omit `shells`. The older
`element_shells(symbol)` helper remains as a compatibility wrapper.

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
| Benzene | one-edge `single` systems on each edge plus a six-electron `pi_ring`; each C-C edge displays as `shared=3e` |
| Diborane | terminal `single` systems plus two explicit `3c-2e` bridge systems |
| Ferrocene | one-edge Cp/C-H systems plus two Cp pi systems and an Fe/Cp electron pool |
| Morphine | every graph edge as a system plus named delocalised systems |

Ferrocene is a good example to look at because the representation keeps the metallocene structure explicit instead of flattening it into a string. This is the fully expanded ADT: no hidden loops, no placeholder edge sets.

```python
from __future__ import annotations

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import AtomicSymbol, Molecule
from ._literal import atom


ferrocene_pretty = Molecule(
    atoms={
        AtomId(1): atom(1, AtomicSymbol.Fe, 0.000, 0.000, 0.000),
        AtomId(2): atom(2, AtomicSymbol.C, 1.194, 0.000, 1.660),
        AtomId(3): atom(3, AtomicSymbol.C, 0.368966, 1.135561, 1.660),
        AtomId(4): atom(4, AtomicSymbol.C, -0.965966, 0.701816, 1.660),
        AtomId(5): atom(5, AtomicSymbol.C, -0.965966, -0.701816, 1.660),
        AtomId(6): atom(6, AtomicSymbol.C, 0.368966, -1.135561, 1.660),
        AtomId(7): atom(7, AtomicSymbol.C, 0.965966, 0.701816, -1.660),
        AtomId(8): atom(8, AtomicSymbol.C, -0.368966, 1.135561, -1.660),
        AtomId(9): atom(9, AtomicSymbol.C, -1.194, 0.000, -1.660),
        AtomId(10): atom(10, AtomicSymbol.C, -0.368966, -1.135561, -1.660),
        AtomId(11): atom(11, AtomicSymbol.C, 0.965966, -0.701816, -1.660),
        AtomId(12): atom(12, AtomicSymbol.H, 2.280, 0.000, 1.565),
        AtomId(13): atom(13, AtomicSymbol.H, 0.704559, 2.168409, 1.565),
        AtomId(14): atom(14, AtomicSymbol.H, -1.844559, 1.340150, 1.565),
        AtomId(15): atom(15, AtomicSymbol.H, -1.844559, -1.340150, 1.565),
        AtomId(16): atom(16, AtomicSymbol.H, 0.704559, -2.168409, 1.565),
        AtomId(17): atom(17, AtomicSymbol.H, 1.844559, 1.340150, -1.565),
        AtomId(18): atom(18, AtomicSymbol.H, -0.704559, 2.168409, -1.565),
        AtomId(19): atom(19, AtomicSymbol.H, -2.280, 0.000, -1.565),
        AtomId(20): atom(20, AtomicSymbol.H, -0.704559, -2.168409, -1.565),
        AtomId(21): atom(21, AtomicSymbol.H, 1.844559, -1.340150, -1.565),
    },
    local_bonds=frozenset(
        {
            Edge(AtomId(2), AtomId(3)),
            Edge(AtomId(2), AtomId(6)),
            Edge(AtomId(2), AtomId(12)),
            Edge(AtomId(3), AtomId(4)),
            Edge(AtomId(3), AtomId(13)),
            Edge(AtomId(4), AtomId(5)),
            Edge(AtomId(4), AtomId(14)),
            Edge(AtomId(5), AtomId(6)),
            Edge(AtomId(5), AtomId(15)),
            Edge(AtomId(6), AtomId(16)),
            Edge(AtomId(7), AtomId(8)),
            Edge(AtomId(7), AtomId(11)),
            Edge(AtomId(7), AtomId(17)),
            Edge(AtomId(8), AtomId(9)),
            Edge(AtomId(8), AtomId(18)),
            Edge(AtomId(9), AtomId(10)),
            Edge(AtomId(9), AtomId(19)),
            Edge(AtomId(10), AtomId(11)),
            Edge(AtomId(10), AtomId(20)),
            Edge(AtomId(11), AtomId(21)),
        }
    ),
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(6),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(2)),
                        Edge(AtomId(1), AtomId(3)),
                        Edge(AtomId(1), AtomId(4)),
                        Edge(AtomId(1), AtomId(5)),
                        Edge(AtomId(1), AtomId(6)),
                        Edge(AtomId(2), AtomId(3)),
                        Edge(AtomId(2), AtomId(6)),
                        Edge(AtomId(3), AtomId(4)),
                        Edge(AtomId(4), AtomId(5)),
                        Edge(AtomId(5), AtomId(6)),
                    }
                ),
                "cp1_pi",
            ),
        ),
        (
            SystemId(2),
            mk_bonding_system(
                NonNegative(6),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(7)),
                        Edge(AtomId(1), AtomId(8)),
                        Edge(AtomId(1), AtomId(9)),
                        Edge(AtomId(1), AtomId(10)),
                        Edge(AtomId(1), AtomId(11)),
                        Edge(AtomId(7), AtomId(8)),
                        Edge(AtomId(7), AtomId(11)),
                        Edge(AtomId(8), AtomId(9)),
                        Edge(AtomId(9), AtomId(10)),
                        Edge(AtomId(10), AtomId(11)),
                    }
                ),
                "cp2_pi",
            ),
        ),
        (
            SystemId(3),
            mk_bonding_system(
                NonNegative(6),
                frozenset(
                    {
                        Edge(AtomId(1), AtomId(2)),
                        Edge(AtomId(1), AtomId(3)),
                        Edge(AtomId(1), AtomId(4)),
                        Edge(AtomId(1), AtomId(5)),
                        Edge(AtomId(1), AtomId(6)),
                        Edge(AtomId(1), AtomId(7)),
                        Edge(AtomId(1), AtomId(8)),
                        Edge(AtomId(1), AtomId(9)),
                        Edge(AtomId(1), AtomId(10)),
                        Edge(AtomId(1), AtomId(11)),
                    }
                ),
                "fe_backdonation",
            ),
        ),
    ),
)
```

See the full expanded ADT in [`moladt/examples/ferrocene.py`](moladt/examples/ferrocene.py).

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

`make view` opens six built-in examples in one browser page. Click an atom to see stored shell/orbital glyphs, coordinates, 3D edge lengths, effective orders, bonding systems, and bond angles from the molecule coordinates.

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
