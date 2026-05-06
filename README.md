# MolADT-Bayes-Python

MolADT represents molecules as typed data for Bayesian modelling, feature generation, inverse design, validation, and viewing.

The core object is not just a string and not just a graph. It keeps atoms, coordinates, bonding systems, formal charge, and optional shell/orbital data in explicit fields that can be inspected, mutated, scored, serialized, and shared with the Haskell repo. The edge set is derived from bonding-system member edges.

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
  , systems    :: [(SystemId, BondingSystem)]
  , smilesStereochemistry :: SmilesStereochemistry
  }
```

```python
@dataclass(frozen=True, slots=True)
class Molecule:
    atoms: Mapping[AtomId, Atom]
    systems: tuple[tuple[SystemId, BondingSystem], ...] = ()
    smiles_stereochemistry: SmilesStereochemistry = field(default_factory=SmilesStereochemistry)
```

The canonical bonding layer is `systems`: every edge is an instance of a
`BondingSystem`. A conventional single, double, triple, or quadruple bond is a
one-edge system with `2`, `4`, `6`, or `8` shared electrons respectively.
Pretty printers and viewers display those as `single covalent`,
`double covalent`, `triple covalent`, or `quadruple covalent`.
An ionic contact is also a one-edge `BondingSystem`, but it shares `0`
electrons, carries tag `ionic`, and keeps the formal charge on the atoms.
For example, sodium chloride stores `Na#1` as `+1`, `Cl#2` as `-1`, and the
Na-Cl edge as one `ionic` system.

Pretty printers derive their edge rows from the bonding systems. They show the
total electrons shared over each edge, then the effective order. For example, a
benzene C-C edge displays as `shared=3e` and `order=1.50`: `2e` from its
unnamed one-edge system plus `1e/edge` from the six-electron `pi_ring`
system. The viewer lists the same explicit bonding systems.

System identifiers are stable display IDs, not chemistry. Checked examples and
parsers put named or multi-edge systems first, so benzene uses `SystemId(1)` for
`pi_ring` and then numbers the ordinary one-edge covalent systems after it.

Use [`same_molecule`](docs/molecule-equality.md) when you want equality modulo
container ordering, such as atom maps, system tuples, member-edge sets, and
annotation tuples. It keeps atom and system identifiers meaningful.

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
builders can use `element_attributes(symbol)` and omit `shells`; there is no
separate shell lookup layer.

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
| Benzene | `single covalent` one-edge systems on each edge plus a six-electron `pi_ring`; each C-C edge displays as `shared=3e` |
| Diborane | four terminal B-H `single covalent` systems plus two explicit `3c-2e` bridge systems; no direct B-B singleton |
| Ferrocene | Cp/C-H `single covalent` systems plus two Cp pi systems and one Fe-Cp coordination system; Fe is `+2` and one representative carbon on each Cp ring is `-1` |
| Sodium chloride | `Na+` and `Cl-` atoms plus one zero-electron `ionic` edge system |
| Morphine | every graph edge as a system plus named delocalised systems |

Ferrocene is a useful example because the metallocene structure is explicit
without flattening it into a string. The Cp pi systems only span Cp ring C-C
edges. The Fe interaction is one coordination system over the ten Fe-C contacts.
The ordinary Cp/C-H and Cp/Cp covalent edges are still present as unnamed
one-edge systems that display as `single covalent`.

| System | Shared electrons | Member edges |
| --- | --- | --- |
| `cp1_pi` | `6e` | the five C-C edges in the first Cp ring |
| `cp2_pi` | `6e` | the five C-C edges in the second Cp ring |
| `fe_cp_coordination` | `12e` | the ten Fe-C contacts |

See the full expanded ADT in [`moladt/examples/ferrocene.py`](moladt/examples/ferrocene.py).

## View Molecules

```bash
make view
make molecule-viewer VIEWER_EXAMPLES="benzene diborane ferrocene"
make python-pretty-example EXAMPLE=morphine
```

`make view` opens seven built-in examples in one browser page, including sodium chloride's charged 0e ionic edge. Charge appears as blue and red halos around positive and negative atoms; atoms participating in an ionic bonding system get larger, more opaque charge gradients, and ionic edges draw a blue-to-red gradient between those charged atoms. Ordinary covalent edges are dark grey: single bonds draw one line, double bonds draw two lines around the edge axis, triple bonds draw three, and quadruple bonds draw four. Non-standard systems such as pi rings, bridge bonds, and coordination keep a separate coloured overlay; ordinary covalent systems are not repeated as page labels. Click an atom to see coordinates, 3D edge lengths, effective orders, bonding systems, and bond angles from the molecule coordinates.

Use `OPEN_VIEWER=1` to open generated viewer pages automatically. Viewer commands
also print a portable `file://` URL, so if the operating system does not open a
browser you can open the reported URL manually.

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
