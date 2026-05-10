# MolADT-Bayes-Python

MolADT represents molecules as typed data for Bayesian modelling, feature
generation, inverse design, validation, and viewing.

The core object is not just a string and not just a graph. It keeps:

- atoms
- coordinates
- bonding systems
- formal charge
- optional shell/orbital data
- an edge set derived from bonding-system member edges

Those fields can be inspected, mutated, scored, serialized, and shared with the
Haskell repo.

<p align="center">
  <img src="docs/assets/ferrocene.png" alt="Ferrocene in the MolADT viewer" width="280">
  <img src="docs/assets/diborane.png" alt="Diborane in the MolADT viewer" width="280">
</p>

[Quickstart](docs/quickstart.md) · [Representation](docs/representation.md) · [Parsing](docs/parsing.md) · [Validator](#validator) · [Examples](docs/examples.md) · [Equality](docs/molecule-equality.md) · [CLI](docs/cli.md) · [Models](docs/models.md) · [Benchmarks](docs/inference-and-benchmarks.md) · [Outputs](docs/outputs.md)

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

For the default FreeSolv model and inverse-design loop:

```bash
make freesolv
make inverse-design TARGET=-5.0
```

Setup artifacts stay inside this checkout:

- `make python-setup` creates `./.venv`
- `make python-cmdstan-install` creates `./.cmdstan` only when you explicitly run legacy Stan model overrides

## Why The ADT Matters

MolADT gives Bayesian molecular work a typed support where the same molecule
object is used by:

- priors
- proposal kernels
- validators
- feature maps
- posterior predictive scores
- exported candidates

That matters for inverse design because the FreeSolv task can:

- start from a FreeSolv-derived molecule prior
- grow typed candidates
- score each molecule with the unchanged GP posterior predictive distribution
- write the exact generated ADT back to disk

The FreeSolv model is posterior predictive:

- it gives a distribution over hydration free energy for a given molecule
- it is not itself a molecule-generating prior

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

The canonical Dietz bonding layer is `systems`:

- every edge is an instance of a `BondingSystem`
- a single covalent bond is a one-edge `2e` system
- a double covalent bond is a one-edge `4e` system
- a triple covalent bond is a one-edge `6e` system
- a quadruple covalent bond is a one-edge `8e` system
- an ionic contact is a one-edge `0e` system tagged `ionic`
- formal charge stays on atoms rather than on the edge

For example, sodium chloride stores:

- `Na#1` as `+1`
- `Cl#2` as `-1`
- the Na-Cl edge as one `ionic` system

Pretty printers and viewers derive display from the bonding systems:

- ordinary `2e`, `4e`, `6e`, and `8e` one-edge systems display as
  `single covalent`, `double covalent`, `triple covalent`, and
  `quadruple covalent`
- edge rows show the total electrons shared over each edge
- edge rows also show the effective order
- a benzene C-C edge displays as `shared=3e` and `order=1.50`
- that benzene value is `2e` from the unnamed one-edge system plus `1e/edge`
  from the six-electron `pi_ring`

System identifiers are stable display IDs, not chemistry:

- checked examples and parsers put named or multi-edge systems first
- benzene uses `SystemId(1)` for `pi_ring`
- ordinary one-edge covalent systems are then numbered after it

Use [`same_molecule`](docs/molecule-equality.md) when you want equality modulo:

- atom-map ordering
- system-tuple ordering
- member-edge set ordering
- annotation tuple ordering

It keeps atom and system identifiers meaningful.

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

`ElementAttributes` also carries the default shell data:

- simple atom builders can use `element_attributes(symbol)`
- atom builders can omit `shells`
- there is no separate shell lookup layer

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
| Ferrocene | Cp/C-H `single covalent` systems plus two Fe-centred Cp delocalised systems; Fe is `+2` and one representative carbon on each Cp ring is `-1` |
| Sodium chloride | `Na+` and `Cl-` atoms plus one zero-electron `ionic` edge system |
| Morphine | every graph edge as a system, including a `double covalent` alkene edge, plus a phenyl delocalised system |

Ferrocene is a useful example because the metallocene structure stays explicit:

- it is not flattened into a string
- each Cp delocalised system spans five Cp ring C-C edges
- each Cp delocalised system also spans five Fe-C contacts to the central iron
- overlapping delocalised and covalent systems get separate dashed lanes in the
  viewer
- ordinary Cp/C-H and Cp/Cp covalent edges are still present as unnamed
  one-edge systems that display as `single covalent`

| System | Shared electrons | Member edges |
| --- | --- | --- |
| `cp1_pi` | `6e` | the five C-C edges in the first Cp ring plus the five Fe-C contacts to that ring |
| `cp2_pi` | `6e` | the five C-C edges in the second Cp ring plus the five Fe-C contacts to that ring |

See the full expanded ADT in [`moladt/examples/ferrocene.py`](moladt/examples/ferrocene.py).

## Validator

`validate_molecule(molecule)` is a representation validator, not a physical
chemistry oracle. It rejects malformed MolADT values before parsing, viewing,
feature extraction, or inverse-design scoring continue.

It checks:

- atom map keys match the atom IDs
- coordinates and element metadata are finite
- `SystemId`s are positive and unique
- bonding systems are non-empty
- bonding-system edges reference existing atoms
- cached member atoms match the member edges
- duplicate bonding systems are not present
- SMILES stereochemistry annotations only point at known atoms
- ordinary one-edge covalent systems with `2`, `4`, `6`, or `8` shared
  electrons are unnamed and display as single/double/triple/quadruple covalent
  bonds
- one-edge `0e` systems are tagged `ionic`
- `ionic` systems share zero electrons over exactly one edge

It deliberately does not:

- prove a molecule is physically realistic
- choose protonation states
- infer missing hydrogens
- decide whether a delocalised system is chemically preferred

Task-specific checks can add those rules separately. The FreeSolv
inverse-design task now runs this generic validator before its own FreeSolv
geometry and generation-contract checks.

## View Molecules

```bash
make view
make molecule-viewer VIEWER_EXAMPLES="benzene diborane ferrocene"
make python-pretty-example EXAMPLE=morphine
```

`make view` opens seven built-in examples in one browser page.

The viewer renders:

- sodium chloride's charged `0e` ionic edge
- positive and negative formal charge as blue and red halos
- charge halo size and opacity proportional to formal-charge magnitude
- an extra halo boost for atoms in an ionic bonding system
- ionic edges as blue-to-red gradients between charged atoms
- ordinary covalent edges in dark grey
- single covalent bonds as one line
- double covalent bonds as two lines around the edge axis
- triple covalent bonds as three lines
- quadruple covalent bonds as four lines
- overlapping bonding systems as separate dashed lanes, including ordinary
  covalent versus delocalised overlap in ferrocene
- non-standard systems such as pi rings, bridge bonds, and coordination as
  coloured dashed overlays

The right panel lists:

- all bonding systems
- `single covalent`, `double covalent`, `triple covalent`, and
  `quadruple covalent` one-edge systems
- selected atom coordinates
- 3D edge lengths
- effective orders
- bonding systems and bond angles from molecule coordinates

Bonding-system text labels are not drawn over the molecule.

Viewer opening behavior:

- use `OPEN_VIEWER=1` to open generated viewer pages automatically
- viewer commands also print a portable `file://` URL
- if the operating system does not open a browser, open the reported URL
  manually

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

`make inverse-design`:

- uses the latest `results/freesolv/run_*` Bayesian GP artifact
- samples initial chains from the MolADT FreeSolv prior
- reweights that prior with the unchanged GP target likelihood
- generates 1,000 candidates
- writes the top 10 by Bayesian credible score

Output notes:

- geometry values are audited in the outputs
- the sampler is not conditioned on physical plausibility
- generated reference outputs live in `results/inverse_design/reference/`

## Benchmarks

| Benchmark | Target | Main command |
| --- | --- | --- |
| FreeSolv | hydration free energy | `make freesolv` |
| FreeSolv inverse design | target hydration free energy | `make inverse-design TARGET=-5.0` |
| QM9 | dipole moment `mu` | `make qm9long` |
| ZINC timing | representation throughput | `make timing` |

Benchmark details are in [Inference and benchmarks](docs/inference-and-benchmarks.md), [Models and features](docs/models.md), [Outputs](docs/outputs.md), and [results README](results/README.md).

## Python And Haskell

MolADT JSON is the boundary shared by the Python and Haskell repos:

- Python writes MolADT JSON
- Haskell reads the same MolADT JSON shape
- atom IDs, system IDs, bonding systems, charges, and stereochemistry remain
  explicit

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
stack run moladtbayes -- freesolv-wl-system-gp
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
