# MolADT-Bayes-Python

Design molecules as data.

MolADT is a typed molecule representation for Bayesian modelling, feature generation, validation, and inverse design. It keeps the chemistry in the object: atoms, coordinates, sigma bonds, delocalised or multicentre bonding systems, formal charges, and stereochemistry all live as structured fields instead of being compressed into a string.

Not just a SMILES string. Not just a graph. A molecule object you can inspect, mutate, validate, score, serialize, and share.

[Quickstart](docs/quickstart.md) · [Representation](docs/representation.md) · [Examples](docs/examples.md) · [CLI](docs/cli.md) · [Models](docs/models.md) · [Benchmarks](docs/inference-and-benchmarks.md) · [Outputs](docs/outputs.md)

## Why MolADT

Molecular ML pipelines often begin with a compact boundary format, then add chemistry back later. MolADT starts with the chemistry.

| What you need | What MolADT gives you |
| --- | --- |
| Explicit structure | `atoms`, `local_bonds`, `systems`, and `smiles_stereochemistry` as first-class fields |
| Typed proposal space | proposal kernels operate on molecule fields and preserve invariants before scoring |
| Unusual bonding | diborane `3c-2e` bridges, ferrocene Cp/metal systems, aromatic pi systems |
| Safe scoring | validate valence, connectivity, bonding systems, and FreeSolv-specific candidate rules before prediction |
| Interop | JSON payloads and processed matrices shared with the Haskell repo |
| Benchmarks | FreeSolv, QM9, ZINC timing, and committed result artifacts |

Read the deeper case for the representation in [MolADT representation](docs/representation.md) and [orbitals](docs/orbitals.md).

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
make view
```

`make view` refreshes the repo Python bytecode cache, regenerates the viewer HTML, and opens six example molecules in one browser page with a scrollable molecule list. Click any atom to show its stored orbital glyphs, coordinates, 3D edge lengths, and bond angles calculated from the molecule's 3D coordinates.

For Stan-backed FreeSolv runs:

```bash
make python-cmdstan-install
```

`make python-setup` creates `./.venv` inside this repo. `make python-cmdstan-install` creates `./.cmdstan`. Both are local to the checkout.

Full setup notes live in [Quickstart](docs/quickstart.md).

## See A Molecule

```bash
make view
make python-pretty-example EXAMPLE=morphine
make molecule-viewer VIEWER_EXAMPLES=ferrocene
make test-molecule-viewer
```

`make view` opens the built-in example molecules in one browser page. `make molecule-viewer` writes a fresh HTML viewer under `results/viewer/`. Click an atom to see orbitals, coordinates, 3D edge lengths, and bond angles.

Use `VIEWER_EXAMPLES` to choose molecules:

```bash
make molecule-viewer VIEWER_EXAMPLES="benzene diborane ferrocene"
```

Use `OPEN_VIEWER=1` when you want Make to open the generated page:

```bash
OPEN_VIEWER=1 make python-pretty-example EXAMPLE=ferrocene
OPEN_VIEWER=1 make molecule-viewer VIEWER_EXAMPLES=diborane
OPEN_VIEWER=1 make inverse-design TARGET=-5.0
```

Inverse design opens the top 10 generated molecules by default.

More examples: [Examples](docs/examples.md), [Parsing and rendering](docs/parsing.md), and [CLI reference](docs/cli.md).

## Pretty Representation Examples

The built-in examples are written as explicit typed molecules. They are intentionally not just display fixtures: the same values can be validated, serialized, opened in the viewer, featurized, and used as proposal states.

Canonical normal form means fully expanded Python ADT source: atoms sorted by `AtomId`, `Edge(AtomId(a), AtomId(b))` normalized with `a < b`, systems sorted by `SystemId`, and no loops hiding atoms, bonds, or bonding systems. `to-python` and `to-example` emit that form for parser fixtures.

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

## The ADT At The Center

The sibling Haskell repo states the concept as record ADTs. Python mirrors that shape with typed dataclasses, so the molecule remains the same object across generation, validation, scoring, JSON, and Haskell interop.

```haskell
data Molecule = Molecule
  { atoms      :: Map AtomId Atom
  , localBonds :: Set Edge
  , systems    :: [(SystemId, BondingSystem)]
  , smilesStereochemistry :: SmilesStereochemistry
  }
```

```python
@dataclass(frozen=True, slots=True)
class Molecule:
    atoms: Mapping[AtomId, Atom]
    local_bonds: frozenset[Edge]
    systems: tuple[tuple[SystemId, BondingSystem], ...]
    smiles_stereochemistry: SmilesStereochemistry = field(default_factory=SmilesStereochemistry)
```

An atom is not only a label in a graph. It carries element attributes, position, charge, and local shell data:

```haskell
data Atom = Atom
  { atomID       :: AtomId
  , attributes   :: ElementAttributes
  , coordinate   :: Coordinate
  , shells       :: Shells
  , formalCharge :: Int
  }
```

```python
@dataclass(frozen=True, slots=True)
class Atom:
    atom_id: AtomId
    attributes: ElementAttributes
    coordinate: Coordinate
    shells: Shells
    formal_charge: int = 0
```

Delocalised and multicentre bonds are first-class values:

```haskell
data BondingSystem = BondingSystem
  { sharedElectrons :: NonNegative
  , memberAtoms     :: Set AtomId
  , memberEdges     :: Set Edge
  , tag             :: Maybe String
  }
```

```python
@dataclass(frozen=True, slots=True)
class BondingSystem:
    shared_electrons: NonNegative
    member_atoms: frozenset[AtomId]
    member_edges: frozenset[Edge]
    tag: str | None = None
```

The orbital layer is typed too:

```haskell
data P = Px | Py | Pz
data D = Dxy | Dyz | Dxz | Dx2y2 | Dz2

data Orbital subshellType = Orbital
  { orbitalType      :: subshellType
  , electronCount    :: Int
  , orientation      :: Maybe Coordinate
  , hybridComponents :: Maybe [(Double, PureOrbital)]
  }

data Shell = Shell
  { principalQuantumNumber :: Int
  , sSubShell              :: Maybe (SubShell So)
  , pSubShell              :: Maybe (SubShell P)
  , dSubShell              :: Maybe (SubShell D)
  , fSubShell              :: Maybe (SubShell F)
  }
```

```python
class So(Enum):
    S = "s"

class P(Enum):
    PX = "px"
    PY = "py"
    PZ = "pz"

class D(Enum):
    DXY = "dxy"
    DYZ = "dyz"
    DXZ = "dxz"
    DX2Y2 = "dx2y2"
    DZ2 = "dz2"

class F(Enum):
    FXXX = "fxxx"
    FXXY = "fxxy"
    FXXZ = "fxxz"
    FXYY = "fxyy"
    FXYZ = "fxyz"
    FXZZ = "fxzz"
    FZZZ = "fzzz"

@dataclass(frozen=True, slots=True)
class PureSOrbital:
    orbital: So

@dataclass(frozen=True, slots=True)
class PurePOrbital:
    orbital: P

@dataclass(frozen=True, slots=True)
class PureDOrbital:
    orbital: D

@dataclass(frozen=True, slots=True)
class PureFOrbital:
    orbital: F

PureOrbital: TypeAlias = PureSOrbital | PurePOrbital | PureDOrbital | PureFOrbital
SubshellType = TypeVar("SubshellType", So, P, D, F)

@dataclass(frozen=True, slots=True)
class Orbital(Generic[SubshellType]):
    orbital_type: SubshellType
    electron_count: int
    orientation: Coordinate | None = None
    hybrid_components: tuple[tuple[float, PureOrbital], ...] | None = None

@dataclass(frozen=True, slots=True)
class Shell:
    principal_quantum_number: int
    s_subshell: SubShell[So] | None = None
    p_subshell: SubShell[P] | None = None
    d_subshell: SubShell[D] | None = None
    f_subshell: SubShell[F] | None = None

Shells: TypeAlias = tuple[Shell, ...]
```

That lets a heavier atom such as iodine be represented by shell occupancy rather than by an opaque string. Its final valence shell is the `5s2 5p5` part of the ADT:

```haskell
Shell
  { principalQuantumNumber = 5
  , sSubShell = Just (SubShell
      [ Orbital
          { orbitalType      = So
          , electronCount    = 2
          , orientation      = Nothing
          , hybridComponents = Nothing
          }
      ])
  , pSubShell = Just (SubShell
      [ Orbital
          { orbitalType      = Px
          , electronCount    = 2
          , orientation      = Just (angCoord 1 0 0)
          , hybridComponents = Nothing
          }
      , Orbital
          { orbitalType      = Py
          , electronCount    = 2
          , orientation      = Just (angCoord 0 1 0)
          , hybridComponents = Nothing
          }
      , Orbital
          { orbitalType      = Pz
          , electronCount    = 1
          , orientation      = Just (angCoord 0 0 1)
          , hybridComponents = Nothing
          }
      ])
  , dSubShell = Nothing
  , fSubShell = Nothing
  }
```

Python stores the same iodine occupancy in [`moladt/chem/orbital.py`](moladt/chem/orbital.py):

```python
IODINE: Shells = (
    _shell(1, s_counts=(2,)),
    _shell(2, s_counts=(2,), p_counts=(2, 2, 2)),
    _shell(3, s_counts=(2,), p_counts=(2, 2, 2), d_counts=(2, 2, 2, 2, 2)),
    _shell(4, s_counts=(2,), p_counts=(2, 2, 2), d_counts=(2, 2, 2, 2, 2)),
    _shell(5, s_counts=(2,), p_counts=(2, 2, 1)),
)
```

See [Representation](docs/representation.md), [Orbitals](docs/orbitals.md), and [Examples](docs/examples.md) for how the same ADT covers diborane, ferrocene, morphine, benzene, and molecules converted into MolADT JSON.

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
make inverse-design-view
```

The first command checks the predictive model on the FreeSolv split. The second command uses the latest `results/freesolv/run_*` FreeSolv GP artifact to search for molecules near the requested hydration free energy. Initial chains are sampled from the valid MolADT FreeSolv prior by default, reweighted by the unchanged GP target likelihood; pass `SEED_MOLECULE=water` for a fixed water start. If the newest FreeSolv run is incomplete, inverse design fails instead of falling back to an older model. The inverse-design run writes top generated molecule files under `results/inverse_design/run_.../` and refreshes the reference files under `results/inverse_design/reference/`. `make inverse-design-view` opens the saved reference molecules in one viewer page; use `INVERSE_DESIGN_VIEW_DIR=results/inverse_design/run_...` to inspect a specific run.

The FreeSolv generator uses MolADT/Dietz-aware edits whose move rules choose only locally feasible atoms, elements, and rings. Before scoring, generated candidates pass the FreeSolv generation contract: connected, neutral, CHONFCl-only, closed-valence, sound bonding systems, non-overlapping coordinates, plausible local bond lengths, van der Waals non-bonded clearance, and minimum local bond angles. That enforces the relevant graph, electron-count, and coordinate sanity rules for this generator; it is not a substitute for quantum relaxation, thermodynamic stability analysis, or a synthesizability filter.

Benchmark details are in [Inference and benchmarks](docs/inference-and-benchmarks.md), [Models and features](docs/models.md), [Outputs](docs/outputs.md), and [results README](results/README.md).

## Modules

| Area | Start here |
| --- | --- |
| First run | [Quickstart](docs/quickstart.md) |
| Representation | [MolADT representation](docs/representation.md), [orbitals](docs/orbitals.md) |
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
