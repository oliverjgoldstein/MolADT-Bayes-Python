# Representation

MolADT is a molecule ADT.

The central object is not a SMILES string, a plain graph, or an untyped hypergraph. It is a record: atoms, Dietz bonding systems, coordinates, charges, and optional shell data all have their own fields. The edge network is derived from bonding-system member edges.

## Molecule ADT

The sibling Haskell repo writes the core representation directly as an algebraic data type:

```haskell
data Molecule = Molecule
  { atoms      :: Map AtomId Atom
  , systems    :: [(SystemId, BondingSystem)]
  , smilesStereochemistry :: SmilesStereochemistry
  }
```

Python mirrors the same shape with typed dataclasses and snake-case names:

```python
@dataclass(frozen=True, slots=True)
class Molecule:
    atoms: Mapping[AtomId, Atom]
    systems: tuple[tuple[SystemId, BondingSystem], ...] = ()
    smiles_stereochemistry: SmilesStereochemistry = field(default_factory=SmilesStereochemistry)
```

| Haskell field | Python field | Meaning |
| --- | --- | --- |
| `atoms` | `atoms` | Atom table keyed by stable ids. |
| `systems` | `systems` | Canonical Dietz electron-sharing systems. |
| `smilesStereochemistry` | `smiles_stereochemistry` | Optional SMILES stereo annotations. |

Every edge is represented by a bonding system. Conventional single, double,
triple, and quadruple bonds are one-edge systems sharing `2`, `4`, `6`, and `8`
electrons, respectively, displayed as `single covalent`, `double covalent`,
`triple covalent`, and `quadruple covalent`.
Ionic contacts use the same edge rule: they are one-edge systems with `0`
shared electrons, tag `ionic`, and formal charge on the atoms.

System identifiers are stable display IDs. Parsers and checked examples assign
them to named or multi-edge systems first, then to the ordinary one-edge
covalent systems. In benzene, `SystemId(1)` is therefore the `pi_ring`; the
sigma edge systems follow.

## Atom ADT

An atom is the point where element data, position, charge, and orbital shells meet:

```haskell
data AtomicSymbol = H | ... | Og

data ElementAttributes = ElementAttributes
  { symbol       :: AtomicSymbol
  , atomicNumber :: Int
  , atomicWeight :: Double
  , defaultShells :: Shells
  }

data Atom = Atom
  { atomID       :: AtomId
  , attributes   :: ElementAttributes
  , coordinate   :: Coordinate
  , shells       :: Shells
  , formalCharge :: Int
  }
```

```python
class AtomicSymbol(Enum):
    H = "H"
    ...
    Og = "Og"

@dataclass(frozen=True, slots=True)
class ElementAttributes:
    symbol: AtomicSymbol
    atomic_number: int
    atomic_weight: float
    shells: Shells | None = None

@dataclass(frozen=True, slots=True)
class Atom:
    atom_id: AtomId
    attributes: ElementAttributes
    coordinate: Coordinate
    shells: Shells | None = None
    formal_charge: int = 0
```

`AtomicSymbol` covers all 118 official elements. `element_attributes(symbol)`
uses CIAAW 2024 standard atomic weights where they exist; elements without a
standard atomic weight use the NIST SP 966 June 2024 longest-lived-isotope mass
number. Detailed default shell objects are still only attached for the
project's audited element subset; other elements keep `shells=None`.

The `formalCharge` field is explicit. Shells are optional on atoms and are also
available from `ElementAttributes`, so simple constructors can take defaults
from `element_attributes(symbol)` while JSON/parser boundaries can omit shell
payloads.

## Dietz Layer

MolADT stores bonding as electron-sharing systems:

```haskell
newtype AtomId   = AtomId Integer
newtype SystemId = SystemId Int

data Edge = Edge AtomId AtomId

data BondingSystem = BondingSystem
  { sharedElectrons :: NonNegative
  , memberAtoms     :: Set AtomId
  , memberEdges     :: Set Edge
  , tag             :: Maybe String
  }
```

```python
@dataclass(frozen=True, slots=True, order=True)
class AtomId:
    value: int

@dataclass(frozen=True, slots=True, order=True)
class SystemId:
    value: int

@dataclass(frozen=True, slots=True, order=True)
class Edge:
    a: AtomId
    b: AtomId

@dataclass(frozen=True, slots=True)
class BondingSystem:
    shared_electrons: NonNegative
    member_atoms: frozenset[AtomId]
    member_edges: frozenset[Edge]
    tag: str | None = None
```

That is why the same molecule can carry both conventional bonds and explicit
non-classical bonding in one layer. Examples:

| Molecule | What the ADT stores explicitly |
| --- | --- |
| Benzene | six `single covalent` one-edge systems plus a six-electron `pi_ring` system |
| Diborane | four terminal B-H `single covalent` systems plus two `3c-2e` bridge systems |
| Ferrocene | Cp/C-H `single covalent` systems plus Fe-centred Cp delocalised systems; `Fe#1` is `+2`, with one representative `-1` carbon per Cp ring |
| Sodium chloride | `Na+` and `Cl-` atoms plus one zero-electron `ionic` system over the Na-Cl edge |
| Morphine | every graph edge as a system, including a `double covalent` alkene edge, plus a phenyl delocalisation system |

## Orbital Layer

Shells are also ADTs. The Haskell shape is:

```haskell
data So = So
data P = Px | Py | Pz
data D = Dxy | Dyz | Dxz | Dx2y2 | Dz2
data F = Fxxx | Fxxy | Fxxz | Fxyy | Fxyz | Fxzz | Fzzz

data PureOrbital
  = PureSo So
  | PureP  P
  | PureD  D
  | PureF  F

data Orbital subshellType = Orbital
  { orbitalType      :: subshellType
  , electronCount    :: Int
  , orientation      :: Maybe Coordinate
  , hybridComponents :: Maybe [(Double, PureOrbital)]
  }

newtype SubShell subshellType = SubShell
  { orbitals :: [Orbital subshellType] }

data Shell = Shell
  { principalQuantumNumber :: Int
  , sSubShell              :: Maybe (SubShell So)
  , pSubShell              :: Maybe (SubShell P)
  , dSubShell              :: Maybe (SubShell D)
  , fSubShell              :: Maybe (SubShell F)
  }

type Shells = [Shell]
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
class SubShell(Generic[SubshellType]):
    orbitals: tuple[Orbital[SubshellType], ...]

@dataclass(frozen=True, slots=True)
class Shell:
    principal_quantum_number: int
    s_subshell: SubShell[So] | None = None
    p_subshell: SubShell[P] | None = None
    d_subshell: SubShell[D] | None = None
    f_subshell: SubShell[F] | None = None

Shells: TypeAlias = tuple[Shell, ...]
```

For iodine, the final valence shell is represented as `5s2 5p5`:

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

Python stores iodine as the same shell tuple:

```python
IODINE: Shells = (
    _shell(1, s_counts=(2,)),
    _shell(2, s_counts=(2,), p_counts=(2, 2, 2)),
    _shell(3, s_counts=(2,), p_counts=(2, 2, 2), d_counts=(2, 2, 2, 2, 2)),
    _shell(4, s_counts=(2,), p_counts=(2, 2, 2), d_counts=(2, 2, 2, 2, 2)),
    _shell(5, s_counts=(2,), p_counts=(2, 2, 1)),
)
```

See [Orbitals](orbitals.md) for the orbital model in more detail.

## Why It Helps

SMILES is a useful boundary string. It is not the best working object for Bayesian generation.

MolADT keeps the support of the model explicit:

- priors operate over molecule fields
- proposal kernels modify typed atoms, edges, systems, and charges
- validators enforce valence, connectivity, and bonding-system invariants
- feature maps and posterior scores stay attached to the exact generated object
- JSON exports preserve the same typed candidate for later inspection

For the FreeSolv inverse task, generation rules are built around feasible local moves over this ADT. Validation remains the final guardrail, but the generator is not trying arbitrary strings and hoping they parse.

## Boundary Formats

MolADT can read and write boundary formats:

- SDF for structure files
- conservative SMILES for classical strings
- MolADT JSON for typed interchange
- standalone HTML viewer exports

The boundary format is not the model. The `Molecule` ADT is the model.

See [Examples](examples.md), [Parsing](parsing.md), [Orbitals](orbitals.md), [CLI](cli.md), and [Haskell interop](haskell_interop.md).
