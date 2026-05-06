# Orbitals

Atoms can carry local shell and orbital data. Shells are optional on `Atom`,
and default element shells live in `ElementAttributes`.

```text
Molecule -> Atom -> Shells? -> Shell -> SubShell -> Orbital
```

The Python types live in [`moladt/chem/orbital.py`](../moladt/chem/orbital.py), and mirror the ADT shape used by the sibling Haskell repo.

## ADT Shape

```haskell
data So = So
  deriving (Show, Eq, Read)

data P = Px | Py | Pz
  deriving (Show, Eq, Read)

data D = Dxy | Dyz | Dxz | Dx2y2 | Dz2
  deriving (Show, Eq, Read)

data F = Fxxx | Fxxy | Fxxz | Fxyy | Fxyz | Fxzz | Fzzz
  deriving (Show, Eq, Read)

data PureOrbital
  = PureSo So
  | PureP  P
  | PureD  D
  | PureF  F
  deriving (Show, Eq, Read)

data Orbital subshellType = Orbital
  { orbitalType      :: subshellType
  , electronCount    :: Int
  , orientation      :: Maybe Coordinate
  , hybridComponents :: Maybe [(Double, PureOrbital)]
  } deriving (Show, Eq, Read)

newtype SubShell subshellType = SubShell
  { orbitals :: [Orbital subshellType]
  } deriving (Show, Eq, Read)

data Shell = Shell
  { principalQuantumNumber :: Int
  , sSubShell              :: Maybe (SubShell So)
  , pSubShell              :: Maybe (SubShell P)
  , dSubShell              :: Maybe (SubShell D)
  , fSubShell              :: Maybe (SubShell F)
  } deriving (Show, Eq, Read)

type Shells = [Shell]
```

The Python version is the same structure in dataclass form:

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

An `Orbital` records its typed orbital constructor, electron count, optional orientation, and optional hybrid components. A `Shell` records one principal quantum number and optional `s`, `p`, `d`, and `f` subshells. A molecule can omit atom shells at a parser or JSON boundary; constructors fill defaults from `element_attributes(symbol)` when available.

## Iodine Valence

Iodine is a useful example because the shell data is still compact but no longer looks like a toy carbon example. Its final valence shell is `5s2 5p5`:

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

Python stores the same iodine shell occupancy directly:

```python
IODINE: Shells = (
    _shell(1, s_counts=(2,)),
    _shell(2, s_counts=(2,), p_counts=(2, 2, 2)),
    _shell(3, s_counts=(2,), p_counts=(2, 2, 2), d_counts=(2, 2, 2, 2, 2)),
    _shell(4, s_counts=(2,), p_counts=(2, 2, 2), d_counts=(2, 2, 2, 2, 2)),
    _shell(5, s_counts=(2,), p_counts=(2, 2, 1)),
)
```

## Why Keep This

This gives MolADT a typed place for local electronic structure:

- shell occupancy
- directional orbital character
- simple hybrid descriptions

That is richer than a graph-only molecule while staying much lighter than a quantum chemistry engine.

## What It Does Not Claim

The orbital layer is not:

- a basis set
- an SCF state
- a molecular orbital calculation
- a Hamiltonian or overlap matrix

It is local structured chemistry data attached to atoms. See [Representation](representation.md) for where `shells` sits inside `Atom`.
