from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeAlias, TypeVar, assert_never

from .coordinate import Coordinate, mk_angstrom


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

    def __post_init__(self) -> None:
        if self.electron_count < 0:
            raise ValueError("electron_count must be >= 0")
        if self.hybrid_components is not None:
            object.__setattr__(self, "hybrid_components", tuple(self.hybrid_components))

    def to_dict(self) -> dict[str, Any]:
        return {
            "orbital_type": self.orbital_type.value,
            "electron_count": self.electron_count,
            "orientation": None if self.orientation is None else self.orientation.to_dict(),
            "hybrid_components": None
            if self.hybrid_components is None
            else [
                {"weight": weight, "pure_orbital": pure_orbital_to_dict(pure)}
                for weight, pure in self.hybrid_components
            ],
        }

    def pretty(self) -> str:
        from .pretty import pretty_text

        return pretty_text(self)

    def __str__(self) -> str:
        return self.pretty()


@dataclass(frozen=True, slots=True)
class SubShell(Generic[SubshellType]):
    orbitals: tuple[Orbital[SubshellType], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "orbitals", tuple(self.orbitals))

    def to_dict(self) -> dict[str, Any]:
        return {"orbitals": [orbital.to_dict() for orbital in self.orbitals]}

    def pretty(self) -> str:
        from .pretty import pretty_text

        return pretty_text(self)

    def __str__(self) -> str:
        return self.pretty()


@dataclass(frozen=True, slots=True)
class Shell:
    principal_quantum_number: int
    s_subshell: SubShell[So] | None = None
    p_subshell: SubShell[P] | None = None
    d_subshell: SubShell[D] | None = None
    f_subshell: SubShell[F] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "principal_quantum_number": self.principal_quantum_number,
            "s_subshell": None if self.s_subshell is None else self.s_subshell.to_dict(),
            "p_subshell": None if self.p_subshell is None else self.p_subshell.to_dict(),
            "d_subshell": None if self.d_subshell is None else self.d_subshell.to_dict(),
            "f_subshell": None if self.f_subshell is None else self.f_subshell.to_dict(),
        }

    def pretty(self) -> str:
        from .pretty import pretty_text

        return pretty_text(self)

    def __str__(self) -> str:
        return self.pretty()


Shells: TypeAlias = tuple[Shell, ...]


def pure_orbital_to_dict(pure_orbital: PureOrbital) -> dict[str, str]:
    match pure_orbital:
        case PureSOrbital(orbital=orbital):
            return {"kind": "s", "orbital": orbital.value}
        case PurePOrbital(orbital=orbital):
            return {"kind": "p", "orbital": orbital.value}
        case PureDOrbital(orbital=orbital):
            return {"kind": "d", "orbital": orbital.value}
        case PureFOrbital(orbital=orbital):
            return {"kind": "f", "orbital": orbital.value}
        case _ as unreachable:
            assert_never(unreachable)


def pure_orbital_from_dict(data: dict[str, Any]) -> PureOrbital:
    kind = data["kind"]
    orbital = str(data["orbital"])
    if kind == "s":
        return PureSOrbital(So(orbital))
    if kind == "p":
        return PurePOrbital(P(orbital))
    if kind == "d":
        return PureDOrbital(D(orbital))
    if kind == "f":
        return PureFOrbital(F(orbital))
    raise ValueError(f"Unknown pure orbital kind: {kind}")


def ang_coord(x: float, y: float, z: float) -> Coordinate:
    return Coordinate(mk_angstrom(x), mk_angstrom(y), mk_angstrom(z))


def _orbital(
    orbital_type: SubshellType,
    electron_count: int,
    orientation: Coordinate | None = None,
) -> Orbital[SubshellType]:
    return Orbital(orbital_type=orbital_type, electron_count=electron_count, orientation=orientation)


def _subshell(*orbitals: Orbital[SubshellType]) -> SubShell[SubshellType]:
    return SubShell(orbitals=tuple(orbitals))


def _shell(
    principal_quantum_number: int,
    *,
    s_counts: tuple[int, ...] | None = None,
    p_counts: tuple[int, int, int] | None = None,
    d_counts: tuple[int, int, int, int, int] | None = None,
    f_counts: tuple[int, int, int, int, int, int, int] | None = None,
) -> Shell:
    s_subshell = None
    p_subshell = None
    d_subshell = None
    f_subshell = None
    if s_counts is not None:
        s_subshell = _subshell(*(_orbital(So.S, count) for count in s_counts))
    if p_counts is not None:
        p_subshell = _subshell(
            _orbital(P.PX, p_counts[0], ang_coord(1.0, 0.0, 0.0)),
            _orbital(P.PY, p_counts[1], ang_coord(0.0, 1.0, 0.0)),
            _orbital(P.PZ, p_counts[2], ang_coord(0.0, 0.0, 1.0)),
        )
    if d_counts is not None:
        s2 = 2.0**0.5
        d_subshell = _subshell(
            _orbital(D.DXY, d_counts[0], ang_coord(1.0 / s2, 1.0 / s2, 0.0)),
            _orbital(D.DYZ, d_counts[1], ang_coord(0.0, 1.0 / s2, 1.0 / s2)),
            _orbital(D.DXZ, d_counts[2], ang_coord(1.0 / s2, 0.0, 1.0 / s2)),
            _orbital(D.DX2Y2, d_counts[3], ang_coord(1.0 / s2, -1.0 / s2, 0.0)),
            _orbital(D.DZ2, d_counts[4], ang_coord(0.0, 0.0, 1.0)),
        )
    if f_counts is not None:
        f_subshell = _subshell(
            _orbital(F.FXXX, f_counts[0], ang_coord(1.0, 0.0, 0.0)),
            _orbital(F.FXXY, f_counts[1], ang_coord(1.0, 1.0, 0.0)),
            _orbital(F.FXXZ, f_counts[2], ang_coord(1.0, 0.0, 1.0)),
            _orbital(F.FXYY, f_counts[3], ang_coord(1.0, 1.0, 0.0)),
            _orbital(F.FXYZ, f_counts[4], ang_coord(1.0, 1.0, 1.0)),
            _orbital(F.FXZZ, f_counts[5], ang_coord(1.0, 0.0, 1.0)),
            _orbital(F.FZZZ, f_counts[6], ang_coord(0.0, 0.0, 1.0)),
        )
    return Shell(
        principal_quantum_number=principal_quantum_number,
        s_subshell=s_subshell,
        p_subshell=p_subshell,
        d_subshell=d_subshell,
        f_subshell=f_subshell,
    )


HYDROGEN: Shells = (_shell(1, s_counts=(1,)),)
CARBON: Shells = (_shell(1, s_counts=(2,)), _shell(2, s_counts=(2,), p_counts=(1, 1, 0)))
NITROGEN: Shells = (_shell(1, s_counts=(2,)), _shell(2, s_counts=(2,), p_counts=(1, 1, 1)))
OXYGEN: Shells = (_shell(1, s_counts=(2,)), _shell(2, s_counts=(2,), p_counts=(2, 1, 1)))
BORON: Shells = (_shell(1, s_counts=(2,)), _shell(2, s_counts=(2,), p_counts=(1, 0, 0)))
FLUORINE: Shells = (_shell(1, s_counts=(2,)), _shell(2, s_counts=(2,), p_counts=(2, 2, 1)))
SODIUM: Shells = (
    _shell(1, s_counts=(2,)),
    _shell(2, s_counts=(2,), p_counts=(2, 2, 2)),
    _shell(3, s_counts=(1,)),
)
CHLORINE: Shells = (
    _shell(1, s_counts=(2,)),
    _shell(2, s_counts=(2,), p_counts=(2, 2, 2)),
    _shell(3, s_counts=(2,), p_counts=(2, 2, 1)),
)
PHOSPHORUS: Shells = (
    _shell(1, s_counts=(2,)),
    _shell(2, s_counts=(2,), p_counts=(2, 2, 2)),
    _shell(3, s_counts=(2,), p_counts=(1, 1, 1)),
)
SILICON: Shells = (
    _shell(1, s_counts=(2,)),
    _shell(2, s_counts=(2,), p_counts=(2, 2, 2)),
    _shell(3, s_counts=(2,), p_counts=(1, 1, 0)),
)
SULFUR: Shells = (
    _shell(1, s_counts=(2,)),
    _shell(2, s_counts=(2,), p_counts=(2, 2, 2)),
    _shell(3, s_counts=(2,), p_counts=(2, 1, 1)),
)
BROMINE: Shells = (
    _shell(1, s_counts=(2,)),
    _shell(2, s_counts=(2,), p_counts=(2, 2, 2)),
    _shell(3, s_counts=(2,), p_counts=(2, 2, 2), d_counts=(2, 2, 2, 2, 2)),
    _shell(4, s_counts=(2,), p_counts=(2, 2, 1)),
)
IODINE: Shells = (
    _shell(1, s_counts=(2,)),
    _shell(2, s_counts=(2,), p_counts=(2, 2, 2)),
    _shell(3, s_counts=(2,), p_counts=(2, 2, 2), d_counts=(2, 2, 2, 2, 2)),
    _shell(4, s_counts=(2,), p_counts=(2, 2, 2), d_counts=(2, 2, 2, 2, 2)),
    _shell(5, s_counts=(2,), p_counts=(2, 2, 1)),
)
IRON: Shells = (
    _shell(1, s_counts=(2,)),
    _shell(2, s_counts=(2,), p_counts=(2, 2, 2)),
    _shell(3, s_counts=(2,), p_counts=(2, 2, 2), d_counts=(2, 1, 1, 1, 1)),
    _shell(4, s_counts=(2,)),
)
