from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True, order=True)
class AtomId:
    value: int

    def __post_init__(self) -> None:
        if self.value < 1:
            raise ValueError("AtomId must be positive")

    def to_dict(self) -> dict[str, int]:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AtomId:
        return cls(int(data["value"]))


@dataclass(frozen=True, slots=True, order=True)
class SystemId:
    value: int

    def __post_init__(self) -> None:
        if self.value < 1:
            raise ValueError("SystemId must be positive")

    def to_dict(self) -> dict[str, int]:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SystemId:
        return cls(int(data["value"]))


@dataclass(frozen=True, slots=True, order=True)
class NonNegative:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("NonNegative must be >= 0")

    def to_dict(self) -> dict[str, int]:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NonNegative:
        return cls(int(data["value"]))


@dataclass(frozen=True, slots=True, order=True)
class Edge:
    a: AtomId
    b: AtomId

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError("self-bond")
        if self.a.value > self.b.value:
            original_a = self.a
            object.__setattr__(self, "a", self.b)
            object.__setattr__(self, "b", original_a)

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {"a": self.a.to_dict(), "b": self.b.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edge:
        return cls(AtomId.from_dict(data["a"]), AtomId.from_dict(data["b"]))

    def __str__(self) -> str:
        return f"{self.a.value}-{self.b.value}"


def mk_edge(a: AtomId, b: AtomId) -> Edge:
    return Edge(a, b)


def atoms_of_edge(edge: Edge) -> tuple[AtomId, AtomId]:
    return edge.a, edge.b


@dataclass(frozen=True, slots=True)
class BondingSystem:
    shared_electrons: NonNegative
    member_atoms: frozenset[AtomId]
    member_edges: frozenset[Edge]
    tag: str | None = None

    def __post_init__(self) -> None:
        member_edges = frozenset(self.member_edges)
        member_atoms = frozenset(self.member_atoms)
        derived_atoms = frozenset(atom for edge in member_edges for atom in atoms_of_edge(edge))
        if member_atoms != derived_atoms:
            raise ValueError("member_atoms must match atoms implied by member_edges")
        object.__setattr__(self, "member_edges", member_edges)
        object.__setattr__(self, "member_atoms", member_atoms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_electrons": self.shared_electrons.to_dict(),
            "member_atoms": [atom.to_dict() for atom in sorted(self.member_atoms)],
            "member_edges": [edge.to_dict() for edge in sorted(self.member_edges)],
            "tag": self.tag,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BondingSystem:
        return cls(
            shared_electrons=NonNegative.from_dict(data["shared_electrons"]),
            member_atoms=frozenset(AtomId.from_dict(item) for item in data["member_atoms"]),
            member_edges=frozenset(Edge.from_dict(item) for item in data["member_edges"]),
            tag=data.get("tag"),
        )

    def pretty(self) -> str:
        from .pretty import pretty_text

        return pretty_text(self)

    def __str__(self) -> str:
        return self.pretty()


def mk_bonding_system(
    shared_electrons: NonNegative,
    member_edges: frozenset[Edge] | set[Edge],
    tag: str | None = None,
) -> BondingSystem:
    edges = frozenset(member_edges)
    atoms = frozenset(atom for edge in edges for atom in atoms_of_edge(edge))
    return BondingSystem(
        shared_electrons=shared_electrons,
        member_atoms=atoms,
        member_edges=edges,
        tag=tag,
    )
