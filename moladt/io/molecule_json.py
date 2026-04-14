from __future__ import annotations

import json
from typing import Any, Protocol, cast

from ..chem.coordinate import Coordinate
from ..chem.constants import element_shells
from ..chem.dietz import AtomId, BondingSystem, Edge, SystemId
from ..chem.molecule import (
    Atom,
    AtomicSymbol,
    ElementAttributes,
    Molecule,
    SmilesAtomStereo,
    SmilesAtomStereoClass,
    SmilesBondStereo,
    SmilesBondStereoDirection,
    SmilesStereochemistry,
)
from ..chem.orbital import D, F, P, So, Orbital, Shell, SubShell, pure_orbital_from_dict


class _OrjsonModule(Protocol):
    OPT_INDENT_2: int
    OPT_SORT_KEYS: int

    def dumps(self, obj: Any, /, *, option: int = 0) -> bytes: ...

    def loads(self, obj: str | bytes | bytearray | memoryview, /) -> Any: ...


try:
    import orjson as _orjson  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional wheel is missing locally.
    orjson: _OrjsonModule | None = None
else:
    orjson = cast(_OrjsonModule, _orjson)


def smiles_atom_stereo_to_dict(stereo: SmilesAtomStereo) -> dict[str, Any]:
    return {
        "center": stereo.center.to_dict(),
        "stereo_class": stereo.stereo_class.value,
        "configuration": stereo.configuration,
        "token": stereo.token,
    }


def smiles_atom_stereo_from_dict(data: dict[str, Any]) -> SmilesAtomStereo:
    return SmilesAtomStereo(
        center=AtomId.from_dict(data["center"]),
        stereo_class=SmilesAtomStereoClass(str(data["stereo_class"])),
        configuration=int(data["configuration"]),
        token=str(data["token"]),
    )


def smiles_bond_stereo_to_dict(stereo: SmilesBondStereo) -> dict[str, Any]:
    return {
        "start_atom": stereo.start_atom.to_dict(),
        "end_atom": stereo.end_atom.to_dict(),
        "direction": stereo.direction.value,
    }


def smiles_bond_stereo_from_dict(data: dict[str, Any]) -> SmilesBondStereo:
    return SmilesBondStereo(
        start_atom=AtomId.from_dict(data["start_atom"]),
        end_atom=AtomId.from_dict(data["end_atom"]),
        direction=SmilesBondStereoDirection(str(data["direction"])),
    )


def smiles_stereochemistry_to_dict(stereochemistry: SmilesStereochemistry) -> dict[str, Any]:
    return {
        "atom_stereo": [smiles_atom_stereo_to_dict(item) for item in stereochemistry.atom_stereo],
        "bond_stereo": [smiles_bond_stereo_to_dict(item) for item in stereochemistry.bond_stereo],
    }


def smiles_stereochemistry_from_dict(data: dict[str, Any]) -> SmilesStereochemistry:
    return SmilesStereochemistry(
        atom_stereo=tuple(smiles_atom_stereo_from_dict(item) for item in data.get("atom_stereo", [])),
        bond_stereo=tuple(smiles_bond_stereo_from_dict(item) for item in data.get("bond_stereo", [])),
    )


def element_attributes_to_dict(attributes: ElementAttributes) -> dict[str, Any]:
    return {
        "symbol": attributes.symbol.value,
        "atomic_number": attributes.atomic_number,
        "atomic_weight": attributes.atomic_weight,
    }


def element_attributes_from_dict(data: dict[str, Any]) -> ElementAttributes:
    return ElementAttributes(
        symbol=AtomicSymbol(str(data["symbol"])),
        atomic_number=int(data["atomic_number"]),
        atomic_weight=float(data["atomic_weight"]),
    )


def atom_to_dict(atom: Atom) -> dict[str, Any]:
    return {
        "atom_id": atom.atom_id.to_dict(),
        "attributes": element_attributes_to_dict(atom.attributes),
        "coordinate": atom.coordinate.to_dict(),
        "shells": [shell.to_dict() for shell in atom.shells],
        "formal_charge": atom.formal_charge,
    }


def atom_from_dict(data: dict[str, Any]) -> Atom:
    attributes = element_attributes_from_dict(data["attributes"])
    shells_data = data.get("shells", [])
    shells = tuple(_shell_from_dict(item) for item in shells_data) if shells_data else element_shells(attributes.symbol)
    return Atom(
        atom_id=AtomId.from_dict(data["atom_id"]),
        attributes=attributes,
        coordinate=Coordinate.from_dict(data["coordinate"]),
        shells=shells,
        formal_charge=int(data.get("formal_charge", 0)),
    )


def molecule_to_dict(molecule: Molecule) -> dict[str, Any]:
    return {
        "atoms": [
            {"atom_id": atom_id.to_dict(), "atom": atom_to_dict(atom)}
            for atom_id, atom in molecule.atoms.items()
        ],
        "local_bonds": [edge.to_dict() for edge in sorted(molecule.local_bonds)],
        "systems": [
            {"system_id": system_id.to_dict(), "bonding_system": bonding_system.to_dict()}
            for system_id, bonding_system in molecule.systems
        ],
        "smiles_stereochemistry": smiles_stereochemistry_to_dict(molecule.smiles_stereochemistry),
    }


def molecule_from_dict(data: dict[str, Any]) -> Molecule:
    atoms = {
        AtomId.from_dict(item["atom_id"]): atom_from_dict(item["atom"])
        for item in data["atoms"]
    }
    local_bonds = frozenset(Edge.from_dict(item) for item in data["local_bonds"])
    systems = tuple(
        (SystemId.from_dict(item["system_id"]), BondingSystem.from_dict(item["bonding_system"]))
        for item in data["systems"]
    )
    smiles_stereochemistry = smiles_stereochemistry_from_dict(data.get("smiles_stereochemistry", {}))
    return Molecule(
        atoms=atoms,
        local_bonds=local_bonds,
        systems=systems,
        smiles_stereochemistry=smiles_stereochemistry,
    )


def molecule_to_json(molecule: Molecule) -> str:
    payload = molecule_to_dict(molecule)
    if orjson is not None:
        return orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode("utf-8")
    return json.dumps(payload, indent=2, sort_keys=True)


def molecule_to_json_bytes(molecule: Molecule) -> bytes:
    payload = molecule_to_dict(molecule)
    if orjson is not None:
        return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def molecule_from_json(payload: str | bytes) -> Molecule:
    if orjson is not None:
        return molecule_from_dict(orjson.loads(payload))
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return molecule_from_dict(json.loads(payload))


def _orbital_type_from_value(value: str) -> So | P | D | F:
    for orbital_enum in (So, P, D, F):
        try:
            return orbital_enum(value)
        except ValueError:
            continue
    raise ValueError(f"Unknown orbital type: {value}")


def _orbital_from_dict(data: dict[str, Any]) -> Orbital[Any]:
    orientation_data = data.get("orientation")
    hybrid_data = data.get("hybrid_components")
    hybrid_components = None
    if hybrid_data is not None:
        hybrid_components = tuple(
            (float(item["weight"]), pure_orbital_from_dict(item["pure_orbital"]))
            for item in hybrid_data
        )
    return Orbital(
        orbital_type=_orbital_type_from_value(str(data["orbital_type"])),
        electron_count=int(data["electron_count"]),
        orientation=None if orientation_data is None else Coordinate.from_dict(orientation_data),
        hybrid_components=hybrid_components,
    )


def _subshell_from_dict(data: dict[str, Any]) -> SubShell[Any]:
    return SubShell(orbitals=tuple(_orbital_from_dict(item) for item in data["orbitals"]))


def _shell_from_dict(data: dict[str, Any]) -> Shell:
    return Shell(
        principal_quantum_number=int(data["principal_quantum_number"]),
        s_subshell=None if data.get("s_subshell") is None else _subshell_from_dict(data["s_subshell"]),
        p_subshell=None if data.get("p_subshell") is None else _subshell_from_dict(data["p_subshell"]),
        d_subshell=None if data.get("d_subshell") is None else _subshell_from_dict(data["d_subshell"]),
        f_subshell=None if data.get("f_subshell") is None else _subshell_from_dict(data["f_subshell"]),
    )


__all__ = [
    "atom_from_dict",
    "atom_to_dict",
    "element_attributes_from_dict",
    "element_attributes_to_dict",
    "molecule_from_dict",
    "molecule_from_json",
    "molecule_to_dict",
    "molecule_to_json",
    "molecule_to_json_bytes",
    "smiles_atom_stereo_from_dict",
    "smiles_atom_stereo_to_dict",
    "smiles_bond_stereo_from_dict",
    "smiles_bond_stereo_to_dict",
    "smiles_stereochemistry_from_dict",
    "smiles_stereochemistry_to_dict",
]
