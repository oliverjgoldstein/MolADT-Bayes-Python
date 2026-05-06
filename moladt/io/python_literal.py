from __future__ import annotations

import json

from ..chem.molecule import Molecule


def molecule_to_python_literal(molecule: Molecule, *, variable_name: str = "molecule") -> str:
    """Render a Molecule as explicit Python ADT construction code."""

    lines: list[str] = [
        "from __future__ import annotations",
        "",
        "from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system",
    ]
    if molecule.smiles_stereochemistry.atom_stereo or molecule.smiles_stereochemistry.bond_stereo:
        lines.extend(
            [
                "from moladt.chem.molecule import (",
                "    AtomicSymbol,",
                "    Molecule,",
                "    SmilesAtomStereo,",
                "    SmilesAtomStereoClass,",
                "    SmilesBondStereo,",
                "    SmilesBondStereoDirection,",
                "    SmilesStereochemistry,",
                ")",
            ]
        )
    else:
        lines.append("from moladt.chem.molecule import AtomicSymbol, Molecule")
    lines.extend(
        [
            "from moladt.examples._literal import atom",
            "",
            "",
            f"{variable_name} = Molecule(",
            "    atoms={",
        ]
    )
    for atom_id, atom_value in sorted(molecule.atoms.items(), key=lambda item: item[0].value):
        charge_arg = "" if atom_value.formal_charge == 0 else f", formal_charge={atom_value.formal_charge}"
        lines.append(
            "        "
            f"AtomId({atom_id.value}): atom({atom_id.value}, AtomicSymbol.{atom_value.attributes.symbol.name}, "
            f"{_format_float(atom_value.coordinate.x.value)}, "
            f"{_format_float(atom_value.coordinate.y.value)}, "
            f"{_format_float(atom_value.coordinate.z.value)}{charge_arg}),"
        )
    lines.append("    },")
    if molecule.systems:
        lines.append("    systems=(")
        for system_id, system in sorted(molecule.systems, key=lambda item: item[0].value):
            lines.extend(
                [
                    "        (",
                    f"            SystemId({system_id.value}),",
                    "            mk_bonding_system(",
                    f"                NonNegative({system.shared_electrons.value}),",
                    "                frozenset(",
                ]
            )
            if system.member_edges:
                lines.append("                    {")
                for edge in sorted(system.member_edges):
                    lines.append(f"                        Edge(AtomId({edge.a.value}), AtomId({edge.b.value})),")
                lines.append("                    }")
            else:
                lines.append("                    {}")
            lines.extend(
                [
                    "                ),",
                    f"                {_format_tag(system.tag)},",
                    "            ),",
                    "        ),",
                ]
            )
        lines.append("    ),")
    else:
        lines.append("    systems=(),")
    if molecule.smiles_stereochemistry.atom_stereo or molecule.smiles_stereochemistry.bond_stereo:
        lines.extend(_render_stereochemistry(molecule))
    lines.append(")")
    return "\n".join(lines) + "\n"


def _render_stereochemistry(molecule: Molecule) -> list[str]:
    lines = [
        "    smiles_stereochemistry=SmilesStereochemistry(",
        "        atom_stereo=(",
    ]
    for atom_item in molecule.smiles_stereochemistry.atom_stereo:
        lines.append(
            "            "
            f"SmilesAtomStereo(AtomId({atom_item.center.value}), "
            f"SmilesAtomStereoClass.{atom_item.stereo_class.name}, "
            f"{atom_item.configuration}, "
            f"{atom_item.token!r}),"
        )
    lines.extend(
        [
            "        ),",
            "        bond_stereo=(",
        ]
    )
    for bond_item in molecule.smiles_stereochemistry.bond_stereo:
        lines.append(
            "            "
            f"SmilesBondStereo(AtomId({bond_item.start_atom.value}), "
            f"AtomId({bond_item.end_atom.value}), "
            f"SmilesBondStereoDirection.{bond_item.direction.name}),"
        )
    lines.extend(
        [
            "        ),",
            "    ),",
        ]
    )
    return lines


def _format_float(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:.3f}"


def _format_tag(tag: str | None) -> str:
    return "None" if tag is None else json.dumps(tag)


__all__ = ["molecule_to_python_literal"]
