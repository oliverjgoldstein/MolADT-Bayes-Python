from __future__ import annotations

import argparse
from pathlib import Path

from .chem.molecule_ops import pretty_print_molecule
from .chem.validate import validate_molecule
from .examples import MANUSCRIPT_EXAMPLES, get_manuscript_example
from .io.molecule_json import molecule_from_json, molecule_to_json
from .io.sdf import read_sdf_record
from .io.smiles import molecule_to_smiles, parse_smiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m moladt.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse and validate an SDF file")
    parse_parser.add_argument("path")
    parse_parser.add_argument(
        "--properties",
        action="store_true",
        help="Print the raw SDF properties block after the molecule report",
    )

    parse_smiles_parser = subparsers.add_parser("parse-smiles", help="Parse and validate a SMILES string")
    parse_smiles_parser.add_argument("smiles")

    to_smiles_parser = subparsers.add_parser("to-smiles", help="Render an SDF molecule as a SMILES string")
    to_smiles_parser.add_argument("path")

    to_json_parser = subparsers.add_parser("to-json", help="Convert an SDF molecule into MolADT JSON")
    to_json_parser.add_argument("path")

    from_json_parser = subparsers.add_parser("from-json", help="Load MolADT JSON and pretty-print the typed molecule")
    from_json_parser.add_argument("path")

    pretty_example_parser = subparsers.add_parser(
        "pretty-example",
        help="Render a manuscript-facing built-in example molecule",
    )
    pretty_example_parser.add_argument("name", choices=tuple(sorted(MANUSCRIPT_EXAMPLES)))

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "parse":
        return _handle_parse(Path(args.path), include_properties=args.properties)
    if args.command == "parse-smiles":
        return _handle_parse_smiles(args.smiles)
    if args.command == "to-smiles":
        return _handle_to_smiles(Path(args.path))
    if args.command == "to-json":
        return _handle_to_json(Path(args.path))
    if args.command == "from-json":
        return _handle_from_json(Path(args.path))
    if args.command == "pretty-example":
        return _handle_pretty_example(args.name)
    raise RuntimeError(f"Unsupported command: {args.command}")


def _handle_parse(path: Path, *, include_properties: bool = False) -> int:
    record = read_sdf_record(path)
    validate_molecule(record.molecule)
    print(f"Title: {record.title or '(blank)'}")
    print(pretty_print_molecule(record.molecule))
    if include_properties and record.properties:
        print("Properties:")
        for key in sorted(record.properties):
            print(f"  {key}: {record.properties[key]}")
    return 0


def _handle_parse_smiles(smiles_text: str) -> int:
    molecule = parse_smiles(smiles_text)
    validate_molecule(molecule)
    print(pretty_print_molecule(molecule))
    return 0


def _handle_to_smiles(path: Path) -> int:
    record = read_sdf_record(path)
    validate_molecule(record.molecule)
    print(molecule_to_smiles(record.molecule))
    return 0


def _handle_to_json(path: Path) -> int:
    record = read_sdf_record(path)
    validate_molecule(record.molecule)
    print(molecule_to_json(record.molecule))
    return 0


def _handle_from_json(path: Path) -> int:
    molecule = molecule_from_json(path.read_bytes())
    validate_molecule(molecule)
    print(pretty_print_molecule(molecule))
    return 0


def _handle_pretty_example(name: str) -> int:
    example = get_manuscript_example(name)
    validate_molecule(example.molecule)
    print(example.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
