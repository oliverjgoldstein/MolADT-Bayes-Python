from __future__ import annotations

import argparse
import keyword
from pathlib import Path

from .chem.molecule import Molecule
from .chem.molecule_ops import pretty_print_molecule
from .chem.validate import validate_molecule
from .examples import (
    MANUSCRIPT_EXAMPLES,
    benzene_pretty,
    diborane_pretty,
    ferrocene_pretty,
    get_manuscript_example,
    hydrogen,
    methane,
    morphine_pretty,
    oxygen,
    sodium_chloride,
    water,
)
from .io.molecule_json import molecule_from_json, molecule_to_json
from .io.python_literal import molecule_to_python_literal
from .io.sdf import read_sdf_record
from .io.smiles import molecule_to_smiles, parse_smiles
from .viewer import (
    molecule_viewer_uri,
    open_molecule_viewer,
    write_molecule_viewer_collection_html,
    write_molecule_viewer_html,
)


EXAMPLE_VIEWER_MOLECULES: dict[str, tuple[str, Molecule]] = {
    "benzene": ("Benzene", benzene_pretty),
    "diborane": ("Diborane (B2H6)", diborane_pretty),
    "ferrocene": ("Ferrocene (Fe(C5H5)2)", ferrocene_pretty),
    "hydrogen": ("Hydrogen", hydrogen),
    "methane": ("Methane", methane),
    "morphine": ("Morphine", morphine_pretty),
    "oxygen": ("Oxygen", oxygen),
    "sodium_chloride": ("Sodium chloride (NaCl)", sodium_chloride),
    "water": ("Water", water),
}

DEFAULT_VIEW_EXAMPLES = ("benzene", "diborane", "ferrocene", "morphine", "methane", "water", "sodium_chloride")


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

    to_python_parser = subparsers.add_parser("to-python", help="Convert an SDF molecule into an explicit Python Molecule literal")
    to_python_parser.add_argument("path")
    to_python_parser.add_argument("--name", default="molecule", help="Variable name to assign in the generated Python code")
    to_python_parser.add_argument("-o", "--output", help="Optional file to write instead of printing")

    to_example_parser = subparsers.add_parser("to-example", help="Alias for to-python")
    to_example_parser.add_argument("path")
    to_example_parser.add_argument("--name", default="molecule", help="Variable name to assign in the generated Python code")
    to_example_parser.add_argument("-o", "--output", help="Optional file to write instead of printing")

    from_json_parser = subparsers.add_parser("from-json", help="Load MolADT JSON and pretty-print the typed molecule")
    from_json_parser.add_argument("path")

    viewer_parser = subparsers.add_parser("view-html", help="Export an interactive 3D molecule viewer HTML file")
    viewer_parser.add_argument("path", nargs="+", help="Input MolADT JSON file(s)")
    viewer_parser.add_argument("-o", "--output", help="Output HTML path")
    viewer_parser.add_argument("--title", help="Viewer title")
    viewer_parser.add_argument(
        "--open-viewer",
        action="store_true",
        help="Open the written viewer HTML in the default browser.",
    )
    viewer_parser.add_argument(
        "--format",
        choices=("json",),
        help="Input format. The viewer path accepts MolADT JSON; convert SDF with to-json first.",
    )

    view_examples_parser = subparsers.add_parser(
        "view-examples",
        help="Export a multi-molecule viewer for built-in ADT examples",
    )
    view_examples_parser.add_argument(
        "names",
        nargs="*",
        choices=tuple(sorted(EXAMPLE_VIEWER_MOLECULES)),
        help=(
            "Built-in examples to include. Defaults to benzene, diborane, ferrocene, morphine, "
            "methane, water, sodium_chloride."
        ),
    )
    view_examples_parser.add_argument("-o", "--output", help="Output HTML path")
    view_examples_parser.add_argument("--title", help="Viewer title")
    view_examples_parser.add_argument(
        "--open-viewer",
        action="store_true",
        help="Open the written viewer HTML in the default browser.",
    )

    pretty_example_parser = subparsers.add_parser(
        "pretty-example",
        help="Render a manuscript-facing built-in example molecule",
    )
    pretty_example_parser.add_argument("name", choices=tuple(sorted(MANUSCRIPT_EXAMPLES)))
    pretty_example_parser.add_argument(
        "--viewer-output",
        help="Optional HTML viewer output path for this example.",
    )
    pretty_example_parser.add_argument(
        "--open-viewer",
        action="store_true",
        help="Also write and open this example in the default browser.",
    )

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
    if args.command in {"to-python", "to-example"}:
        return _handle_to_python(
            Path(args.path),
            variable_name=args.name,
            output_path=None if args.output is None else Path(args.output),
        )
    if args.command == "from-json":
        return _handle_from_json(Path(args.path))
    if args.command == "view-html":
        return _handle_view_html(
            [Path(path) for path in args.path],
            output_path=None if args.output is None else Path(args.output),
            title=args.title,
            input_format=args.format,
            open_viewer=args.open_viewer,
        )
    if args.command == "view-examples":
        return _handle_view_examples(
            tuple(args.names or DEFAULT_VIEW_EXAMPLES),
            output_path=None if args.output is None else Path(args.output),
            title=args.title,
            open_viewer=args.open_viewer,
        )
    if args.command == "pretty-example":
        return _handle_pretty_example(
            args.name,
            viewer_output=None if args.viewer_output is None else Path(args.viewer_output),
            open_viewer=args.open_viewer,
        )
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


def _handle_to_python(path: Path, *, variable_name: str, output_path: Path | None = None) -> int:
    if not variable_name.isidentifier() or keyword.iskeyword(variable_name):
        raise ValueError("--name must be a valid Python identifier")
    record = read_sdf_record(path)
    validate_molecule(record.molecule)
    source = molecule_to_python_literal(record.molecule, variable_name=variable_name)
    if output_path is None:
        print(source, end="")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(source, encoding="utf-8")
        print(output_path)
    return 0


def _handle_from_json(path: Path) -> int:
    molecule = molecule_from_json(path.read_bytes())
    validate_molecule(molecule)
    print(pretty_print_molecule(molecule))
    return 0


def _handle_view_html(
    paths: list[Path],
    *,
    output_path: Path | None,
    title: str | None,
    input_format: str | None,
    open_viewer: bool = False,
) -> int:
    entries = [_read_viewer_entry(path, input_format=input_format) for path in paths]
    for _, molecule in entries:
        validate_molecule(molecule)
    if len(entries) == 1:
        default_title = entries[0][0]
        resolved_output = output_path or paths[0].with_suffix(".viewer.html")
        written = write_molecule_viewer_html(entries[0][1], resolved_output, title=title or default_title)
    else:
        resolved_output = output_path or Path("results") / "viewer" / "molecules.viewer.html"
        written = write_molecule_viewer_collection_html(
            entries,
            resolved_output,
            title=title or f"{len(entries)} MolADT molecules",
        )
    _print_viewer_location(written)
    if open_viewer:
        _open_viewer_with_fallback(written)
    return 0


def _handle_view_examples(
    names: tuple[str, ...],
    *,
    output_path: Path | None,
    title: str | None,
    open_viewer: bool = False,
) -> int:
    entries = tuple(EXAMPLE_VIEWER_MOLECULES[name] for name in names)
    for _, molecule in entries:
        validate_molecule(molecule)
    resolved_output = output_path or Path("results") / "viewer" / "examples.viewer.html"
    written = write_molecule_viewer_collection_html(
        entries,
        resolved_output,
        title=title or "MolADT example molecules",
    )
    _print_viewer_location(written)
    if open_viewer:
        _open_viewer_with_fallback(written)
    return 0


def _read_viewer_entry(path: Path, *, input_format: str | None) -> tuple[str, Molecule]:
    if input_format not in (None, "json"):
        raise ValueError("view-html accepts MolADT JSON only; convert SDF with to-json first")
    if path.suffix.lower() != ".json":
        raise ValueError("view-html accepts MolADT JSON files only; convert SDF with to-json first")
    return path.stem, molecule_from_json(path.read_bytes())


def _handle_pretty_example(name: str, *, viewer_output: Path | None = None, open_viewer: bool = False) -> int:
    example = get_manuscript_example(name)
    validate_molecule(example.molecule)
    print(example.render())
    if open_viewer or viewer_output is not None:
        output_path = viewer_output or Path("results") / "viewer" / f"{example.slug}.viewer.html"
        written = write_molecule_viewer_html(example.molecule, output_path, title=example.title)
        print(f"Viewer: {written}")
        print(f"Viewer URL: {molecule_viewer_uri(written)}")
        if open_viewer:
            _open_viewer_with_fallback(written)
    return 0


def _print_viewer_location(path: Path) -> None:
    print(path)
    print(f"Viewer URL: {molecule_viewer_uri(path)}")


def _open_viewer_with_fallback(path: Path) -> None:
    uri = molecule_viewer_uri(path)
    if open_molecule_viewer(path):
        print(f"Opened viewer: {uri}")
    else:
        print(f"Viewer open request failed; open this URL manually: {uri}")


if __name__ == "__main__":
    raise SystemExit(main())
