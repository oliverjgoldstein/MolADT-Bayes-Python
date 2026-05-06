from __future__ import annotations

import json
from pathlib import Path

from moladt.cli import main
from moladt.io import molecule_to_json, read_sdf_record


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_parse_cli_is_minimal_by_default(capsys) -> None:
    result = main(["parse", str(PROJECT_ROOT / "molecules" / "benzene.sdf")])
    output = capsys.readouterr().out

    assert result == 0
    assert "Title:" in output
    assert "Molecule Report" in output
    assert "SMILES Stereochemistry" not in output
    assert "Properties:" not in output


def test_parse_cli_can_print_properties_on_request(capsys) -> None:
    result = main(["parse", "--properties", str(PROJECT_ROOT / "molecules" / "benzene.sdf")])
    output = capsys.readouterr().out

    assert result == 0
    assert "Properties:" in output
    assert "PUBCHEM_SMILES" in output


def test_to_json_cli_outputs_moladt_json(capsys) -> None:
    result = main(["to-json", str(PROJECT_ROOT / "molecules" / "benzene.sdf")])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert result == 0
    assert payload["atoms"]
    assert payload["local_bonds"]
    assert payload["systems"]


def test_to_python_cli_outputs_explicit_molecule_literal(capsys) -> None:
    result = main(["to-python", str(PROJECT_ROOT / "molecules" / "benzene.sdf"), "--name", "benzene_generated"])
    output = capsys.readouterr().out
    namespace: dict[str, object] = {}

    exec(output, namespace)

    assert result == 0
    assert "benzene_generated = Molecule(" in output
    assert "AtomId(1): atom(1, AtomicSymbol.C, 2.866, 1.000, 0.000)" in output
    assert "Edge(AtomId(1), AtomId(2))" in output
    assert "mk_bonding_system(" in output
    assert "for " not in output
    assert "range(" not in output
    assert "zip(" not in output
    assert "atom_map(" not in output
    assert "sigma_bonds(" not in output
    assert namespace["benzene_generated"].atoms


def test_to_example_cli_can_write_explicit_molecule_literal(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "benzene_generated.py"
    result = main(
        [
            "to-example",
            str(PROJECT_ROOT / "molecules" / "benzene.sdf"),
            "--name",
            "benzene_generated",
            "--output",
            str(output_path),
        ]
    )
    output = capsys.readouterr().out
    source = output_path.read_text(encoding="utf-8")

    assert result == 0
    assert output.strip() == str(output_path)
    assert "benzene_generated = Molecule(" in source
    assert "AtomId(1): atom(1, AtomicSymbol.C, 2.866, 1.000, 0.000)" in source
    assert "for " not in source
    assert "range(" not in source


def test_from_json_cli_round_trips_molecule(tmp_path: Path, capsys) -> None:
    record = read_sdf_record(PROJECT_ROOT / "molecules" / "benzene.sdf")
    json_path = tmp_path / "benzene.moladt.json"
    json_path.write_text(molecule_to_json(record.molecule), encoding="utf-8")

    result = main(["from-json", str(json_path)])
    output = capsys.readouterr().out

    assert result == 0
    assert "Molecule Report" in output
    assert "Edge Network" in output
