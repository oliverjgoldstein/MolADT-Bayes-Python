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


def test_from_json_cli_round_trips_molecule(tmp_path: Path, capsys) -> None:
    record = read_sdf_record(PROJECT_ROOT / "molecules" / "benzene.sdf")
    json_path = tmp_path / "benzene.moladt.json"
    json_path.write_text(molecule_to_json(record.molecule), encoding="utf-8")

    result = main(["from-json", str(json_path)])
    output = capsys.readouterr().out

    assert result == 0
    assert "Molecule Report" in output
    assert "Sigma Network" in output
