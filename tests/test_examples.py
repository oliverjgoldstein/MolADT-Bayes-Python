from __future__ import annotations

from moladt.chem.validate import validate_molecule
from moladt.examples import diborane_pretty, ferrocene_pretty


def test_diborane_constructs_a_valid_molecule() -> None:
    assert validate_molecule(diborane_pretty) == diborane_pretty


def test_ferrocene_constructs_a_valid_molecule() -> None:
    assert validate_molecule(ferrocene_pretty) == ferrocene_pretty

