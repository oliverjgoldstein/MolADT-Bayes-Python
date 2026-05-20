from __future__ import annotations

from moladt.chem.constants import ALL_ATOMIC_SYMBOLS, element_attributes
from moladt.chem.molecule import AtomicSymbol
from moladt.io.sdf import parse_sdf


def test_element_table_covers_all_official_elements() -> None:
    assert len(ALL_ATOMIC_SYMBOLS) == 118
    assert set(ALL_ATOMIC_SYMBOLS) == set(AtomicSymbol)
    assert element_attributes(AtomicSymbol.Og).atomic_number == 118
    assert element_attributes(AtomicSymbol.Tc).atomic_weight == 97.0
    assert element_attributes(AtomicSymbol.Zr).atomic_weight == 91.222


def test_sdf_parser_accepts_all_official_element_symbols() -> None:
    atom_lines = [
        f"{float(index):10.4f}{0.0:10.4f}{0.0:10.4f} {symbol.value:<3} 0  0  0  0  0  0  0  0  0  0  0  0"
        for index, symbol in enumerate(ALL_ATOMIC_SYMBOLS, start=1)
    ]
    sdf = "\n".join(
        [
            "all-elements",
            "MolADT",
            "generated",
            f"{len(ALL_ATOMIC_SYMBOLS):>3}{0:>3}  0  0  0  0  0  0  0  0  0  0 V2000",
            *atom_lines,
            "M  END",
            "$$$$",
        ]
    )

    molecule = parse_sdf(sdf)

    assert len(molecule.atoms) == 118
    assert molecule.atoms[next(reversed(molecule.atoms))].attributes.symbol is AtomicSymbol.Og
