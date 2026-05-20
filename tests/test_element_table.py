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


def test_audited_default_orbitals_match_neutral_atomic_numbers() -> None:
    audited = [
        (symbol, element_attributes(symbol))
        for symbol in ALL_ATOMIC_SYMBOLS
        if element_attributes(symbol).shells is not None
    ]

    assert audited
    for symbol, attributes in audited:
        assert _shell_electron_count(attributes.shells) == attributes.atomic_number, symbol


def test_representative_orbital_occupancy_signatures_are_stable() -> None:
    assert "2p110" in _shell_signature(element_attributes(AtomicSymbol.C).shells)
    assert "3d21111" in _shell_signature(element_attributes(AtomicSymbol.Fe).shells)
    assert "3p221" in _shell_signature(element_attributes(AtomicSymbol.Cl).shells)
    assert "4p221" in _shell_signature(element_attributes(AtomicSymbol.Br).shells)
    assert "5p221" in _shell_signature(element_attributes(AtomicSymbol.I).shells)


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


def _shell_electron_count(shells) -> int:
    if shells is None:
        return 0
    return sum(
        orbital.electron_count
        for shell in shells
        for subshell in (shell.s_subshell, shell.p_subshell, shell.d_subshell, shell.f_subshell)
        if subshell is not None
        for orbital in subshell.orbitals
    )


def _shell_signature(shells) -> str:
    if shells is None:
        return ""
    parts: list[str] = []
    for shell in shells:
        for label, subshell in (
            ("s", shell.s_subshell),
            ("p", shell.p_subshell),
            ("d", shell.d_subshell),
            ("f", shell.f_subshell),
        ):
            if subshell is not None:
                parts.append(
                    f"{shell.principal_quantum_number}{label}"
                    + "".join(str(orbital.electron_count) for orbital in subshell.orbitals)
                )
    return ".".join(parts)
