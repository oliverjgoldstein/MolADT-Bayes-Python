from __future__ import annotations

from .coordinate import Angstrom, mk_angstrom
from .molecule import AtomicSymbol, ElementAttributes
from . import orbital


def normalize_symbols(symbol_a: AtomicSymbol, symbol_b: AtomicSymbol) -> tuple[AtomicSymbol, AtomicSymbol]:
    return (symbol_a, symbol_b) if symbol_a.value <= symbol_b.value else (symbol_b, symbol_a)


_BOND_LENGTHS: dict[tuple[int, AtomicSymbol, AtomicSymbol], Angstrom] = {
    (1, AtomicSymbol.H, AtomicSymbol.H): mk_angstrom(0.74),
    (1, AtomicSymbol.H, AtomicSymbol.C): mk_angstrom(1.09),
    (1, AtomicSymbol.H, AtomicSymbol.N): mk_angstrom(1.01),
    (1, AtomicSymbol.H, AtomicSymbol.O): mk_angstrom(0.96),
    (1, AtomicSymbol.H, AtomicSymbol.Fe): mk_angstrom(1.52),
    (1, AtomicSymbol.H, AtomicSymbol.B): mk_angstrom(1.19),
    (1, AtomicSymbol.C, AtomicSymbol.C): mk_angstrom(1.54),
    (1, AtomicSymbol.C, AtomicSymbol.N): mk_angstrom(1.47),
    (1, AtomicSymbol.C, AtomicSymbol.O): mk_angstrom(1.43),
    (1, AtomicSymbol.C, AtomicSymbol.Fe): mk_angstrom(1.84),
    (1, AtomicSymbol.C, AtomicSymbol.B): mk_angstrom(1.55),
    (1, AtomicSymbol.N, AtomicSymbol.N): mk_angstrom(1.45),
    (1, AtomicSymbol.N, AtomicSymbol.O): mk_angstrom(1.40),
    (1, AtomicSymbol.N, AtomicSymbol.Fe): mk_angstrom(1.76),
    (1, AtomicSymbol.N, AtomicSymbol.B): mk_angstrom(1.55),
    (1, AtomicSymbol.O, AtomicSymbol.O): mk_angstrom(1.48),
    (1, AtomicSymbol.O, AtomicSymbol.Fe): mk_angstrom(1.70),
    (1, AtomicSymbol.O, AtomicSymbol.B): mk_angstrom(1.49),
    (1, AtomicSymbol.Fe, AtomicSymbol.Fe): mk_angstrom(2.48),
    (1, AtomicSymbol.Fe, AtomicSymbol.B): mk_angstrom(2.03),
    (1, AtomicSymbol.B, AtomicSymbol.B): mk_angstrom(1.59),
    (2, AtomicSymbol.H, AtomicSymbol.H): mk_angstrom(0.74),
    (2, AtomicSymbol.H, AtomicSymbol.C): mk_angstrom(1.06),
    (2, AtomicSymbol.H, AtomicSymbol.N): mk_angstrom(1.01),
    (2, AtomicSymbol.H, AtomicSymbol.O): mk_angstrom(0.96),
    (2, AtomicSymbol.H, AtomicSymbol.Fe): mk_angstrom(1.52),
    (2, AtomicSymbol.H, AtomicSymbol.B): mk_angstrom(1.19),
    (2, AtomicSymbol.C, AtomicSymbol.C): mk_angstrom(1.34),
    (2, AtomicSymbol.C, AtomicSymbol.N): mk_angstrom(1.27),
    (2, AtomicSymbol.C, AtomicSymbol.O): mk_angstrom(1.20),
    (2, AtomicSymbol.C, AtomicSymbol.Fe): mk_angstrom(1.64),
    (2, AtomicSymbol.C, AtomicSymbol.B): mk_angstrom(1.37),
    (2, AtomicSymbol.N, AtomicSymbol.N): mk_angstrom(1.25),
    (2, AtomicSymbol.N, AtomicSymbol.O): mk_angstrom(1.20),
    (2, AtomicSymbol.N, AtomicSymbol.Fe): mk_angstrom(1.64),
    (2, AtomicSymbol.N, AtomicSymbol.B): mk_angstrom(1.33),
    (2, AtomicSymbol.O, AtomicSymbol.O): mk_angstrom(1.21),
    (2, AtomicSymbol.O, AtomicSymbol.Fe): mk_angstrom(1.58),
    (2, AtomicSymbol.O, AtomicSymbol.B): mk_angstrom(1.26),
    (2, AtomicSymbol.Fe, AtomicSymbol.Fe): mk_angstrom(2.26),
    (2, AtomicSymbol.Fe, AtomicSymbol.B): mk_angstrom(1.89),
    (2, AtomicSymbol.B, AtomicSymbol.B): mk_angstrom(1.59),
    (3, AtomicSymbol.H, AtomicSymbol.H): mk_angstrom(0.74),
    (3, AtomicSymbol.H, AtomicSymbol.C): mk_angstrom(1.06),
    (3, AtomicSymbol.H, AtomicSymbol.N): mk_angstrom(1.01),
    (3, AtomicSymbol.H, AtomicSymbol.O): mk_angstrom(0.96),
    (3, AtomicSymbol.H, AtomicSymbol.Fe): mk_angstrom(1.52),
    (3, AtomicSymbol.H, AtomicSymbol.B): mk_angstrom(1.19),
    (3, AtomicSymbol.C, AtomicSymbol.C): mk_angstrom(1.20),
    (3, AtomicSymbol.C, AtomicSymbol.N): mk_angstrom(1.14),
    (3, AtomicSymbol.C, AtomicSymbol.O): mk_angstrom(1.13),
    (3, AtomicSymbol.C, AtomicSymbol.Fe): mk_angstrom(1.44),
    (3, AtomicSymbol.C, AtomicSymbol.B): mk_angstrom(1.19),
    (3, AtomicSymbol.N, AtomicSymbol.N): mk_angstrom(1.10),
    (3, AtomicSymbol.N, AtomicSymbol.O): mk_angstrom(1.06),
    (3, AtomicSymbol.N, AtomicSymbol.Fe): mk_angstrom(1.50),
    (3, AtomicSymbol.N, AtomicSymbol.B): mk_angstrom(1.20),
    (3, AtomicSymbol.O, AtomicSymbol.O): mk_angstrom(1.21),
    (3, AtomicSymbol.O, AtomicSymbol.Fe): mk_angstrom(1.58),
    (3, AtomicSymbol.O, AtomicSymbol.B): mk_angstrom(1.20),
    (3, AtomicSymbol.Fe, AtomicSymbol.Fe): mk_angstrom(2.26),
    (3, AtomicSymbol.Fe, AtomicSymbol.B): mk_angstrom(1.89),
    (3, AtomicSymbol.B, AtomicSymbol.B): mk_angstrom(1.59),
}


_NOMINAL_VALENCE: dict[AtomicSymbol, tuple[int, int]] = {
    AtomicSymbol.H: (2, 2),
    AtomicSymbol.C: (8, 10),
    AtomicSymbol.N: (6, 6),
    AtomicSymbol.O: (4, 4),
    AtomicSymbol.F: (2, 2),
    AtomicSymbol.P: (6, 10),
    AtomicSymbol.S: (4, 12),
    AtomicSymbol.Cl: (2, 2),
    AtomicSymbol.Br: (2, 2),
    AtomicSymbol.B: (6, 8),
    AtomicSymbol.Fe: (0, 12),
    AtomicSymbol.I: (2, 2),
    AtomicSymbol.Na: (2, 2),
}


_ELEMENT_ATTRIBUTES: dict[AtomicSymbol, ElementAttributes] = {
    AtomicSymbol.O: ElementAttributes(AtomicSymbol.O, 8, 15.999),
    AtomicSymbol.H: ElementAttributes(AtomicSymbol.H, 1, 1.008),
    AtomicSymbol.N: ElementAttributes(AtomicSymbol.N, 7, 14.007),
    AtomicSymbol.C: ElementAttributes(AtomicSymbol.C, 6, 12.011),
    AtomicSymbol.B: ElementAttributes(AtomicSymbol.B, 5, 10.811),
    AtomicSymbol.Fe: ElementAttributes(AtomicSymbol.Fe, 26, 55.845),
    AtomicSymbol.F: ElementAttributes(AtomicSymbol.F, 9, 18.998),
    AtomicSymbol.Cl: ElementAttributes(AtomicSymbol.Cl, 17, 35.453),
    AtomicSymbol.S: ElementAttributes(AtomicSymbol.S, 16, 32.065),
    AtomicSymbol.Br: ElementAttributes(AtomicSymbol.Br, 35, 79.904),
    AtomicSymbol.P: ElementAttributes(AtomicSymbol.P, 15, 30.974),
    AtomicSymbol.I: ElementAttributes(AtomicSymbol.I, 53, 126.904),
    AtomicSymbol.Na: ElementAttributes(AtomicSymbol.Na, 11, 22.990),
}


_ELEMENT_SHELLS = {
    AtomicSymbol.O: orbital.OXYGEN,
    AtomicSymbol.H: orbital.HYDROGEN,
    AtomicSymbol.N: orbital.NITROGEN,
    AtomicSymbol.C: orbital.CARBON,
    AtomicSymbol.B: orbital.BORON,
    AtomicSymbol.Fe: orbital.IRON,
    AtomicSymbol.F: orbital.FLUORINE,
    AtomicSymbol.Cl: orbital.CHLORINE,
    AtomicSymbol.S: orbital.SULFUR,
    AtomicSymbol.Br: orbital.BROMINE,
    AtomicSymbol.P: orbital.PHOSPHORUS,
    AtomicSymbol.I: orbital.IODINE,
    AtomicSymbol.Na: orbital.SODIUM,
}


def equilibrium_bond_length(
    bond_order: int,
    symbol_a: AtomicSymbol,
    symbol_b: AtomicSymbol,
) -> Angstrom | None:
    normalized = normalize_symbols(symbol_a, symbol_b)
    return _BOND_LENGTHS.get((bond_order, normalized[0], normalized[1]))


def nominal_valence(symbol: AtomicSymbol) -> tuple[int, int]:
    return _NOMINAL_VALENCE[symbol]


def get_max_bonds_symbol(symbol: AtomicSymbol) -> float:
    _, max_electrons = nominal_valence(symbol)
    return max_electrons / 2.0


def element_attributes(symbol: AtomicSymbol) -> ElementAttributes:
    return _ELEMENT_ATTRIBUTES[symbol]


def element_shells(symbol: AtomicSymbol) -> orbital.Shells:
    return _ELEMENT_SHELLS[symbol]
