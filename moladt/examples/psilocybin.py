from __future__ import annotations

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import AtomicSymbol, Molecule
from ._literal import atom, bond


PSILOCYBIN_PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/compound/10624#section=Structures"
PSILOCYBIN_PUBCHEM_SMILES = "CN(C)CCC1=CNC2=C1C(=CC=C2)OP(=O)(O)O"

# Connectivity follows the PubChem CID 10624 canonical structure:
# [3-[2-(dimethylamino)ethyl]-1H-indol-4-yl] dihydrogen phosphate
#
# The Dietz view used here is inferred from that classical boundary structure:
# - one fused 10-electron indole pi system over the bicyclic aromatic core
# - one 2-electron phosphoryl system on the P=O edge
# - the dimethylaminoethyl side chain and O-phosphate linkage remain ordinary
#   sigma edges
#
_INDOLE_PI_EDGES = (
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 6),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 9),
)

_PHOSPHORYL_EDGE = (
    (16, 17),
)

def _edge_from_index_pair(atom_pair: tuple[int, int]) -> Edge:
    return Edge(AtomId(atom_pair[0]), AtomId(atom_pair[1]))

psilocybin_pretty = Molecule(
    atoms={
        atom.atom_id: atom
        for atom in (
            atom(1, AtomicSymbol.C, -5.400, 1.200, 0.200),
            atom(2, AtomicSymbol.N, -4.200, 0.500, 0.000),
            atom(3, AtomicSymbol.C, -4.300, -0.900, -0.200),
            atom(4, AtomicSymbol.C, -2.900, 1.200, 0.000),
            atom(5, AtomicSymbol.C, -1.700, 0.500, 0.000),
            atom(6, AtomicSymbol.C, -0.500, 1.100, 0.100),
            atom(7, AtomicSymbol.C, 0.700, 0.500, 0.100),
            atom(8, AtomicSymbol.N, 1.800, 1.200, 0.000),
            atom(9, AtomicSymbol.C, 3.000, 0.500, 0.000),
            atom(10, AtomicSymbol.C, 2.900, -0.900, 0.100),
            atom(11, AtomicSymbol.C, 1.700, -1.500, 0.200),
            atom(12, AtomicSymbol.C, 0.500, -0.900, 0.200),
            atom(13, AtomicSymbol.C, -0.700, -1.500, 0.200),
            atom(14, AtomicSymbol.C, -1.800, -0.900, 0.100),
            atom(15, AtomicSymbol.O, 1.600, -2.900, 0.300),
            atom(16, AtomicSymbol.P, 2.400, -4.100, 0.600),
            atom(17, AtomicSymbol.O, 3.900, -4.000, 0.600),
            atom(18, AtomicSymbol.O, 1.800, -5.400, 0.700),
            atom(19, AtomicSymbol.O, 1.900, -3.700, 2.000),
        )
    },
    local_bonds=frozenset(
        {
            bond(1, 2),
            bond(2, 3),
            bond(2, 4),
            bond(4, 5),
            bond(5, 6),
            bond(6, 7),
            bond(6, 10),
            bond(7, 8),
            bond(8, 9),
            bond(9, 10),
            bond(9, 14),
            bond(10, 11),
            bond(11, 12),
            bond(11, 15),
            bond(12, 13),
            bond(13, 14),
            bond(15, 16),
            bond(16, 17),
            bond(16, 18),
            bond(16, 19),
        }
    ),
    systems=(
        (
            SystemId(1),
            mk_bonding_system(
                NonNegative(10),
                frozenset(_edge_from_index_pair(atom_pair) for atom_pair in _INDOLE_PI_EDGES),
                "indole_pi_system",
            ),
        ),
        (
            SystemId(2),
            mk_bonding_system(
                NonNegative(2),
                frozenset(_edge_from_index_pair(atom_pair) for atom_pair in _PHOSPHORYL_EDGE),
                "phosphoryl",
            ),
        ),
    ),
)
