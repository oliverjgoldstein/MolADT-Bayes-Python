from __future__ import annotations

from pathlib import Path

from ..chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from ..chem.molecule import Molecule
from ..io.sdf import read_sdf_record


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


_PROJECT_ROOT = Path(__file__).resolve().parents[2]

psilocybin_pretty = Molecule(
    atoms=read_sdf_record(_PROJECT_ROOT / "molecules" / "psilocybin.sdf").molecule.atoms,
    local_bonds=read_sdf_record(_PROJECT_ROOT / "molecules" / "psilocybin.sdf").molecule.local_bonds,
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
