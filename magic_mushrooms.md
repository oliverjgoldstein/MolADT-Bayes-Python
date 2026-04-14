# Magic Mushrooms

This note is about the molecular representation, not pharmacology.

## Psilocybin

Reference structure: [PubChem CID 10624](https://pubchem.ncbi.nlm.nih.gov/compound/10624#section=Structures)

- name: psilocybin
- PubChem formula: `C12H17N2O4P`
- PubChem boundary structure: `CN(C)CCC1=CNC2=C1C(=CC=C2)OP(=O)(O)O`
- 3D SDF conformer source: [`molecules/psilocybin.sdf`](molecules/psilocybin.sdf)
- built-in MolADT object: [`moladt/examples/psilocybin.py`](moladt/examples/psilocybin.py)
- CLI views: `./.venv/bin/python -m moladt.cli parse molecules/psilocybin.sdf` and `./.venv/bin/python -m moladt.cli pretty-example psilocybin`

### Inferred Dietz View

The PubChem page gives a classical indole phosphate structure. The Dietz reading used in this repo is:

- the dimethylaminoethyl chain is ordinary sigma connectivity
- the fused indole core is one explicit `10`-electron bonding system, `indole_pi_system`
- the `P=O` bond is one explicit `2`-electron bonding system, `phosphoryl`
- the `O-P(OH)2` fragment otherwise stays as ordinary sigma connectivity
- there are no atom-centered SMILES stereochemistry flags on this boundary structure

The code path is:

- parse [`molecules/psilocybin.sdf`](molecules/psilocybin.sdf) with the repo SDF reader
- keep the parsed atoms, coordinates, and sigma edges
- replace the localized boundary bond-order story with the inferred Dietz systems below

That means the object is not stored as alternating double bonds around the indole ring. It is stored as:

- one sigma framework over the whole molecule
- one fused aromatic indole pool
- one phosphoryl pool

### Psilocybin in Code

The built-in example is written out explicitly as atoms, sigma edges, and Dietz bonding systems:

```python
from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import AtomicSymbol, Molecule
from moladt.examples._literal import atom, bond

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
                frozenset(
                    {
                        Edge(AtomId(6), AtomId(7)),
                        Edge(AtomId(7), AtomId(8)),
                        Edge(AtomId(8), AtomId(9)),
                        Edge(AtomId(9), AtomId(10)),
                        Edge(AtomId(10), AtomId(6)),
                        Edge(AtomId(10), AtomId(11)),
                        Edge(AtomId(11), AtomId(12)),
                        Edge(AtomId(12), AtomId(13)),
                        Edge(AtomId(13), AtomId(14)),
                        Edge(AtomId(14), AtomId(9)),
                    }
                ),
                "indole_pi_system",
            ),
        ),
        (
            SystemId(2),
            mk_bonding_system(
                NonNegative(2),
                frozenset({Edge(AtomId(16), AtomId(17))}),
                "phosphoryl",
            ),
        ),
    ),
)
```

That is the full MolADT example form: the example stores the atom table, coordinates, and sigma framework directly, while the Dietz layer adds the fused indole `pi` pool and the explicit `P=O` pool.

### Why This Fits MolADT

Psilocybin is not non-classical in the diborane or ferrocene sense, but it still benefits from the ADT:

- the indole aromaticity is one named bonding system instead of a localized bond pattern
- the phosphate group can keep the `P=O` contribution explicit without forcing the whole fragment into localized double-bond syntax
- the side chain, aromatic core, and phosphate are separated cleanly into sigma edges plus bonding systems

### File Map

- file-backed source: [`molecules/psilocybin.sdf`](molecules/psilocybin.sdf)
- source object: [`moladt/examples/psilocybin.py`](moladt/examples/psilocybin.py)
- manuscript wrapper: [`moladt/examples/manuscript.py`](moladt/examples/manuscript.py)
- examples overview: [`docs/examples.md`](docs/examples.md)
