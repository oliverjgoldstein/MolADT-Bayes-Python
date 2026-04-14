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

The built-in example keeps the parsed SDF atoms and sigma edges, then adds the two explicit bonding systems:

```python
from moladt.chem.dietz import AtomId, Edge, NonNegative, SystemId, mk_bonding_system
from moladt.chem.molecule import Molecule
from moladt.io.sdf import read_sdf_record

psilocybin_pretty = Molecule(
    atoms=read_sdf_record("molecules/psilocybin.sdf").molecule.atoms,
    local_bonds=read_sdf_record("molecules/psilocybin.sdf").molecule.local_bonds,
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

That is the full MolADT example form: the SDF file supplies the atom table, coordinates, and sigma framework, while the Dietz layer only adds the fused indole `pi` pool and the explicit `P=O` pool.

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
