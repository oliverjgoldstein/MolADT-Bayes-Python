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
