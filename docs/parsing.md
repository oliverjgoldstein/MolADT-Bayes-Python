# Parsing

MolADT can start from SDF, SMILES, or MolADT JSON.

## SDF

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
./.venv/bin/python -m moladt.cli parse --properties molecules/benzene.sdf
./.venv/bin/python -m moladt.cli perceive-sdf molecules/benzene.sdf
./.venv/bin/python scripts/audit_sdf_bonding_perception.py data/raw/freesolv/sdffiles --limit 20
```

The parser accepts SDF V2000 and a core V3000 CTAB subset: atom coordinates, bond tables, atom-local charges, and property blocks. Bond table entries become bonding systems: single, double, triple, and non-aromatic quadruple bonds are one-edge systems sharing 2, 4, 6, and 8 electrons, displayed as `single covalent`, `double covalent`, `triple covalent`, and `quadruple covalent`.

When an SDF single edge connects a charged sodium cation to a charged anion in
the supported set (`F`, `Cl`, `Br`, `I`, `O`, `N`, or `S`), the parser stores
that edge as one `ionic` bonding system with `0` shared electrons. The atom
formal charges remain on the atoms.

Delocalised systems are inferred from structure-table evidence, not from an
arbitrary SDF property convention. The parser calls the shared bonding
perception rules in `moladt.chem.bonding_perception`. Those rules currently
cover aromatic six-rings, two-edge oxo resonance such as carboxylate and nitro
groups, amide C(O)-N resonance, conjugated diene paths, borane B-H-B bridges
from geometry, and Fe-cyclopentadienyl systems. FreeSolv SDF files keep fields
such as `partial_bond_orders` and `atom_types` as properties; those fields are
parsed for inspection but are not treated as authoritative delocalisation
annotations.

## MolADT JSON

```bash
./.venv/bin/python -m moladt.cli to-json molecules/benzene.sdf > benzene.moladt.json
./.venv/bin/python -m moladt.cli from-json benzene.moladt.json
```

MolADT JSON is the shared boundary format used by the Python and Haskell repos.

## Explicit Python ADT

```bash
./.venv/bin/python -m moladt.cli to-python molecules/benzene.sdf --name benzene_generated > benzene_generated.py
./.venv/bin/python -m moladt.cli to-example molecules/benzene.sdf --name benzene_generated --output moladt/examples/benzene_generated.py
```

`to-python` and its `to-example` alias convert parsed geometry into canonical expanded `Molecule(...)` source with literal `AtomId(...)`, `Edge(...)`, and `mk_bonding_system(...)` entries. Atoms are sorted by `AtomId`, edges are normalized and sorted, and systems are sorted by `SystemId`. The output is meant for checked examples and reviewable fixtures, so it does not hide atoms or bonding systems behind generated loops.

For display stability, parsers assign low `SystemId` values to named or
multi-edge systems before the ordinary one-edge covalent systems. A parsed
benzene fixture therefore has `SystemId(1)` for `pi_ring`, followed by the
one-edge 2e systems.

Python API:

```python
from moladt.chem.validate import validate_molecule
from moladt.io.molecule_json import molecule_from_json, molecule_to_json
from moladt.io.sdf import read_sdf
from moladt.io.smiles import molecule_to_smiles

molecule = validate_molecule(read_sdf("molecules/benzene.sdf"))
payload = molecule_to_json(molecule)
round_tripped = validate_molecule(molecule_from_json(payload))

print(molecule_to_smiles(round_tripped))
```

## SMILES

```bash
./.venv/bin/python -m moladt.cli parse-smiles 'c1ccccc1'
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
```

The SMILES path is conservative. It covers the classical subset used by tests and examples. It emits the same `single covalent`, `double covalent`, `triple covalent`, and `quadruple covalent` one-edge systems, adds `pi_ring` systems for supported aromatic six-rings, and does not try to encode every MolADT molecule.

Bracket charges are retained. `[Na+][Cl-]` parses as two charged atoms plus one
zero-electron `ionic` bonding system and renders back to `[Na+][Cl-]`.

## Viewer

```bash
make molecule-viewer VIEWER_EXAMPLES=ferrocene
./.venv/bin/python -m moladt.cli view-html benzene.moladt.json --output benzene.viewer.html
```

This writes a standalone HTML viewer under `results/viewer/`. The Make target uses built-in explicit ADT examples. The direct `view-html` command accepts MolADT JSON. Viewer commands print a portable `file://` URL for manual opening on any OS if auto-open fails. Click an atom in the viewer to show coordinates, 3D edge lengths, effective orders, bonding systems, and bond angles calculated from the molecule's coordinate data. Charge is shown as blue/red halos around charged atoms; halo size and opacity scale with formal-charge magnitude, atoms participating in an ionic bonding system get an additional boost, and ionic edges draw a blue-to-red gradient between charged atoms. Ordinary covalent edges are dark grey single/double/triple/quadruple line sets; the right panel lists those covalent systems and whether they are single, double, triple, or quadruple. If an edge belongs to multiple bonding systems, each system overlay gets a dashed lane, including covalent versus delocalised overlap in ferrocene. Pi, bridge, coordination, and other non-standard systems are labelled as delocalised bonding in the panel and use coloured dashed overlays on the canvas.

## Rule Of Thumb

- Use SDF only as an import boundary or parser fixture.
- Use SMILES for compact classical boundary strings.
- Use explicit Python ADTs for checked examples.
- Use MolADT JSON when another tool or the viewer needs the typed molecule as a file.

See [SMILES scope](smiles-scope-and-validation.md) for parser limits.
