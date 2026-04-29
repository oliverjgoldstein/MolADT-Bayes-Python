# Representation

MolADT is a typed molecule object.

It is not just:

- a SMILES string
- a plain graph
- a hypergraph

It is a record with atoms, local bonds, bonding systems, stereochemistry, coordinates, charges, and shell data.

## The Core Shape

```text
Molecule = atoms + local_bonds + systems + smiles_stereochemistry
```

| Part | Meaning |
| --- | --- |
| `atoms` | Atom table with ids, element data, coordinates, shells, and charges. |
| `local_bonds` | Ordinary two-atom sigma edges. |
| `systems` | Dietz bonding systems for delocalized or multicenter electrons. |
| `smiles_stereochemistry` | Stereo flags preserved from boundary SMILES input. |

## Why It Helps

SMILES is a good boundary string. It is not the best working object for Bayesian generation.

MolADT keeps the object explicit so code can:

- edit a molecule directly
- validate it
- compute features
- score it with a Bayesian model
- export the exact generated candidate

That is why the FreeSolv inverse-design task can generate molecules under typed chemistry rules instead of proposing arbitrary strings and filtering later.

## Inspect A Molecule

```python
from moladt.chem.validate import validate_molecule
from moladt.examples import benzene

molecule = validate_molecule(benzene)

symbols = [atom.attributes.symbol.value for atom in molecule.atoms.values()]
system_tags = [system.tag for _, system in molecule.systems]

print(symbols.count("C"), symbols.count("H"))
print(len(molecule.local_bonds), system_tags)
```

Benzene is still a normal graph through `local_bonds`, but its six-electron aromatic system is also explicit in `systems`.

## Examples

| Molecule | What MolADT can say explicitly |
| --- | --- |
| Benzene | six local ring edges plus a `pi_ring` system |
| Diborane | two `3c-2e` bridge systems |
| Ferrocene | Cp ring systems plus Fe/Cp bonding pools |
| Morphine | fused graph, delocalization, and stored stereochemistry flags |

## Viewer

Export a molecule to a standalone HTML viewer:

```bash
make molecule-viewer VIEWER_INPUT=molecules/benzene.sdf
```

The viewer shows local bonds and Dietz bonding-system annotations separately.

## Boundary Formats

MolADT can read and write boundary formats:

- SDF for structure files
- conservative SMILES for classical strings
- MolADT JSON for typed interchange

The boundary format is not the model. The `Molecule` object is the model.

See [ADT model](data-model.md), [Examples](examples.md), and [Parsing](parsing.md).
