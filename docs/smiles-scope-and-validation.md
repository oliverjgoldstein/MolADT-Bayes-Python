# SMILES Scope

The SMILES layer is conservative by design. It should accept what it can represent cleanly and reject the rest.

## Supported

- atoms and bracket atoms
- formal charges
- explicit bracket hydrogens
- implicit terminal hydrogens on supported bare atoms
- branches
- ring digits `1-9`
- single, double, triple, and quadruple bonds
- aromatic six-membered ring recovery into `pi_ring`
- atom-centered `@` / `@@`
- bond directions `/` and `\`

## Not Supported

- diborane-style multicenter SMILES rendering
- ferrocene-style organometallic rendering
- arbitrary delocalized systems
- components needing more than 9 ring closures
- full stereo regeneration from stored MolADT stereo annotations

Those molecules can still be represented in MolADT. The limit is only the current SMILES boundary.

## Validation

CLI flows validate before printing, rendering, or benchmarking.

Validation catches:

- missing atom references
- self-bonds
- inconsistent bond maps
- valence violations

## Practical Rule

Use SMILES for classical molecules. Use MolADT directly when the bonding structure needs more than a string can say.

Related files:

- [`moladt/io/smiles.py`](../moladt/io/smiles.py)
- [`moladt/chem/validate.py`](../moladt/chem/validate.py)
