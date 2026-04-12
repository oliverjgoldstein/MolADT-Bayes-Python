# SMILES Scope and Validation

The SMILES layer in this repo is intentionally conservative. It is meant to cover a stable classical subset cleanly and to reject structures that do not fit that subset instead of silently inventing an encoding.

## Supported

- atoms and bracket atoms
- explicit bracket hydrogens and formal charges
- implicit terminal hydrogens on supported bare atoms such as `C`, `N`, `O`, halogens, and aromatic lowercase atoms
- atom-centered `@` and `@@` stereochemistry on bracket atoms
- directional `/` and `\` bond annotations
- branches
- ring digits `1-9`
- single, double, and triple bonds
- graph-based six-membered `pi_ring` recovery from aromatic lowercase input
- benzene-style aromatic input such as `c1ccccc1`
- fused classical ring cases such as the stereochemical morphine boundary string, with ring closures, localized double bonds, and atom-centered stereochemistry preserved conservatively

## Not Supported

- non-classical multicenter systems such as diborane and ferrocene
- arbitrary delocalized systems outside localized double/triple bonds and simple six-edge `pi_ring` cases
- components that need more than 9 ring closures

Those molecules still remain representable in the MolADT core. The restriction is on the current SMILES parser and renderer, not on the core molecule ADT.

## What Gets Validated

User-facing CLI flows call `validate_molecule` before rendering or benchmarking:

- `parse` reads an SDF record, then validates the resulting MolADT structure.
- `parse-smiles` parses the conservative SMILES subset, then validates the resulting MolADT structure.
- `to-smiles` validates the molecule before trying to render it.

Validation checks include:

- self-bonds and missing-atom references
- symmetric bond-map construction
- maximum valence bounds by element

## Rendering Boundary

`to-smiles` and `molecule_to_smiles(...)` work on validated molecules in the supported classical subset. Current renderer rejections come from the code, for example:

- `SMILES rendering only supports localized double/triple bonds and six-edge pi rings`
- `pi_ring must be a simple six-membered cycle to render as SMILES`
- `SMILES rendering currently supports at most 9 ring closures per component`

That is why:

- `c1ccccc1` is supported input
- bare `C` and bare `O` become methane- and water-style MolADT objects with inferred terminal hydrogens
- the stereochemical morphine boundary string preserves its five atom-centered `@`/`@@` flags and keeps its localized double-bond pattern explicit
- an explicit Kekule string stays explicit and does not get silently promoted to a delocalized `pi_ring`
- parsed SMILES stereochemistry is stored on `molecule.smiles_stereochemistry`
- the current renderer does not yet regenerate `@`, `@@`, `/`, or `\` from stored stereochemistry annotations
- benzene from `molecules/benzene.sdf` renders successfully
- ferrocene and diborane remain MolADT examples but are not `to-smiles` targets

## Benchmarking Boundary

The predictive benchmarks use MolADT as the benchmark object. Boundary SMILES still matter because they are one of the ways the repo builds the typed molecule, but the published benchmark graphs compare the best local MolADT Stan run against MoleculeNet rather than running a separate SMILES benchmark row.

The ZINC timing benchmark is an interoperability/runtime benchmark rather than the central representation comparison. It keeps its own ingest-path measurements and matched local MolADT corpus, and it breaks out the manifest CSV field-to-string baseline before the local SMILES and MolADT parse stages.

## Related Files

- [`moladt/io/smiles.py`](../moladt/io/smiles.py)
- [`moladt/chem/validate.py`](../moladt/chem/validate.py)
- [`tests/test_parser_roundtrip.py`](../tests/test_parser_roundtrip.py)
- [`tests/test_validation_properties.py`](../tests/test_validation_properties.py)
