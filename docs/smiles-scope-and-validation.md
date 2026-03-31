# SMILES Scope and Validation

The SMILES layer in this repo is intentionally conservative. It is meant to cover a stable classical subset cleanly and to reject structures that do not fit that subset instead of silently inventing an encoding.

## Supported

- atoms and bracket atoms
- bracket hydrogens and formal charges
- branches
- ring digits `1-9`
- single, double, and triple bonds
- benzene-style aromatic input such as `c1ccccc1`

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
- benzene from `molecules/benzene.sdf` renders successfully
- ferrocene and diborane remain MolADT examples but are not `to-smiles` targets

## Benchmarking Boundary

The predictive benchmarks use canonicalized classical SMILES or MolADT-derived features. The structural branch is not reported as a raw SDF descriptor path: structure-backed molecules are parsed into the MolADT object first and then featurized from that ADT.

The ZINC timing benchmark still measures RDKit parsing/canonicalization, and the MolADT-enabled path now also builds a matched local corpus with one MolADT JSON file per molecule plus one canonical SMILES file with the same molecule count. It then measures the local MolADT SMILES parser against local MolADT file parsing on that matched corpus.

## Related Files

- [`moladt/io/smiles.py`](../moladt/io/smiles.py)
- [`moladt/chem/validate.py`](../moladt/chem/validate.py)
- [`tests/test_parser_roundtrip.py`](../tests/test_parser_roundtrip.py)
- [`tests/test_validation_properties.py`](../tests/test_validation_properties.py)
