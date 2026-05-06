# Porting Notes

Short notes for keeping the Python and Haskell repos aligned.

- Python keeps the Haskell ADT layering: `atoms`, `systems`, and stereo annotations.
- `Molecule.atoms` is read-only and `systems` is a tuple, so the public object behaves like an immutable record.
- Use `MutableMolecule` only as an edit scratchpad.
- Stan uses rate parameters for `gamma(alpha, beta)`, so Haskell shape/scale priors are converted before fitting.
- The SDF parser stays lightweight: atoms, bonds, charges, coordinates, and property blocks.
- Orbital data is declarative. It is not a quantum chemistry engine.
- Diborane and ferrocene are validation examples for multicenter and organometallic bonding systems.
