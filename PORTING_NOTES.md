# Porting Notes

- The Python core keeps the Haskell ADT layering, but `Molecule.atoms` is stored as a read-only `mappingproxy` and `systems` as a tuple to keep the public structure immutable.
- The Stan model follows the Haskell prior scales, but Stan's `gamma(alpha, beta)` uses a rate parameter. The Python port therefore converts the Haskell shape/scale priors into Stan shape/rate form.
- The SDF parser intentionally stays lightweight. It parses V2000 atoms, bonds, `M  CHG` formal charges, and the same six-member alternating-bond aromatic-ring heuristic used by the Haskell code.
- Orbital/electronic annotations are preserved as typed dataclasses and enums. The Python port keeps them declarative and does not attach quantum-chemistry behavior to them.
- The example acceptance criteria require `diborane` and `ferrocene` to validate. To keep those multicenter and organometallic examples usable under the same validator structure, the Python valence table allows boron up to four bonds and carbon up to five in the conservative heuristic.
