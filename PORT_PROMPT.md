You are porting the current working tree, which is the Haskell repository MolADT-Bayes-Haskell, into a Python 3.11 package named `moladt`.

Mission
Create a faithful semantic rewrite of the repository in Python. Preserve the MolADT design: a Dietz-style constitution layer, explicit coordinates, optional orbital/electronic annotations, reactions, examples, parser, validator, and a Bayesian property-inference demo. Do not produce a superficial OOP rewrite or a SMILES-first wrapper.

Non-negotiable design decisions
- The molecule must remain a typed algebraic data type. Do not collapse the core representation into strings, dicts, pandas rows, NetworkX graphs, or RDKit mol objects.
- Use the Python stdlib ADT pattern:
  - product types => `@dataclass(frozen=True, slots=True)`
  - runtime-checked scalar wrappers/newtypes => tiny frozen dataclasses (`AtomId`, `SystemId`, `NonNegative`, `Angstrom`)
  - finite tags => `Enum`
  - sum types => one dataclass per constructor + union alias (e.g. `Condition = TempCondition | PressureCondition`)
  - deconstruction => `match/case`
  - exhaustiveness => `assert_never` and mypy checks
- Use `frozenset` for immutable edge/system sets.
- Preserve smart constructors like `mk_edge` and `mk_bonding_system`.
- Keep the layered model:
  1. Dietz constitution
  2. coordinates/stereo layer
  3. orbital/electronic annotations
  4. reaction layer
- Preserve examples: benzene, diborane, ferrocene, water, methane/sample molecules.
- Preserve the lightweight SDF parser and validator semantics instead of outsourcing the chemistry core to RDKit.
- Do not transliterate LazyPPL into Python. Replace it with a Stan model + Python wrapper.

Study these Haskell files first and mirror their semantics
- `package.yaml`
- `app/Main.hs`
- `src/Chem/Dietz.hs`
- `src/Chem/Molecule.hs`
- `src/Chem/Molecule/Coordinate.hs`
- `src/Chem/Validate.hs`
- `src/Chem/IO/SDF.hs`
- `src/Orbital.hs`
- `src/Reaction.hs`
- `src/LogPModel.hs`
- `src/SampleMolecules.hs`
- `src/ExampleMolecules/Benzene.hs`
- `src/ExampleMolecules/BenzenePretty.hs`
- `src/ExampleMolecules/Diborane.hs`
- `src/ExampleMolecules/Ferrocene.hs`
- tests under `test/`

Target Python layout
- `pyproject.toml`
- `README.md`
- `moladt/chem/dietz.py`
- `moladt/chem/coordinate.py`
- `moladt/chem/molecule.py`
- `moladt/chem/validate.py`
- `moladt/chem/constants.py`
- `moladt/chem/orbital.py`
- `moladt/io/sdf.py`
- `moladt/reaction.py`
- `moladt/examples/...`
- `moladt/inference/descriptors.py`
- `moladt/inference/logp.py`
- `moladt/stan/logp_regression.stan`
- `moladt/cli.py`
- `tests/...`

Port these semantics exactly where practical
- `Edge` is a canonical undirected pair; reject self-bonds.
- `BondingSystem` stores shared electrons, cached member atoms, member edges, optional tag.
- `Molecule` stores atom map, sigma adjacency, and Dietz bonding systems.
- `effective_order`, `neighbors_sigma`, `edge_systems`, distance helpers, etc.
- `validate_molecule` checks atom existence, self-bond rejection, symmetry/bookkeeping, and conservative element-wise valence usage.
- `parse_sdf` reads V2000 atoms/bonds, applies `M  CHG` formal charges, and does the same intentionally lightweight aromatic six-ring detection used by the Haskell parser.

Inference scope
- Do not attempt to sample latent discrete molecular structure in Stan.
- Instead, port the existing `LogPModel` into a Stan-backed Bayesian regression over descriptors computed from MolADT molecules.
- Recreate the descriptor pipeline from Haskell: weight, polar proxy, surface area, bond order, heavy atoms, halogens, aromatic rings, aromatic atom fraction, rotatable bonds.
- Recreate the hierarchical prior structure from Haskell as closely as possible:
  - linear scale ~ Gamma(2, 0.2)
  - quadratic scale ~ Gamma(2, 0.05)
  - descriptor scale ~ Gamma(2, 0.1)
  - intercept ~ Normal(0, 0.5)
  - coefficient families ~ Normal(0, corresponding scale)
  - observation model uses logP likelihood with sigma 0.2 unless there is a compelling reason to infer sigma and you document the change.
- Use PyStan 3 correctly:
  - install package `pystan`
  - import module `stan`
  - compile with `stan.build(...)`
  - sample with `posterior.sample(...)`
- Produce posterior summaries and posterior predictive estimates for:
  - water
  - molecules in DB2
- Expose a CLI like:
  - `python -m moladt.cli parse molecules/benzene.sdf`
  - `python -m moladt.cli infer-logp --train logp/DB1.sdf --test molecules/water.sdf --num-chains 4 --num-samples 1000 --seed 1`

Type safety / code style
- Use `from __future__ import annotations`
- Keep the core model mostly functional and immutable
- Do not use pydantic/attrs/networkx for the core ADTs
- Use `TypeAlias` or simple alias assignment for unions; avoid 3.12-only syntax unless you intentionally raise the minimum Python version
- Add mypy configuration and make the core package type-check cleanly
- Add serialization helpers (`to_dict`/`from_dict` or JSON) for Molecule and related ADTs

Tests and acceptance criteria
- Port the edge canonicalization and validation property tests
- Add parser round-trip/smoke tests for benzene and water SDFs
- Add tests for benzene `pi_ring` handling
- Add tests for diborane and ferrocene examples constructing valid molecules
- Add a smoke test that builds the Stan model and runs a tiny posterior sample
- Before finishing, run:
  - `pytest -q`
  - `python -m moladt.cli parse molecules/benzene.sdf`
  - `python -m moladt.cli infer-logp --train logp/DB1.sdf --test molecules/water.sdf --num-chains 4 --num-samples 200 --seed 1`
- Update README with exact install/run instructions for Linux/macOS and note that Windows users should use WSL2 for PyStan.

Important
- This is a faithful Python port of the Haskell semantics, not a reimagining.
- Preserve the ADT architecture even when Python would make it easy to cheat with loose dicts.
- Keep strings as I/O boundaries, not the core representation.
- If a Haskell construct does not port cleanly, document the decision in `PORTING_NOTES.md` and choose the closest typed-Python analogue.
- Actually implement the port. Do not stop at a plan.
