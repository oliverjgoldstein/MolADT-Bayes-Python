# MolADT-Bayes-Python

MolADT is a molecular representation that stays explicit where SMILES is compressed.

This repo is the main MolADT benchmark runner. It keeps the representation, parsing, timing, and the focused predictive comparisons for FreeSolv and QM9.

## Start

```bash
make python-setup

make freesolv
make qm9
make timing
```

Those are the front-door commands.

- `make freesolv` runs the focused hydration free energy benchmark.
- `make qm9` runs the focused dipole moment benchmark.
- `make timing` runs the ZINC parsing and file-ingest timing benchmark.

The raw source files those commands use live in this repo under `data/raw/`.

If a raw dataset file is too large to keep in GitHub, the download path fetches it on demand. Large transfers and archive extraction show live byte counts, extraction entry counts, throughput, and elapsed time.

## Representation

MolADT keeps atoms, sigma bonds, Dietz-style bonding systems, coordinates, shells, and orbitals explicit.

If you want the orbital layer explained on its own, see [Orbitals and theoretical chemistry](docs/orbitals.md).

In the benchmark, that appears as three MolADT-facing branches:

- `moladt`
- `moladt_typed`, shown as `MolADT+`
- `moladt_typed_geom`, shown as `MolADT+ 3D`

`MolADT+` is the richer descriptor view. It adds typed pair channels, radial distance channels, bond-angle channels, torsion channels, and bonding-system summaries. `MolADT+ 3D` adds the geometry model path on top.

### SMILES vs MolADT

| Example | SMILES side | MolADT side |
| --- | --- | --- |
| Diborane | Wikipedia SMILES: `[BH2]1[H][BH2][H]1`. Standard but not faithful here: it flattens the two bridging hydrogens into ordinary graph connectivity instead of explicit `3c-2e` pools. | [`moladt/examples/diborane.py`](moladt/examples/diborane.py) stores two explicit Dietz bridge systems: `bridge_h3_3c2e` and `bridge_h4_3c2e`. |
| Ferrocene | Wikipedia SMILES: `[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]`. Standard but not faithful here: it represents ferrocene as separated ionic fragments, not the shared `eta^5` metal-ring pools we want. | [`moladt/examples/ferrocene.py`](moladt/examples/ferrocene.py) stores two Cp `pi` systems plus `fe_backdonation`. |
| Morphine | Wikipedia SMILES: `CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5`. This is a faithful standard boundary string for the classical graph and stereochemistry. | [`moladt/examples/morphine.py`](moladt/examples/morphine.py) stores the fused graph directly in `local_bonds` and makes the delocalization explicit with `alkene_bridge` and `phenyl_pi_ring`. |

That is the intended boundary: keep what SMILES really says when it is present, and use explicit Dietz systems where SMILES would otherwise flatten or omit the chemistry.

### Immutability and Traversal

In Python, `Molecule` is immutable. Its top-level fields are exposed as read-only values:

- `atoms`
- `local_bonds`
- `systems`
- `smiles_stereochemistry`

You can destructure those four fields directly because `Molecule.__iter__` yields them in that order:

```python
atoms, local_bonds, systems, smiles_stereochemistry = molecule
```

You can then iterate or sample from each attribute explicitly:

```python
atom_id, atom = next(iter(molecule.atoms.items()))
edge = next(iter(molecule.local_bonds))
system_entry = next(iter(molecule.systems), None)
atom_stereo = next(iter(molecule.smiles_stereochemistry.atom_stereo), None)
bond_stereo = next(iter(molecule.smiles_stereochemistry.bond_stereo), None)
```

The object itself does not mutate. If you want a changed molecule, construct a new one from the old one rather than editing in place.

### Mutable Sampling Workspace

For probabilistic sampling, proposal kernels, or local graph surgery, use `MutableMolecule` as a scratch state and freeze it back to `Molecule` when the proposal is ready.

`MutableMolecule` keeps the same four top-level attributes, but stores them in mutable collections:

- `atoms` as a `dict`
- `local_bonds` as a `set`
- `systems` as a `list`
- `smiles_stereochemistry` as a replaceable `SmilesStereochemistry` value

Create it from an immutable molecule with `molecule.to_mutable()` or `MutableMolecule.from_molecule(molecule)`:

```python
from dataclasses import replace

from moladt import AtomId, MutableMolecule, SmilesStereochemistry

state = MutableMolecule.from_molecule(molecule)

state.local_bonds.clear()
state.atoms[AtomId(1)] = replace(state.atoms[AtomId(1)], formal_charge=1)
state.smiles_stereochemistry = SmilesStereochemistry()

proposal = state.freeze()
```

The purpose is not to replace the immutable ADT. It gives samplers a writable workspace for birth-death moves, coordinate perturbations, charge proposals, and system edits, then returns to the canonical immutable `Molecule` for validation, serialization, and model input.

The leaf records such as `Atom`, `BondingSystem`, and the stereo entries are still immutable value objects. In a sampler, you mutate the collections or replace those records wholesale rather than editing them in place.

## Why This Exists

SMILES is useful as a boundary format. It is not the right center of this project.

- It hides geometry.
- It flattens richer bonding structure into a renderer-oriented string.
- It is awkward for non-classical chemistry.
- It gives the model a serialization instead of the chemistry object.

MolADT treats the chemistry object as the source of truth and lets SMILES sit at the edge.

## Inference

The front-page commands use the strongest user-facing paths in this repo instead of the old Stan-heavy default.

- FreeSolv uses CatBoost for the fair tabular comparison and DimeNet++ for the geometry-aware branch.
- QM9 `mu` uses CatBoost for the fair tabular comparison and ViSNet for the geometry-aware branch.
- ZINC timing is model-free and measures parse and ingest cost directly.

The older Stan baselines and broader benchmark sweeps still exist, but they are now reference material rather than the landing-page story.

If you want the Stan baselines as well, install CmdStan with:

```bash
make python-cmdstan-install
```

## Outputs

Each run writes a small bundle under `results/`.

- FreeSolv: `results/freesolv/run_<timestamp>/`
- QM9: `results/qm9/run_<timestamp>/`
- Timing: `results/timing/run_<timestamp>/`

The key files are:

- `results.csv`
- `rmse_train_test_vs_literature.svg`
- `inference_sweep_overview.svg`
- `timing_overview.svg` for timing runs
- `figures/metric_comparisons/*.svg`
- `details/`

The comparison charts now separate the fair tabular story from the mixed-family frontier story. The frontier view is where `MolADT+ 3D` appears.

## Read More

- [MolADT representation](docs/representation.md)
- [Orbitals and theoretical chemistry](docs/orbitals.md)
- [Parsing and rendering](docs/parsing.md)
- [Models and benchmarks](docs/models.md)
- [Data sources](docs/data-sources.md)
- [Outputs](docs/outputs.md)
- [Quickstart](docs/quickstart.md)
- [Deep benchmark reference](docs/inference-and-benchmarks.md)
- [CLI reference](docs/cli.md)
- [Haskell interop](docs/haskell_interop.md)

## Related Repo

- [MolADT-Bayes-Haskell](https://github.com/oliverjgoldstein/MolADT-Bayes-Haskell)
