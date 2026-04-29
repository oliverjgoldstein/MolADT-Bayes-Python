# Orbitals

Atoms carry local shell and orbital data.

```text
Molecule -> Atom -> Shells -> Shell -> SubShell -> Orbital
```

The types live in [`moladt/chem/orbital.py`](../moladt/chem/orbital.py).

## What Is Stored

An `Orbital` records:

- orbital type
- electron count
- optional orientation
- optional hybrid components

A `Shell` records one principal quantum number and optional `s`, `p`, `d`, and `f` subshells.

## Why Keep This

This gives MolADT a typed place for local electronic structure:

- shell occupancy
- directional orbital character
- simple hybrid descriptions

That is richer than a graph-only molecule while staying much lighter than a quantum chemistry engine.

## What It Does Not Claim

The orbital layer is not:

- a basis set
- an SCF state
- a molecular orbital calculation
- a Hamiltonian or overlap matrix

It is local structured chemistry data attached to atoms.

## Small Example

```python
from moladt.chem.orbital import CARBON

core_shell = CARBON[0]
valence_shell = CARBON[1]
```

See [ADT model](data-model.md) for where `shells` sits inside `Atom`.
