# Orbitals

This page covers the orbital layer in the Python MolADT and why its types fit theoretical chemistry rather than just software bookkeeping.

## What Is In The ADT

In Python, orbitals are part of the atom description:

`Molecule -> Atom -> Shells -> Shell -> SubShell -> Orbital`

The core types live in [`moladt/chem/orbital.py`](../moladt/chem/orbital.py):

- `So`, `P`, `D`, `F` enumerate the pure orbital families
- `Orbital[SubshellType]` stores one orbital with:
  - `orbital_type`
  - `electron_count`
  - optional `orientation`
  - optional `hybrid_components`
- `SubShell[SubshellType]` stores orbitals of one angular-momentum family
- `Shell` stores a principal quantum number plus optional `s`, `p`, `d`, and `f` subshells

Atoms then carry `shells`, so the orbital layer is attached directly to the chemistry object rather than being hidden in a side table.

## Worked Example

There are two useful ways to read the orbital layer: inspect a real atom definition already shipped in the library, or build one orbital value directly.

The built-in `CARBON` value already includes a typed valence-shell picture:

```python
from moladt.chem.orbital import CARBON

core_shell = CARBON[0]     # 1s^2
valence_shell = CARBON[1]  # 2s^2, 2p^2
```

That second shell contains one filled `2s` orbital plus two singly occupied directional `p` orbitals, so the ADT is storing local electronic structure explicitly rather than deriving it later from the element symbol.

If you want to state one orbital directly, the shape is just as explicit:

```python
from moladt.chem.orbital import Orbital, P, PurePOrbital, PureSOrbital, So, ang_coord

sp2_like_px = Orbital(
    orbital_type=P.PX,
    electron_count=1,
    orientation=ang_coord(1.0, 0.0, 0.0),
    hybrid_components=(
        (1.0 / (3.0**0.5), PureSOrbital(So.S)),
        ((2.0 / 3.0) ** 0.5, PurePOrbital(P.PX)),
    ),
)
```

This is still one orbital value. `orientation` says where the directional character points, and `hybrid_components` records that the orbital is being described as a local `s`/`p` mixture instead of as a purely atomic `px`.

## Why The Types Fit Theoretical Chemistry

The important point is that the typing follows the chemistry.

- A `Shell` is separated by principal quantum number `n`, which matches the usual shell decomposition.
- A `SubShell[P]` can only contain `p` orbitals, and a `SubShell[D]` can only contain `d` orbitals. You cannot accidentally mix angular-momentum families in one subshell.
- `electron_count` is stored on each orbital, so occupancy is explicit instead of being reconstructed later from a symbol lookup.
- `orientation` is available for directional orbitals, which matters once you care about geometry instead of only graph connectivity.
- `hybrid_components` lets one orbital be described as a linear combination of pure orbitals, which is the right shape for talking about `sp`, `sp2`, and `sp3` style chemistry without pretending the whole ADT is a full SCF package.

That combination is why the types are chemistry-shaped. They do not merely say "this atom is carbon"; they preserve a typed statement about shell structure, angular character, occupancy, and optional hybridization.

## Why This Is Better Than A Reduced Graph

A reduced graph tells you that two atoms are connected. It does not tell you much about the electron picture attached to those atoms.

The orbital layer keeps visible:

- valence-shell occupancy
- directional `p`, `d`, and `f` structure
- hybrid decomposition when it is useful to say it explicitly

That is closer to how a theoretical chemist talks about local atomic structure: not only connectivity, but also shell filling and orbital character.

## What This Does Not Claim

This is still an ADT, not a full ab initio engine.

It does not encode:

- a basis-set expansion
- SCF state
- molecular orbitals over the whole molecule
- overlap, Fock, or Hamiltonian matrices
- radial exponents or full Slater/Gaussian parameterization

So the orbital types are best understood as a typed local electronic description on atoms, not as a complete wavefunction model.

## Why Keep It Anyway

For MolADT, this layer is useful because it keeps more chemically meaningful structure inside the representation itself.

- It makes pretty-printed atoms more informative.
- It gives downstream code a typed place to read shell and orbital information.
- It stays compatible with functional-style value semantics: the orbital objects are explicit, serializable, and immutable.

That is the design goal: richer than a graph-only molecular object, but still simple enough to live inside a general-purpose ADT.
