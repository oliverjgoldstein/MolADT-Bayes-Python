# Example Molecules

This repo keeps its example molecules as checked-in SDF files under `molecules/`. The built-in MolADT objects under `moladt/examples/` are assembled by parsing those same files first and then adding explicit Dietz bonding systems where the example needs them.

## At a Glance

| Example | Where it lives | Easiest inspection | What it demonstrates | Backing |
| --- | --- | --- | --- | --- |
| Benzene | [`molecules/benzene.sdf`](../molecules/benzene.sdf), [`moladt/examples/benzene.py`](../moladt/examples/benzene.py), [`moladt/examples/benzene_pretty.py`](../moladt/examples/benzene_pretty.py) | `./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf` | A classical six-membered ring with one `pi_ring` Dietz system | Both file-backed and built-in |
| Water | [`molecules/water.sdf`](../molecules/water.sdf), [`moladt/examples/sample_molecules.py`](../moladt/examples/sample_molecules.py) | `./.venv/bin/python -m moladt.cli parse molecules/water.sdf` | A minimal classical molecule used for round-trip and SMILES tests | Both file-backed and built-in |
| Morphine | [`molecules/morphine.sdf`](../molecules/morphine.sdf), [`moladt/examples/morphine.py`](../moladt/examples/morphine.py), [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py) | `./.venv/bin/python -m moladt.cli parse molecules/morphine.sdf` or `./.venv/bin/python -m moladt.cli pretty-example morphine` | Explicit Dietz version of the classic morphine sketch, with the standard SMILES stereochemistry flags preserved on the object | Both file-backed and built-in |
| Ferrocene | [`molecules/ferrocene.sdf`](../molecules/ferrocene.sdf), [`moladt/examples/ferrocene.py`](../moladt/examples/ferrocene.py), [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py) | `./.venv/bin/python -m moladt.cli parse molecules/ferrocene.sdf` or `./.venv/bin/python -m moladt.cli pretty-example ferrocene` | Two cyclopentadienyl `pi` systems plus an Fe back-donation-style pool | Both file-backed and built-in |
| Diborane | [`molecules/diborane.sdf`](../molecules/diborane.sdf), [`moladt/examples/diborane.py`](../moladt/examples/diborane.py), [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py) | `./.venv/bin/python -m moladt.cli parse molecules/diborane.sdf` or `./.venv/bin/python -m moladt.cli pretty-example diborane` | Two explicit `3c-2e` bridging hydrogen systems | Both file-backed and built-in |

## Benzene

- File-backed source: [`molecules/benzene.sdf`](../molecules/benzene.sdf)
- Built-in source: [`moladt/examples/benzene.py`](../moladt/examples/benzene.py)
- Pretty alias: [`moladt/examples/benzene_pretty.py`](../moladt/examples/benzene_pretty.py)

Use:

```bash
./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf
./.venv/bin/python -m moladt.cli to-smiles molecules/benzene.sdf
```

This is the main classical example for parser, validator, SDF, and conservative SMILES round-trip coverage.

## Water

- File-backed source: [`molecules/water.sdf`](../molecules/water.sdf)
- Built-in source: [`moladt/examples/sample_molecules.py`](../moladt/examples/sample_molecules.py)

Use:

```bash
./.venv/bin/python -m moladt.cli parse molecules/water.sdf
```

Water is a small sanity-check molecule used in parser and renderer tests. It stays inside the conservative SMILES subset and renders to `[OH2]`.

## Morphine

- File-backed source: [`molecules/morphine.sdf`](../molecules/morphine.sdf)
- Built-in source: [`moladt/examples/morphine.py`](../moladt/examples/morphine.py)
- Manuscript-facing wrapper: [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py)

Use:

```bash
./.venv/bin/python -m moladt.cli parse molecules/morphine.sdf
./.venv/bin/python -m moladt.cli pretty-example morphine
```

Morphine is the cleanest built-in example of why the Dietz ADT is not just a parsed string. The built-in object starts from the parsed SDF atoms and sigma edges, then keeps the alkene plus phenyl ring as explicit bonding systems and preserves the five atom-centered stereochemistry flags from the standard boundary SMILES in `smiles_stereochemistry`.

## Ferrocene

- File-backed source: [`molecules/ferrocene.sdf`](../molecules/ferrocene.sdf)
- Built-in source: [`moladt/examples/ferrocene.py`](../moladt/examples/ferrocene.py)
- Manuscript-facing wrapper: [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py)

Use:

```bash
./.venv/bin/python -m moladt.cli parse molecules/ferrocene.sdf
./.venv/bin/python -m moladt.cli pretty-example ferrocene
```

Ferrocene demonstrates why the MolADT core is broader than the current SMILES renderer. The molecule is representable and pretty-printable in MolADT, but it is not part of the supported classical `to-smiles` subset.

## Diborane

- File-backed source: [`molecules/diborane.sdf`](../molecules/diborane.sdf)
- Built-in source: [`moladt/examples/diborane.py`](../moladt/examples/diborane.py)
- Manuscript-facing wrapper: [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py)

Use:

```bash
./.venv/bin/python -m moladt.cli parse molecules/diborane.sdf
./.venv/bin/python -m moladt.cli pretty-example diborane
```

Diborane shows explicit multicenter bonding with two bridging hydrogen systems. Like ferrocene, it remains a MolADT example, not a supported `to-smiles` target.

## Side-by-Side SMILES vs MolADT

| Example | SMILES Side | MolADT Side |
| --- | --- | --- |
| Diborane | `[BH2]1[H][BH2][H]1` is standard, but it does not say "two explicit 3c-2e bridges". | `pretty-example diborane` shows two explicit `3c-2e` Dietz bridge systems over the bridging hydrogens. |
| Ferrocene | `[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]` is standard, but it splits the sandwich into ionic fragments. | `pretty-example ferrocene` shows the Fe-centered object with two Cp `pi` systems and one `fe_backdonation` pool. |
| Morphine | `CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5` | `pretty-example morphine` shows the same fused graph as direct `local_bonds`, explicit `alkene_bridge` and `phenyl_pi_ring` systems, and preserved atom-centered stereochemistry flags. |

## Related Example Modules

- [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py)
- [`moladt/examples/benzene.py`](../moladt/examples/benzene.py)
- [`moladt/examples/benzene_pretty.py`](../moladt/examples/benzene_pretty.py)
- [`moladt/examples/diborane.py`](../moladt/examples/diborane.py)
- [`moladt/examples/ferrocene.py`](../moladt/examples/ferrocene.py)
- [`moladt/examples/morphine.py`](../moladt/examples/morphine.py)
- [`moladt/examples/sample_molecules.py`](../moladt/examples/sample_molecules.py)
