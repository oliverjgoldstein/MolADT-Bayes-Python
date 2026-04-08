# Example Molecules

This repo includes two small SDF files in `molecules/` and several built-in MolADT objects under `moladt/examples/`. The easiest inspection path is the CLI for file-backed examples and `pretty-example` for the built-in Dietz objects.

## At a Glance

| Example | Where it lives | Easiest inspection | What it demonstrates | Backing |
| --- | --- | --- | --- | --- |
| Benzene | [`molecules/benzene.sdf`](../molecules/benzene.sdf), [`moladt/examples/benzene.py`](../moladt/examples/benzene.py), [`moladt/examples/benzene_pretty.py`](../moladt/examples/benzene_pretty.py) | `./.venv/bin/python -m moladt.cli parse molecules/benzene.sdf` | A classical six-membered ring with one `pi_ring` Dietz system | Both file-backed and built-in |
| Water | [`molecules/water.sdf`](../molecules/water.sdf), [`moladt/examples/sample_molecules.py`](../moladt/examples/sample_molecules.py) | `./.venv/bin/python -m moladt.cli parse molecules/water.sdf` | A minimal classical molecule used for round-trip and SMILES tests | Both file-backed and built-in |
| Morphine | [`moladt/examples/morphine.py`](../moladt/examples/morphine.py), [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py) | `./.venv/bin/python -m moladt.cli pretty-example morphine` | Explicit Dietz version of the classic morphine sketch, with the standard SMILES stereochemistry flags preserved on the object | Built-in object |
| Ferrocene | [`moladt/examples/ferrocene.py`](../moladt/examples/ferrocene.py), [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py) | `./.venv/bin/python -m moladt.cli pretty-example ferrocene` | Two cyclopentadienyl `pi` systems plus an Fe back-donation-style pool | Built-in object |
| Diborane | [`moladt/examples/diborane.py`](../moladt/examples/diborane.py), [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py) | `./.venv/bin/python -m moladt.cli pretty-example diborane` | Two explicit `3c-2e` bridging hydrogen systems | Built-in object |

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

- Built-in source: [`moladt/examples/morphine.py`](../moladt/examples/morphine.py)
- Manuscript-facing wrapper: [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py)

Use:

```bash
./.venv/bin/python -m moladt.cli pretty-example morphine
./.venv/bin/python -m moladt.cli parse-smiles "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
```

Morphine is the cleanest built-in example of why the Dietz ADT is not just a parsed string. The standard boundary SMILES carries ring digits, a localized double-bond pattern, and five atom-centered stereochemistry flags, while the built-in MolADT object stores the fused graph directly as sigma edges, keeps the alkene plus phenyl ring as explicit bonding systems, and preserves those five flags in `smiles_stereochemistry`. The conservative `parse-smiles` path keeps the localized double bonds explicit; the built-in morphine object then groups the alkene and phenyl fragment into Dietz systems on purpose.

## Ferrocene

- Built-in source: [`moladt/examples/ferrocene.py`](../moladt/examples/ferrocene.py)
- Manuscript-facing wrapper: [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py)

Use:

```bash
./.venv/bin/python -m moladt.cli pretty-example ferrocene
```

Ferrocene demonstrates why the MolADT core is broader than the current SMILES renderer. The molecule is representable and pretty-printable in MolADT, but it is not part of the supported classical `to-smiles` subset.

## Diborane

- Built-in source: [`moladt/examples/diborane.py`](../moladt/examples/diborane.py)
- Manuscript-facing wrapper: [`moladt/examples/manuscript.py`](../moladt/examples/manuscript.py)

Use:

```bash
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
