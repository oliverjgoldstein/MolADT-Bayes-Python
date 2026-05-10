# Quickstart

Use these commands from the repo root.

## Install

```bash
make python-setup
```

This creates `./.venv` and installs the package locally. It does not touch your system Python.

The default FreeSolv benchmark does not need CmdStan. Install CmdStan only if you explicitly run the legacy Stan model overrides:

```bash
make python-cmdstan-install
```

This creates `./.cmdstan`.

## Check The CLI

```bash
make python-parse
make python-pretty-example EXAMPLE=morphine
make python-pretty-example EXAMPLE=sodium_chloride
make view
make molecule-viewer VIEWER_EXAMPLES=ferrocene
```

Those commands check SDF parsing, built-in examples, validation, and the standalone molecule viewer. `make view` clears repo Python bytecode caches, overwrites the viewer HTML, and opens seven built-in ADT examples in one browser page, preserving explicit bonding systems such as diborane bridges, ferrocene Fe-centred Cp delocalised bonding, and sodium chloride ionic contact. Charge appears as blue/red halos around charged atoms; halo size and opacity scale with formal-charge magnitude, atoms participating in an ionic bonding system get an additional boost, and ionic edges draw a blue-to-red gradient between charged atoms. Ordinary covalent edges are dark grey: single bonds draw one line, double bonds draw two lines around the edge axis, triple bonds draw three, and quadruple bonds draw four. When an edge belongs to multiple bonding systems, each system overlay gets a dashed lane, including ordinary covalent versus delocalised overlap in ferrocene. Non-standard systems such as pi rings, bridge bonds, and coordination use coloured dashed overlays; the right panel lists all bonding systems, including the ordinary one-edge covalent systems and their single/double/triple/quadruple labels. In the viewer, clicking an atom shows coordinates, 3D edge lengths, effective orders, bonding systems, and bond angles calculated from the molecule's coordinate data; the canvas also shows coordinate axes with Angstrom tick labels.

`make test-molecule-viewer` runs the viewer tests and then opens the configured viewer HTML in the default browser.

Use `OPEN_VIEWER=1` when you want the generated viewer HTML to open in the default browser. Viewer commands also print a portable `file://` URL, so if auto-open fails you can open the reported URL manually:

```bash
OPEN_VIEWER=1 make python-pretty-example EXAMPLE=morphine
OPEN_VIEWER=1 make molecule-viewer VIEWER_EXAMPLES=diborane
```

## Run Benchmarks

```bash
make freesolv
make freesolv-20split
make inverse-design TARGET=-5.0
make inverse-design-view
make qm9long
make timing
```

- `make freesolv` runs the FreeSolv MolADT WL + bonding-system empirical-Bayes GP benchmark.
- `make freesolv-20split` runs the same GP on 20 deterministic random splits and writes a summary plus split assignments under `results/freesolv_20split/`.
- `make inverse-design TARGET=-5.0` samples initial molecules from the valid FreeSolv prior, generates 1,000 valid FreeSolv candidates, and writes the top 10 by the model's Bayesian credible score.
- `make inverse-design-view` opens those saved top 10 molecules in one viewer page.
- `make qm9long` runs the full local QM9 `mu` path.
- `make timing` runs the ZINC representation timing comparison.

Run `make freesolv` before inverse design when you want inverse design paired with the latest FreeSolv benchmark artifact.

## Optional Checks

```bash
make python-test
make python-typecheck
```

## Common Fixes

If Ubuntu, Debian, or WSL is missing `ensurepip`:

```bash
sudo apt update
sudo apt install -y python3-venv
make python-setup
```

If the shell cannot find the venv Python, call it directly:

```bash
./.venv/bin/python -m moladt.cli --help
```

On Windows-style venvs, use:

```bash
./.venv/Scripts/python.exe -m moladt.cli --help
```
