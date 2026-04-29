# Data Sources

The repo vendors the small raw files needed for normal local runs. Larger generated files stay ignored.

## FreeSolv

Vendored files:

- `data/raw/freesolv/SAMPL.csv`
- `data/raw/freesolv/sdffiles/*.sdf`
- `data/raw/freesolv/FreeSolv-master/FreeSolv-master/database.json`

Sources:

- FreeSolv paper: https://doi.org/10.1007/s10822-014-9747-x
- Upstream data: https://github.com/MobleyLab/FreeSolv

The benchmark joins targets from FreeSolv metadata to the vendored SDF structures.

## QM9

Vendored files:

- `data/raw/qm9/qm9.sdf`
- `data/raw/qm9/qm9.sdf.csv`

Sources:

- QM9 paper: https://doi.org/10.1038/sdata.2014.22
- Dataset DOI: https://doi.org/10.6084/m9.figshare.c.978904.v5

If needed, the downloader falls back to the upstream archives.

## ZINC

Timing uses a normalized ZINC SDF source, usually:

- `data/raw/zinc/zinc15_250K_2D.sdf`

Sources:

- ZINC15 paper: https://doi.org/10.1021/acs.jcim.5b00559
- ZINC15 home: http://zinc15.docking.org

If the local file is missing, the downloader uses the DeepChem-hosted archive.
