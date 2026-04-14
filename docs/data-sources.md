# Data Sources

This repo vendors the raw benchmark source files used by the main workflows so the main commands do not need to fetch them first. The repo still ignores generated archives, extracted scratch directories, processed tables, and benchmark outputs.

## FreeSolv

The repo vendors the small subset actually needed by `make freesolv`:

- `data/raw/freesolv/SAMPL.csv`
- `data/raw/freesolv/sdffiles/*.sdf`
- `data/raw/freesolv/FreeSolv-master/FreeSolv-master/database.json`

Upstream sources:

- FreeSolv paper: https://doi.org/10.1007/s10822-014-9747-x
- PubMed entry: https://pubmed.ncbi.nlm.nih.gov/24928188/
- Upstream data repo: https://github.com/MobleyLab/FreeSolv
- CSV URL used by the downloader: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv

The benchmark keys FreeSolv rows from the `642` vendored SDF files and joins targets and names from the bundled FreeSolv metadata, rather than trying to recover identity by re-rendering SDF files back into CSV SMILES text. If several SDF variants exist for one compound id, the loader prefers a `3D` conformer file first, then a `V3000` file, then the remaining SDF paths.

The repo does not need the rest of the upstream simulation archives for the benchmark itself, even though the vendored FreeSolv snapshot still includes several of them.

## QM9

The repo vendors the normalized QM9 source files used by the benchmark pipeline:

- `data/raw/qm9/qm9.sdf`
- `data/raw/qm9/qm9.sdf.csv`

Upstream references:

- QM9 paper: https://doi.org/10.1038/sdata.2014.22
- Scientific Data entry: https://www.nature.com/articles/sdata201422
- Dataset collection DOI: https://doi.org/10.6084/m9.figshare.c.978904.v5

If those files are missing, the downloader falls back to the upstream QM9 archives and target table:

- `https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/qm9.tar.gz`
- `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.tar.gz`
- `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz`
- `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv`

When the extracted archive offers several QM9 SDF variants, the downloader prefers a `3D` conformer SDF first, then a `V3000` file, then the remaining SDF paths.

## ZINC

The timing benchmark now expects a normalized ZINC SDF source file in the raw cache, typically:

- `data/raw/zinc/zinc15_250K_2D.sdf`

Upstream references:

- ZINC15 paper: https://doi.org/10.1021/acs.jcim.5b00559
- PubMed entry: https://pubmed.ncbi.nlm.nih.gov/26479676/
- ZINC15 home page cited by the paper: http://zinc15.docking.org

If that file is missing, the downloader falls back to the DeepChem-hosted ZINC15 archive, using the `2D` archive by default and gracefully falling back to the best available SDF variant if a requested higher-dimensional archive is missing. For example:

- `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/zinc15_250K_2D.tar.gz`
