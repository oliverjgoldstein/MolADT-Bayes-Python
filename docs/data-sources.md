# Data Sources

This repo vendors the raw benchmark source files used by the front-door workflows so the main commands do not need to fetch them first. The repo still ignores generated archives, extracted scratch directories, processed tables, and benchmark outputs.

## FreeSolv

The repo vendors the small subset actually needed by `make freesolv`:

- `data/raw/freesolv/SAMPL.csv`
- `data/raw/freesolv/sdffiles/*.sdf`

Upstream sources:

- FreeSolv paper: https://doi.org/10.1007/s10822-014-9747-x
- PubMed entry: https://pubmed.ncbi.nlm.nih.gov/24928188/
- Upstream data repo: https://github.com/MobleyLab/FreeSolv
- CSV URL used by the downloader: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv

The repo does not vendor the rest of the upstream simulation archives because the benchmark code here only needs the hydration targets and the SDF structures.

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

## ZINC

The repo vendors the normalized ZINC source file used by the timing benchmark:

- `data/raw/zinc/zinc15_250K_2D.csv`

Upstream references:

- ZINC15 paper: https://doi.org/10.1021/acs.jcim.5b00559
- PubMed entry: https://pubmed.ncbi.nlm.nih.gov/26479676/
- ZINC15 home page cited by the paper: http://zinc15.docking.org

If that file is missing, the downloader falls back to the DeepChem-hosted ZINC15 archive, for example:

- `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/zinc15_250K_2D.tar.gz`
