# BAO Compound Dataset Format

This document describes the YAML format used for the **compound** BAO dataset shipped with the Copernican Suite. The folder lives under `data/bao/compound/` and mirrors the structure expected for real BAO sources. JSON files were supported in early versions but have now been removed so that all datasets use a single YAML representation.

Each dataset is stored in its own directory and contains a single YAML file with a `data_points` array. The parser also looks for an optional `metadata_*.yml` file which mirrors the structure used for real datasets, including BibTeX fields. The compound dataset lets developers exercise the BAO pipeline without downloading large public releases. A covariance matrix is intentionally omitted; uncertainties are treated as uncorrelated.

Example `compound.yml`:
```yaml
data_points:
  - name: 6dFGS z=0.106
    redshift: 0.106
    observable_type: DV_over_rs
    value: 2.976
    error: 0.133
    rs_fiducial_Mpc: null
  ...
```

Example `metadata_compound.yml`:
```yaml
dataset_name: Compound BAO dataset
description: Compilation of BAO distance measurements without a covariance matrix
citation: N/A
author: N/A
title: N/A
article: N/A
volume: N/A
ISSN: N/A
arXiv: N/A
url: N/A
DOI: N/A
number: N/A
journal: N/A
publisher: N/A
year: N/A
month: N/A
pages: N/A
notes: Observable types: DV_over_rs (D_V(z)/r_s), DM_over_rs (D_M(z)/r_s), DH_over_rs (D_H(z)/r_s = c/(H(z) r_s)). All r_s values are the model's sound horizon at the drag epoch unless a fiducial r_s is specified. No covariance matrix is available; uncertainties are treated as uncorrelated.
```

All observable types use the naming convention `DV_over_rs`, `DM_over_rs` or `DH_over_rs` to indicate $D_V$, $D_M$ or $D_H$ divided by the sound horizon. The parser converts the YAML to a Pandas `DataFrame` and attaches the metadata to the `.attrs` attribute. The same `metadata_*.yml` structure with `dataset_name`, `description`, `citation`, the `author` list and other BibTeX fields is used for **all** datasets so plot footers can display consistent source information.
