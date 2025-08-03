# BAO Compound Dataset Format

This document describes the YAML format used for the **compound** BAO dataset
shipped with the Copernican Suite. The folder lives under
`data/bao/compound/` and mirrors the structure expected for real BAO sources.
JSON files were supported in early versions but have now been removed so
that all datasets use a single YAML representation.

Each dataset is stored in its own directory and contains a single YAML file
with a `data_points` array. The parser also looks for an optional
`metadata_*.yml` file which provides a dataset name and citation.  The compound
dataset lets developers exercise the BAO pipeline without downloading large
public releases.  A covariance matrix is intentionally omitted; uncertainties
are treated as uncorrelated.
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
description: Compilation of BAO distance measurements from multiple surveys
citation: Reference string with survey citations
notes: Any extra comments
```

All observable types use the naming convention `DV_over_rs`, `DM_over_rs` or
`DH_over_rs` to indicate $D_V$, $D_M$ or $D_H$ divided by the sound horizon. The
parser converts the YAML to a Pandas `DataFrame` and attaches the metadata to the
`.attrs` attribute. This `metadata_*.yml` with `dataset_name`, `description`,
`citation` and optional `notes` is used for **all** datasets so plot footers
can display consistent source information.
