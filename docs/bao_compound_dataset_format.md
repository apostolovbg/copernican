# BAO Compound Dataset Format
This document describes the YAML format used for the **compound** BAO
dataset shipped with the Copernican. The folder lives under
`copernican/datasets/bao/compound/` and mirrors the structure expected for
real BAO sources. JSON files were supported in early versions but have
been removed so that all datasets use a single YAML representation.
Each dataset is stored in its own directory and contains a single YAML file
with a `data_points` array. An optional `metadata_*.yml` file mirrors the
structure used for real datasets, including BibTeX fields. The data loader
reads this metadata after parsing so the parser itself remains trivial. The
compound dataset lets developers exercise the BAO pipeline without downloading
large public releases. A covariance matrix is intentionally omitted;
uncertainties are treated as uncorrelated.
The accompanying parser registers itself under the dataset ID
`compound_bao_set` so ``load_bao_data('compound_bao_set')`` locates it directly
without discovery.
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
dataset_id: compound_bao_set
description: Compilation of BAO distance measurements without a covariance
  matrix
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
notes: Observable types: DV_over_rs (D_V(z)/r_s), DM_over_rs (D_M(z)/r_s),
  DH_over_rs (D_H(z)/r_s = c/(H(z) r_s)). All r_s values are the model's
  sound horizon at the drag epoch unless a fiducial r_s is specified. No
  covariance matrix is available; uncertainties are treated as uncorrelated.
```
## Table of Contents
- [Usage](#usage)
 - [Extending the Dataset](#extending-the-dataset)
## Usage
The compound dataset is primarily intended for automated tests and examples. It
demonstrates how BAO observables are encoded without requiring gigabyte-scale
survey releases. When developing a new parser, model the output DataFrame on
the structure produced by this example: one row per measurement with columns
for the observable, its uncertainty and any fiducial sound horizon.
When a real dataset supplies a covariance matrix the parser should attach the
inverse matrix to `df.attrs['covariance_matrix_inv']`. For uncorrelated data,
as shown here, omitting the matrix is sufficient and the engine will fall back
to diagonal errors. During analysis the engine populates a
`model_prediction` column on the returned DataFrame. The Stage 2 workflow
reuses the same SNe chain whenever both models point to the identical plugin,
ensuring these predictions align perfectly between baseline and alternative
theory curves in diagnostic plots. The matching chi-squared totals recorded in
BAO CSV exports confirm that LCDM-versus-LCDM checks keep the red and blue
curves coincident. All observable types use the naming convention `DV_over_rs`,
`DM_over_rs` or `DH_over_rs` to indicate $D_V$, $D_M$ or $D_H$ divided by the
sound horizon. The parser converts the YAML to a Pandas `DataFrame` and the
data loader attaches the metadata to the `.attrs` attribute. In addition to the
original `dataset_name`, a `dataset_id` is supplied for constructing output
filenames. The same `metadata_*.yml` structure with `dataset_name`,
`dataset_id`, `description`, `notes` and `citation` is used for **all**
datasets so plot footers render the dataset name in bold, followed by its
description, notes and a separate citation line.
### Extending the Dataset
Additional points can be appended to `data_points` to experiment with new BAO
measurements. Keep observable names consistent and supply a metadata file
describing the provenance of the added entries. The lightweight format allows
tests to cover edge cases—such as missing columns or unexpected types—without
shipping large survey catalogues in the repository.
