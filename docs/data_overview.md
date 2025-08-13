# Data Directory Overview

This document explains the layout of the `data/` directory and the role of the
parser scripts stored with each dataset.

```
data/
  sne/        - Supernovae Type Ia datasets
  bao/        - Baryon Acoustic Oscillation measurements
  cmb/        - Cosmic Microwave Background spectra
  gw/         - Gravitational wave observations (placeholder)
  sirens/     - Standard siren events (placeholder)
```

Note: The `gw` and `sirens` parsers are stubs that log a message and return
`None`. Real data support is under development.
Each subdirectory contains one or more dataset sources. A Python file named
`cosmo_parser_*.py` lives inside each source folder and registers a parser
function via decorators from `copernican_lib.data_loaders`.
Folders named `placeholder` are ignored during automatic discovery so work-in-
progress datasets do not appear in interactive menus. When a dataset becomes
usable simply rename the folder and supply a valid parser and metadata file.

Every dataset folder also provides a `metadata_*.yml` describing the
source. Fields such as `dataset_name`, `dataset_id`, `description`,
`citation`, the full `author` list and accompanying BibTeX information
(for example `title`, `volume`, `journal` and `DOI`) are read by
`copernican_lib/data_loaders.py` after the parser returns so individual
parsers remain metadata-agnostic. Parsed DataFrames expose the same
information on their `.attrs` property, and `dataset_id` is used when
constructing output filenames. See `dataset_metadata.md` for a full
description of these fields. The reference files remain read-only.

## Supernovae Datasets

### JLA Betoule+2014
*Source:* "Improved cosmological constraints from a joint analysis of the
SDSS-II and SNLS supernova samples" (Betoule et al. 2014).
*Location:* `data/sne/jla2014/`.
*Parser:* `cosmo_parser_jla2014.py` reads the fixed-width `tablef3.dat` and
extracts the light-curve parameters. The SALT2 nuisance values
\(M_B=-19.05\), \(\alpha=0.141\) and \(\beta=3.101\) convert those
parameters into distance moduli. The systematic covariance from
`tablef4.fit` is projected into the \(\mu\) basis, summed with the
diagonal statistical errors and checked for conditioning. The resulting
matrix is inverted and stored on the returned `DataFrame` alongside the
diagonal errors used for plots.

### Pantheon+ 2022 (Scolnic et al.)
*Source:* Pantheon+SH0ES data release (Scolnic et al. 2022).
*Location:* `data/sne/pantheon/`.
*Parser:* `cosmo_parser_pantheon.py` discovers the single `.dat` and `.cov`
files, reads the distance moduli and verifies the essential columns. The
supernovae are sorted by redshift and the covariance matrix is reshaped and
reordered to match. Its inverse and diagonal errors are attached to the
`DataFrame`. If inversion fails the engine falls back to the diagonal
uncertainties.

## BAO Datasets

### BOSS DR12 BAO Consensus (Alam et al. 2017)
*Source:* "The clustering of galaxies in the completed SDSS-III Baryon
Oscillation Spectroscopic Survey" (Alam et al. 2017).
*Location:* `data/bao/bossdr12/`.
*Parser:* `cosmo_parser_bossdr12.py` reads the `dM/Hz` and `D_V/F_AP` tables
and their individual covariance matrices. The `dM` and `Hz` measurements are
converted to `D_M/rs` and `D_H/rs` with the fiducial sound horizon, while
`D_V/rs` comes directly from the second table. The covariance matrices are
assembled into a block-diagonal structure, propagated through the
transformation and inverted. The resulting `DataFrame` lists three
observables per redshift and stores the inverse covariance and diagonal
errors on `.attrs`. During \(\chi^2\) evaluation the engine contracts the
full residual vector with this inverse covariance and falls back to the
diagonal uncertainties only when the matrix is absent or ill conditioned.

### Compound BAO Dataset
*Source:* synthetic compilation for testing purposes.
*Location:* `data/bao/compound/`.
*Parser:* `cosmo_parser_compound.py` scans the directory for a YAML or JSON
file and loads its `data_points` table into a `DataFrame`. Numeric columns are
coerced to floats and rows missing required fields are discarded. No
covariance matrix is supplied, so the engine assumes uncorrelated errors and
applies a diagonal covariance during \(\chi^2\) evaluation.

## CMB Datasets

### Planck 2018 Lite TT/TE/EE
*Source:* Planck 2018 legacy release.
*Location:* `data/cmb/planck2018lite/`.
*Parser:* `cosmo_parser_cmb_planck2018lite.py` splits `cl_cmb_plik_v22.dat`
into TT, TE and EE blocks by detecting drops in the `\ell` column and
converts each to \(D_\ell\) form. The `c_matrix_plik_v22.dat` covariance is a
Fortran binary; the parser determines its endianness, reads the full matrix,
transforms it to \(D_\ell\) units and inverts it when well conditioned. The
inverse matrix, diagonal errors and CAMB parameter order are stored on
`df.attrs` for later likelihood calculations.

## Adding New Datasets

To add a new dataset create a `data/<type>/<source>/` directory, place your
raw
tables inside and implement `cosmo_parser_<source>.py`. The parser should
return
a `pandas.DataFrame` with observations and attach any auxiliary arrays to
`df.attrs`. Document the dataset in `metadata_<source>.yml` with a
`dataset_name`, a plain-language `description` and the full `citation`. Once
the
folder no longer carries the `placeholder` name it will appear automatically
in
the interactive menus.
