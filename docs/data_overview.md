# Data Directory Overview

This document explains the layout of the `data/` directory and the role of the
parser scripts stored with each dataset.

```
data/
  sne/        - Supernovae Type Ia datasets
  bao/        - Baryon Acoustic Oscillation measurements
  cmb/        - Cosmic Microwave Background spectra
  gw/         - Gravitational-wave standard siren observations (placeholder)
```

Note: The `gw` parsers are stubs that log a message and return `None` while
placeholder management consolidates upcoming gravitational-wave standard siren
support.
Each subdirectory contains one or more dataset sources. A Python file named
`cosmo_parser_*.py` lives inside each source folder and registers a parser
function via decorators from `copernican_lib.dataset_registry`.
Folders named `placeholder` are ignored during automatic discovery so work-in-
progress datasets do not appear in interactive menus. When a dataset becomes
usable simply rename the folder and supply a valid parser and metadata file.

Every dataset folder also provides a `metadata_*.yml` describing the
source. Fields such as `dataset_name`, `dataset_id`, `description`,
`citation`, `license`, the full `author` list and accompanying BibTeX
information (for example `title`, `volume`, `journal` and `DOI`) are read
by `copernican_lib/dataset_registry.py` after the parser returns so individual
parsers remain metadata-agnostic. Parsed DataFrames expose the same
information on their `.attrs` property, and `dataset_id` is used when
constructing output filenames. Each metadata file also documents the exact
data tables consumed by the parser through a `data_files` sequence whose
entries are relative to the dataset directory; keeping that list accurate
lets the loader hash only the files that matter for reproducibility.
The loaders now attach `dataset_version` and `data_path` so manifests retain
the release tag and the exact source directory. They also populate
`independence_assumptions` with the statements quoted in
`copernican_lib/config_schemas/run_config.yml`.
Finally, the loaders compute a SHA256 digest for the metadata files, the
registered parser, and the dataset files listed in `data_files`. When a
metadata file omits `data_files`, the loader falls back to hashing files with
common data extensions (e.g., `.dat`, `.cov`, `.txt`, `.fits`, `.yml`)
while still skipping documentation such as `README`s and `LICENSE`s. These
digests are stored on `df.attrs['file_hashes']` and logged so manifests can
reproduce exact inputs. BAO DataFrames additionally carry a `model_prediction`
column which is populated during analysis and now remains consistent even when
the suite compares a model against itself because the Stage 2 SNe chain is
reused for both roles. See `dataset_metadata.md` for a full description of the
metadata fields. The reference tables remain read-only, while parser `.py`
files and accompanying `metadata_*.yml` files may be updated.

When the MCMC engine runs it writes NetCDF chains that capture burn-in and
production lengths, per-walker acceptance fractions, the complete
log-probability trace and posterior summaries. Walkers that encounter
``nan`` coordinates during burn-in are reseeded automatically so the stored
chains never contain undefined numbers and archived logs stay free of the
emcee warning observed in the latest LCDM self-test.

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

### Union3 UNITY compilation (Rubin et al. 2025)
*Source:* “Union Through UNITY: Cosmology with 2,000 SNe Using a Unified
Bayesian Framework” (Rubin et al. 2025).
*Location:* `data/sne/union3/`.
*Parser:* `cosmo_parser_union3.py` loads `mu_mat_union3_cosmo=2_mu.fits`, exposing
the 22 redshift nodes (first row), the compressed distance moduli (first
column) and the inverse covariance block the likelihood uses directly.
*Status:* The release includes the UNITY bookkeeping (Stan models, helper
utilities, `read_and_sample.py`, all inputs and the UNITY tarball), but the CLI
and GUI only consume the compressed µ/cov matrix today. When the FITS file is
updated rerunning the UNITY steps remains an option; the README inside the
folder explains how to reproduce it.
*License:* MIT via [`licenses/Union3-MIT.txt`](../licenses/Union3-MIT.txt); cite
Rubin et al. (2025) when publishing.

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
*Parser:* `cosmo_parser_compound.py` scans the directory for a YAML file and
loads its `data_points` table into a `DataFrame`. Numeric columns are coerced
to floats and rows missing required fields are discarded. No covariance matrix
is supplied, so the engine assumes uncorrelated errors and applies a diagonal
covariance during \(\chi^2\) evaluation.

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

All reference tables included with the suite are considered read-only. Parser
scripts and metadata may be edited when necessary. If table edits are needed,
copy the dataset to a new directory and adjust the `dataset_id` to avoid
clashing with the shipped files.

## Parser Hash Verification
Each parser module under `data/` is hashed and recorded in
`copernican_lib/dataset_registry.py`'s `TRUSTED_PARSER_DIGESTS` mapping. The
launcher refuses to import a parser unless its SHA256 digest matches the trusted
value so removing metadata such as `Last Updated` markers requires updating the
corresponding hash before users can run the GUI or CLI again. To refresh a hash:

1. Compute the new digest with newline normalisation, for example:

```
python - <<'PY'\nimport hashlib\nfrom pathlib import Path\npath = Path('data/sne/jla2014/cosmo_parser_jla2014.py')\nhashlib.sha256(path.read_bytes().replace(b\"\\r\\n\", b\"\\n\")).hexdigest()\nPY
```

2. Replace the old digest entry in `TRUSTED_PARSER_DIGESTS`.
3. Log the update in `CHANGELOG.md` and extend the `docs/data_overview.md`
   narrative so the history of the hash change follows Law 11.

When metadata-only edits happen, documenting the update here ensures future
contributors understand why hashes moved even when no parser logic changed.
