# Data Directory Overview

This document explains the layout of the `data/` directory and the role of the parser scripts stored with each dataset.

```
data/
  sne/        - Supernovae Type Ia datasets
  bao/        - Baryon Acoustic Oscillation measurements
  cmb/        - Cosmic Microwave Background spectra
  gw/         - Gravitational wave observations (placeholder)
  sirens/     - Standard siren events (placeholder)
```

Note: The `gw` and `sirens` parsers are stubs that return `None`. Real data support is under development.
Each subdirectory contains one or more dataset sources. A Python file named `cosmo_parser_*.py` lives inside each source folder and registers a parser function via decorators from `copernican_lib.data_loaders`.

Every dataset folder also provides a `metadata_*.yml` describing the
source. The fields `dataset_name`, `description`, `citation`, optional
`notes` and `authors_all` are loaded dynamically so no parser hard-codes
them. Parsed DataFrames expose the same information on their `.attrs`
property. See `dataset_metadata.md` for a full description of these
fields. The reference files remain read-only.

## Supernovae Datasets

### JLA Betoule+2014
*Source:* "Improved cosmological constraints from a joint analysis of the SDSS-II and SNLS supernova samples" (Betoule et al. 2014).
*Location:* `data/sne/jla2014/`.
*Parser:* `cosmo_parser_jla2014.py` reads `tablef3.dat`, projects the SALT2 parameter covariance from `tablef4.fit` to distance-modulus space, adds the diagonal statistical errors and stores the inverse of the total covariance matrix. The parser uses the published nuisance parameters \(M_B=-19.05\), \(\alpha=0.141\) and \(\beta=3.101\) by default.

### Pantheon+ 2022 (Scolnic et al.)
*Source:* Pantheon+SH0ES data release (Scolnic et al. 2022).
*Parser:* `cosmo_parser_pantheon.py` loads `Pantheon+SH0ES.dat` together with its
full covariance matrix.  Unlike JLA, the distance moduli are already provided so
no SALT2 nuisance parameters are required. The parser sorts the supernovae by
redshift and reorders the covariance matrix accordingly before inverting it. The
inverse covariance is stored on the returned DataFrame.
