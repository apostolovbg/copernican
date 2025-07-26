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

Each subdirectory contains one or more dataset sources. A Python file named `cosmo_parser_*.py` lives inside each source folder and registers a parser function via decorators from `copernican_lib.data_loaders`.

Every dataset folder also provides a `metadata_*.json` describing the source. The fields `dataset_name`, `description`, `citation`, optional `notes` and `authors_all` are loaded dynamically so no parser hard-codes them. Parsed DataFrames expose the same information on their `.attrs` property. The reference files remain read-only.

## Supernovae Datasets

### JLA Betoule+2014
*Source:* "Improved cosmological constraints from a joint analysis of the SDSS-II and SNLS supernova samples" (Betoule et al. 2014).
*Location:* `data/sne/jla2014/`.
*Parser:* `cosmo_parser_jla2014.py` reads `tablef3.dat` together with the full covariance matrix in `tablef4.fit` to provide distance moduli with systematic uncertainties.

### Pantheon+ 2022 (Scolnic et al.)
*Source:* Pantheon+SH0ES data release (Scolnic et al. 2022).
*Parser:* `cosmo_parser_pantheon.py` loads `Pantheon+SH0ES.dat` together with its full covariance matrix. The inverse covariance is stored on the returned DataFrame.
