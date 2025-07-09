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

The parsers convert raw text or binary files into Pandas DataFrames with metadata stored on the `.attrs` property. These files remain read-only so that reference data is never overwritten.

## Supernovae Datasets

### JLA Betoule+2014 (UniStra)
*Source:* "Improved cosmological constraints from a joint analysis of the SDSS-II and SNLS supernova samples" (Betoule et al. 2014) hosted by the Centre de Données astronomiques de Strasbourg.
*Parser:* `cosmo_parser_h1_unistra.py` reads `tablef3.dat` and reconstructs distance moduli using fixed SALT2 nuisance parameters. Covariance matrices in `tablef4.fit` are currently ignored.

### Pantheon+ 2022 (Scolnic et al.)
*Source:* Pantheon+SH0ES data release (Scolnic et al. 2022).
*Parser:* `cosmo_parser_pantheon.py` loads `Pantheon+SH0ES.dat` together with its full covariance matrix. The inverse covariance is stored on the returned DataFrame.
