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
