# Copernican Suite Architecture

This short document explains the updated folder layout introduced in
version 1.13.0.  The `copernican_lib` package now collects all
reusable modules that were previously found under `scripts/`.  Engines
and data parsers import utilities from this package so they can remain
focused on numerical work.

```
/engines/          - Computational backends
/copernican_lib/   - Shared utilities (data loading, plotting, etc.)
/models/           - JSON model definitions
/data/             - Observational datasets and their parsers
/tests/            - Unit and functional tests
```

`copernican.py` is the command-line entry point that orchestrates model
selection, data loading, optimisation and result generation.  The new
package name emphasises that these modules are part of the suite's core
library and not mere scripts.
