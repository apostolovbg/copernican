# Copernican Suite Architecture

This short document explains the updated folder layout introduced in
version 1.14.2.  The `copernican_lib` package now collects all
reusable modules that were previously found under `scripts/`.  Engines
and data parsers import utilities from this package so they can remain
focused on numerical work.

```
/engines/          - Computational backends
/copernican_lib/   - Shared utilities (data loading, plotting, etc.)
/models/           - YAML model definitions
/data/             - Observational datasets and their parsers
/tests/            - Unit and functional tests
```

`copernican.py` is the command-line entry point that orchestrates model
selection, data loading, optimisation and result generation.  The new
package name emphasises that these modules are part of the suite's core
library and not mere scripts.

LaTeX translations rely on `copernican_lib/latex_utils.py` which reads symbol and function mappings from `latex_mappings.yml`. New commands can be added there without touching the code.
The helper also exposes `latex_to_unicode` for rendering parameter names with
Greek letters and subscripts in console logs.
