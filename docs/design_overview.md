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

All observational data and accompanying metadata are stored exclusively
as YAML files.  Legacy JSON support was removed in version 3.0.0 so that
all parsers operate on a single consistent format.

`copernican.py` is the command-line entry point that orchestrates model
selection, data loading, optimisation and result generation.  The new
package name emphasises that these modules are part of the suite's core
library and not mere scripts.

LaTeX translations rely on `copernican_lib/latex_utils.py` which reads symbol
and function mappings from `latex_mappings.yml`. New commands can be added
there without touching the code.
The helper also exposes `latex_to_unicode` for rendering parameter names with
Greek letters and subscripts in console logs.
Console messages are emitted through `copernican_lib/console_output.py` so
that
all output passes through a single function. The logger patches `print` and
`input` to capture these messages verbatim.

Engines follow a strict interface. `engine_interface.validate_plugin` ensures
that any model plugin supplies the callable hooks required by a backend. This
allows alternative engines—GPU-accelerated solvers, for example—to be swapped
in
without touching the high-level orchestration in `copernican.py`.

To keep multiprocessing predictable, the suite sets the start method to
``spawn`` and validates model YAML only in the main process. Worker processes
operate on sanitised cached models which avoids repeated schema checks and
keeps startup costs low.

Caching is deliberately explicit. Parsed models are written to
`models/cache/` and cleared only when the user exits the program. This
approach
allows repeated runs with different datasets without re-parsing YAML files,
while still letting contributors inspect the generated intermediate files.
