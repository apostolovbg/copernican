# Copernican Suite Architecture
**Last Updated:** 2025-02-14

This short document explains the updated folder layout introduced in
version 1.14.2.  The `copernican_lib` package now collects all
reusable modules that were previously found under `scripts/`.  Engines
and data parsers import utilities from this package so they can remain
focused on numerical work.  As of version 4.3.26 the shared statistical
helpers were extracted into `copernican_lib/statistics.py`, giving every engine
a single implementation of the SNe, BAO and CMB chi-squared calculations.
Version 6.1.0 introduces `copernican_lib/likelihoods/`, a dedicated package
providing reusable log-likelihood helpers and a `JointLike` aggregator while
`statistics.py` exposes thin wrappers for backward compatibility.  Version
6.1.1 tidies the package exports so imports resolve deterministically across
the suite's lint and packaging workflows.

With the retirement of the deterministic combined optimiser the suite now
ships solely with the `cosmo_engine_mcmc` backend.  Engines remain pluggable
via the `cosmo_engine_*.py` naming convention so GPU solvers or new
optimisation strategies can be introduced without altering the orchestration
logic in `copernican.py`.  The shared helpers and validation routines therefore
remain the authoritative source of truth for statistical behaviour.

To keep emcee initialisation numerically stable the sampler now removes any
parameter whose lower and upper bounds are identical—or numerically
indistinguishable—before launching the ensemble.  Those constants re-enter each
likelihood evaluation transparently, ensuring models such as Conformal
Stationary Field Cosmology can keep fixed physical values (for example the
speed of light) without tripping emcee's condition-number safeguard.  When a
model defines only a handful of truly free parameters the engine inflates the
initial walker cloud adaptively until the ensemble's condition number satisfies
``emcee``'s guardrail, so YAML plugins with wildly different scales or exotic
bound combinations no longer require manual tuning before sampling begins.

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

Plotting helpers inside ``copernican_lib/plotter.py`` translate missing
chi-squared totals into ``N/A`` markers before drawing the summary insets. The
guard ensures sampling-only engines no longer interrupt the rendering
pipeline. Supernova-only results also populate ``χ²_Total`` with the SNe
contribution so LCDM self-checks never display ``N/A`` rows when no joint
optimiser is present.

Each evaluation now writes its outputs to a dedicated
`output/copernican-run_YYYYMMDD_HHMMSS` directory.  Besides plots and CSV
tables these folders may contain NetCDF chains produced by
`copernican_lib.chain_io` when the MCMC engine is used.  Chains now record
burn-in length, production steps, per-walker acceptance fractions and the
log-probability trace so convergence diagnostics can be reviewed after a run
without replaying the sampling session. The sampler reseeds any walkers that
acquire ``nan`` coordinates during burn-in, preventing spurious emcee runtime
warnings from polluting the logs.

`copernican.py` is launched through the `start.*` scripts which present a
menu-driven interface. Runtime options are controlled via environment
variables, and the module orchestrates model selection, data loading and
result generation. The package name emphasises that these modules are part
of the suite's core library and not mere scripts. To keep the menu responsive
the dependency scanner records a JSON snapshot of every parsed module under
`.cache/dependency_scan.json` and reuses it when file sizes and modification
times are unchanged. The cache path honours the `COPERNICAN_DEP_CACHE_DIR`
environment variable so automated pipelines can redirect it to writable
storage. The `.cache/` directory is created when needed and is not tracked by
Git, ensuring the scanner's working data never appears in commits.

LaTeX translations rely on `copernican_lib/latex_utils.py` which reads symbol
and function mappings from `latex_mappings.yml`. New commands can be added
there without touching the code.
The helper also exposes `latex_to_unicode` for rendering parameter names with
Greek letters and subscripts in console logs.
Console messages are emitted through `copernican_lib/console_output.py` so
that
all output passes through a single function. The logger patches `print` and
`input` to capture these messages verbatim.

Distance integrals produced by `model_coder.generate_callables` are now
vectorised using a cumulative trapezoid scheme. Arrays of redshifts are
integrated in one pass, keeping MCMC sampling responsive even with large
datasets.

Plots and tabular outputs are generated by `copernican_lib/plotter.py` and
`copernican_lib/csv_writer.py`.  Both modules share filename helpers from
`copernican_lib.utils` to ensure results include the dataset identifier,
model name and timestamp.  This consistent naming scheme simplifies later
comparison between runs.

Engines follow a strict interface. `engine_interface.validate_plugin` ensures
that any model plugin supplies the callable hooks required by a backend. This
allows alternative engines—GPU-accelerated solvers, for example—to be swapped
in without touching the high-level orchestration in `copernican.py`. The MCMC
engine now reuses an existing SNe chain when both selected models share the
same `MODEL_FILENAME`, guaranteeing that downstream BAO and CMB comparisons
remain perfectly aligned when the suite performs self-consistency checks such
as ΛCDM versus ΛCDM runs.

To keep multiprocessing predictable, the suite sets the start method to
``spawn`` and validates model YAML only in the main process. Worker processes
operate on sanitised cached models which avoids repeated schema checks and
keeps startup costs low.

Caching is deliberately explicit. Parsed models are written to
`models/cache/` and cleared only when the user exits the program. This
approach
allows repeated runs with different datasets without re-parsing YAML files,
while still letting contributors inspect the generated intermediate files.
