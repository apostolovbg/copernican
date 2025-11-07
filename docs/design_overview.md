# Copernican Suite Architecture
**Last Updated:** 2025-11-07

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
the suite's lint and packaging workflows.  Version 6.2.0 connects the MCMC
engine directly to `JointLike` and the new
`engine_interface.make_logposterior` helper so posterior evaluations apply
model priors, declared bounds and optional sampling transforms uniformly across
engines while surfacing component-level diagnostics for downstream analysis.
Version 6.4.0 adds an explicit `fixed` prior to `copernican_lib/priors.py`,
exposing deterministic parameters alongside the probabilistic uniform,
Gaussian and log-uniform options.  Earlier releases (6.3.0 and 6.3.1)
centralised validation, Jacobian handling and transform registration so every
engine observes consistent metadata.  The parser now normalises each prior
before it reaches the cache or an engine and rejects the retired
`distribution` alias outright.  Canonicalisation guarantees that
metadata-driven transforms, manifests and engine plugins all observe the same
schema even when the original YAML attempted to declare redundant transforms.

Version 7.2.7 keeps ``tools/update_lock.py`` import-friendly by deferring the
``piptools`` availability check until the helper actually attempts to spawn
``pip-compile``.  The lazily evaluated guard preserves the actionable guidance
for developers who need to install the pinned dependency while allowing
regression tests and lint hooks to import and monkeypatch the module without
tearing down the entire process.

Version 7.3.0 routes every Stage 2 run through :mod:`arviz` after sampling so
the engine records rank-normalised :math:`\hat{R}` values together with bulk
and tail effective sample sizes.  The diagnostics are logged, saved in the
engine result dictionary and embedded inside NetCDF exports.  Downstream tools
and publication scripts therefore consume a single source of truth for
convergence statistics without repeating calculations.

Version 7.2.6 rebuilds the ``make lock`` workflow around
``tools/update_lock.py`` so the helper owns the entire pipeline.
The helper now invokes ``pip-compile`` in a temporary workspace,
normalises the generated header, compares the body against the
repository copy and advances the ``Last Updated`` banner only when
dependencies change.  The release ships companion unit tests so
future refactors cannot resurrect the daily banner churn that
previously broke lint runs.

Version 7.2.5 promotes the resilient quadrature helper into a hard gate for
sound-horizon calculations.  ``copernican_lib.model_coder`` now raises a
``SoundHorizonComputationError`` whenever ``rs_expression`` integrals still
trigger suppressed SciPy ``IntegrationWarning`` instances.  The BAO likelihood
records the failure in its ``LikelihoodState`` metadata and aborts ratio plots
so datasets never display near-zero curves sourced from divergent integrals.

Version 7.2.3 restores the direct pass-through of neutrino-sector CAMB
arguments so model plugins can once again specify `Neff`, individual `mnuN`
entries or mass hierarchies without bespoke likelihood code while keeping the
regression harness synchronised with those choices.

Version 7.0.4 adds resilient version discovery so the macOS launcher continues
to boot even when `copernican_lib.version.get_version` is temporarily missing
during partial upgrades.  The runtime now defers attribute lookups until a
version string is required, mirroring the fallbacks in
`copernican_lib.version.get_version` so manifests and plot footers still record
``"0+unknown"`` when the helper cannot be imported directly.

Version 7.0.5 eliminates repeated DataFrame-to-NumPy conversions inside the
likelihood helpers.  `copernican_lib.likelihoods.SNeLike`, `BAOLike` and
`CMBLike` now cache immutable arrays and residual scratch buffers during
initialisation so multiprocessing workers evaluate log-likelihoods without
allocating fresh arrays on every call.  The refactor keeps engine plugins
pluggable while allowing the MCMC sampler to saturate all configured worker
processes.

Version 6.7.4 kept the multiprocessing contract intact by making both the
joint likelihood wrapper and the generated distance functions picklable.
Version 7.0.0 replaced the legacy `engine_interface` monolith with the
`copernican_lib.plugins` package and the `copernican_lib.posterior` module so
posterior construction became an explicit, picklable dataclass workflow.
Version
7.0.1 cements that transition by registering every SymPy-derived helper
on `copernican_lib.model_coder` using stable names, eliminating the
`_lambdifygenerated` pickling failures that blocked spawn pools launched from
`start.command`. Version 7.0.2 extends the protection to the plugin metadata
by replacing ``MappingProxyType`` wrappers with the picklable
``copernican_lib.plugins.FrozenMapping`` helper, ensuring engine plugins stay
serialisable across macOS and Linux spawn pools. Version 7.0.3 completes the
story by turning the symbolic distance helpers into self-rebuilding wrappers so
spawn workers reconstruct them from the cached SymPy expressions instead of
expecting parent-only module attributes.

With the retirement of the deterministic combined optimiser the suite now
ships solely with the `cosmo_engine_mcmc` backend.  Engines remain pluggable
via the `cosmo_engine_*.py` naming convention so GPU solvers or new
optimisation strategies can be introduced without altering the orchestration
logic in `copernican.py`.  The shared helpers and validation routines therefore
remain the authoritative source of truth for statistical behaviour.

To keep emcee initialisation numerically stable the sampler treats any
parameter whose lower and upper bounds are identical—or numerically
indistinguishable—as fixed.  The parser installs a canonical `type: fixed`
prior in these cases and the engine publishes the resulting constants via
`plugin.FIXED_PARAMS`.  Models such as Conformal Stationary Field Cosmology can
therefore keep the speed of light hard-coded without tripping emcee's
condition-number safeguard.  When a model defines only a handful of truly free
parameters the engine inflates the initial walker cloud adaptively until the
ensemble's condition number satisfies ``emcee``'s guardrail, so YAML plugins
with wildly different scales or exotic bound combinations no longer require
manual tuning before sampling begins.

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

Deterministic reproducibility now extends to the initial walker ensemble as
well.  The Stage 2 engine builds its NumPy generator from
``copernican_lib.utils.get_random_seed``, the same value written to the run
manifest via ``set_random_seed``.  When researchers replay a manifest with an
identical seed, the sampler yields byte-identical chains and log-probability
traces alongside the existing diagnostic summaries.

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
datasets. When a model requests scalar quadrature the generated helper calls a
resilient wrapper around SciPy's `quad`. The wrapper automatically increases
the subdivision limit and, if necessary, slices the interval into multiple
segments before retrying. Version 7.1.4 further remaps semi-infinite and
two-sided infinite integrals onto a logistic domain, inserting supportive
breakpoints automatically so exotic theories—USMFv2 included—no longer emit
repeated fallback warnings when their equations probe extreme redshifts.

Version 7.0.6 removes the automatic sound-horizon fallback. Models that
advertise BAO support must provide an `rs_expression` matching their own
`H(z)` so the integrand never injects duplicate photon terms. The regression
suite now exercises these integrals directly, comparing the generated callable
against a hand-coded quadrature to confirm perfect agreement.

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
