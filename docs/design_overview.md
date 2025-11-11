# Copernican Suite Architecture
**Last Updated:** 2025-11-11

This document expands on the high-level summary in the README by tracing how
the Copernican Suite organises its architecture.  The command-line launcher
(`copernican.py`) steers each run, the `copernican_lib/` package gathers shared
infrastructure, and the `engines/`, `models/` and `data/` directories plug into
that foundation to deliver repeatable analyses.

* `copernican.py` assembles run manifests, dispatches dataset loaders and
  prepares engine inputs so Stage 2 sampling always starts from a consistent
  configuration. The 7.5.3 release keeps the structured Stage 1 seed selector,
  surfaces every validation error encountered while loading alternative models
  and leaves a deliberate blank spacer after logging initialisation so the
  console flow stays tidy without repeating legacy "has initialised" banners.
  Version 7.6.4 keeps the per-walker progress updates introduced previously and
  now imports the timing helper explicitly so the splash screen's one-second
  pause never raises a launcher `NameError` even in heavily patched test
  harnesses.
* `copernican_lib/` contributes the reusable building blocks—data ingestion,
  posterior construction, validation checks, plotting helpers and diagnostics.
  Engines and parsers import from this package instead of reimplementing
  numerical plumbing. Version 7.6.23 moves the Stage 2 progress bar, spinner
  pump and notifier bridge into `copernican_lib.progress` so additional engines
  can reuse the console renderer without depending on the default MCMC module.
  The revision also records the very first frame each batch emits and
  routes all sampler exits through a shared cleanup path so even abrupt
  exceptions cannot leave an orphaned 0% progress bar cluttering subsequent
  diagnostics.
* `engines/` contains back ends such as the default
  ``cosmo_engine_mcmc.py``.  Engines consume `EnginePlugin` definitions,
  evaluate joint likelihoods spanning SNe Ia, BAO and CMB data and surface
  ArviZ-powered convergence diagnostics for downstream tooling.  Version 7.6.4
  reiterates the hard dependency on ArviZ so every run records R-hat and
  effective sample size summaries while keeping supporting utilities resilient
  to module-level monkeypatching.
* `models/` holds YAML descriptions that declare bounds, priors, transforms and
  dataset compatibility.  Each file is compiled into a picklable
  :class:`copernican_lib.plugins.EnginePlugin` so multiprocessing pools can
  reconstruct Stage 2 state deterministically.
* `data/` curates vetted catalogues with parser code and metadata that record
  citations, licensing information and SHA256 digests.  Loaders validate the
  digests before the observations flow into the likelihood pipeline.

### Stage 1 orchestration

`copernican.py`'s Stage 1 loop prints a dedicated banner before invoking the
seed dialog so users immediately see that reproducibility is the first
configuration step. The helper honours ``COPERNICAN_SEED`` overrides, echoes
the choice to the console and logger, and provides numbered options for
accepting the default, entering a custom integer or sampling a random seed
from the full
32-bit space. When the alternative model fails to parse or validate, or when an
engine import raises, ``_normalise_failure_reasons`` flattens exceptions into a
bullet list. ``_prompt_stage1_retry`` then presents a small menu that either
restarts Stage 1 from the top or exits gracefully, ensuring even multi-part
errors—such as missing callable hooks and incompatible bounds—are explained at
the terminal without consulting logs. The sampler questionnaire closes the
stage: it enumerates recommended defaults, allows a full restart, explains
how the fifty-character progress bars will animate during Stage 2 and lets
the operator continue, return to the summary or exit the suite entirely.
That preview now showcases the Unicode partial-block renderer introduced in
version 7.6.9, the weighted-move notifier bridge from version 7.6.10 and the
native carriage-return animator from version 7.6.14 plus the 7.6.15 cleanup
that excises the dormant `tqdm` import. The 7.6.16 refresh layers
timer-driven idle ticks so contributors can immediately see how individual
walker updates glide across the bar without spilling into multiple console
 lines, 7.6.18 keeps the historical square brackets retired while the
 regression suite confirms the bracket-free bars align identically in live
 consoles and captured logs, 7.6.19 routes idle spinner repaints through a
 background pump while dropping walker snapshot logging so the console remains
 the single source of progress updates, and 7.6.20 forces repaint cycles even
 when `emcee` batches pause while clearing finished bars so transcripts never
 retain stale progress lines.

Every run produces a timestamped output directory containing plots, NetCDF
chains and a manifest that records the engine, models, datasets, parameter
choices and Git state.  The shared workflow means new probes—such as the
planned gravitational-wave standard sirens—inherit the same orchestration as
their placeholders are consolidated under a single loader entry.

## Historical context and recent changes

Within the 7.4.5 documentation window the model library gained the
Quantum Relational Synthesis Field v2 definition. The YAML-only plugin
removes the dark sector by renormalising baryonic inertia through a
coherence kernel, reduces the free-parameter count relative to the
archived QRSF implementation and documents the entire theory in a ten-page
manuscript embedded directly in the `description` block. The update also
formalises the policy that internal model versions increment independently
of the Copernican release and that only `cosmo_model_lcdm.yml` is required
for baseline runs; all other models exist as exemplars that may evolve as
their manuscripts expand.

Version 7.4.4 finalises the Stage 5 compatibility layer by turning
`_validate_corner_inputs` into a thin wrapper that forwards directly to
`_prepare_corner_inputs`.  The approach keeps archival automation importing the
legacy name while eliminating the linter warning that appeared when the alias
was a plain assignment.  Documentation mirrors the behaviour so code comments,
guides and tests explain why the wrapper exists alongside the modern helper.

Version 7.4.6 extends that polishing work by teaching the Stage 5 corner plot
to resize itself automatically.  The new geometry helper clamps the overall
figure to twelve inches on each side, scales fonts according to the resulting
panel width and recalibrates footer spacing so text remains legible regardless
of how many parameters a sampler exposes.  The responsive sizing keeps
high-dimensional comparisons from overwhelming Matplotlib while preserving the
classic large-panel aesthetic for the familiar three-parameter ΛCDM checks.

Version 7.6.8 builds on that foundation by deepening the dual-clearance
contract.  The footer padding grows so the axes can never collide with the
text, a new floor keeps the lowest line safely above the canvas edge and the
subplot margins lift the grid to mirror the spacing used throughout the other
Stage 5 figures.  The suptitle shifts downward to match the Stage 3 and Stage 4
plots, preserving the consistent visual hierarchy while retaining the
strictly increasing contour levels, 0.015 line cadence and trimmed dataset
descriptions introduced earlier. Version 7.6.23 extends the clearance by
dropping the entire footer stack by its remaining line span so the top line
clears elongated axis labels and forthcoming gravitational-wave annotations
while staying above the shared clearance floor.

Version 7.6.9 retools the Stage 2 batch progress renderer with Unicode
partial-block glyphs while Version 7.6.10 ensures weighted `emcee` move tables
keep the notifier bridge active. Version 7.6.11 briefly handed the live display
to `tqdm`, Version 7.6.12 disabled adaptive throttling so every walker update
repainted instantly, Version 7.6.13 layered a dedicated walker-progress meter
 and spinner over the bar, Version 7.6.14 retires the third-party wrapper in
 favour of a native carriage-return renderer that keeps macOS terminals on a
 single line while mirroring every glyph into the logs, Version 7.6.15 removes
 the final dormant shim from the engine module so progress now depends solely
 on the bundled renderer, and Version 7.6.16 adds timer-driven idle ticks so
 the spinner continues to animate even when walker callbacks arrive slowly.

Version 7.4.1 adds a sampler-facing perspective to the plotting layer. The new
corner plot automatically thins oversized chains, renders the Stage 2 posterior
with enlarged panels and a footer that details how many samples survived
filtering, which stride produced the figure and whether thinning was required.
It now runs as part of Stage 5 so cosmologists can inspect parameter
degeneracies before diving into manifest tables or NetCDF chains. Earlier work
from Version 7.3.2 keeps the GW
loader in charge of the forthcoming gravitational-wave standard siren datasets.
The separate registry stays retired so discovery remains focused on a single
entry point while documentation quietly reflects the consolidation.

Version 7.3.1 refreshes the interactive prompts that guard Stage 2.  The custom
sampler questionnaire now closes with a numbered confirmation menu that spells
out how to accept, restart, back up or cancel a plan, and the workflow ends
with a matching post-run menu that distinguishes between launching another
evaluation and shutting down cleanly.  These additions mirror the broader
Copernican console style so contributors do not have to remember what terse
single-letter responses stand for.

Version 7.6.4 finalises the removal of the short-lived runtime estimator from
the sampler menu. The launcher now focuses solely on presenting the requested
plan while streaming per-walker progress updates so the fifty-character bar
fills smoothly beneath each batch heading. Operators can accept, adjust or
cancel without speculative timing data, and blank lines continue to separate
batches so long chains remain readable. The release also reiterates the hard
requirement on ArviZ so convergence summaries never quietly disappear and notes
that the splash screen now imports its delay helper explicitly to avoid
`NameError` regressions during suite launches.

Version 7.3.0 routes every Stage 2 run through :mod:`arviz` after sampling so
the engine records rank-normalised :math:`\hat{R}` values together with bulk
and tail effective sample sizes.  The diagnostics are logged, saved in the
engine result dictionary and embedded inside NetCDF exports.  Downstream tools
and publication scripts therefore consume a single source of truth for
convergence statistics without repeating calculations.
The suite now treats ArviZ as a hard dependency.
Environment provisioning fails fast when the package is missing.
Every run retains its convergence evidence.

Version 7.2.7 keeps ``tools/update_lock.py`` import-friendly by deferring the
``piptools`` availability check until the helper actually attempts to spawn
``pip-compile``.  The lazily evaluated guard preserves the actionable guidance
for developers who need to install the pinned dependency while allowing
regression tests and lint hooks to import and monkeypatch the module without
tearing down the entire process.

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
