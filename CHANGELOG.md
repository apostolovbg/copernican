# Changelog
**Last Updated:** 2025-11-08

## How to Log Changes
Add one line for each substantive commit or pull request directly under the
latest version header. Always confirm the actual current date (for example with
`date`) before logging new changes, and make sure each entry's timestamp keeps
the changelog in chronological order—never back-date entries or record future
dates. Record timestamps as ISO dates (`YYYY-MM-DD`) without times or timezone
suffixes. Follow this template:
```
## Version 1.1.0
- 2025-05-27: Added plotting and CSV (Apostol Apostolov)
- 2025-08-22: Hardened SymPy expression handling to block unsafe code and
               added security tests (OpenAI ChatGPT)

## Version 1.0.0
- 2025-05-26: Debugged copernican.py script (AI assistant)
- 2025-05-26: Created copernican.py (Apostol Apostolov)

```
## Log changes here

## Version 7.4.5
- 2025-11-08: Enlarged the Stage 5 corner plot panels, increased font sizes and
  added footer summaries describing sample filtering, thinning stride and
  legacy fallbacks so posterior figures remain readable while preserving
  compatibility. Updated regression tests, documentation references and bumped
  recorded metadata to 7.4.5 (OpenAI ChatGPT)
- 2025-11-08: Added the Quantum Relational Synthesis Field v2 model with a
  manuscript-length description, removed the dark sector from its dynamics,
  documented the ten-page description requirement, clarified that only
  `cosmo_model_lcdm.yml` is mandatory and formalised the policy of bumping
  internal model versions independently of the Copernican release (OpenAI
  ChatGPT)
- 2025-11-08: Re-encoded the QRSFv2 CAMB baryon-density mapping as a folded
  scalar so YAML parsers load the model without syntax errors (OpenAI ChatGPT)

## Version 7.4.4
- 2025-11-08: Converted the `_validate_corner_inputs` alias into a documented
  wrapper around `_prepare_corner_inputs` so Stage 5 keeps the legacy import
  path without triggering `flake8` redefinition warnings. Updated documentation
  to explain the compatibility layer and bumped recorded metadata to 7.4.4
  (OpenAI ChatGPT)

## Version 7.4.3
- 2025-11-08: Renamed the Stage 5 sampler helper to `_prepare_corner_inputs`
  while retaining `_validate_corner_inputs` as a compatibility alias so
  downstream tooling keeps importing the legacy name without tripping lints.
  Updated the regression tests, refreshed repository documentation and bumped
  recorded metadata to 7.4.3 (OpenAI ChatGPT)

## Version 7.4.2
- 2025-11-08: Restored compatibility with legacy corner-plot validators that
  still return only samples and labels by deriving thinning statistics inside
  `plotter.plot_corner`, logging the fallback, extending regression coverage and
  refreshing the documentation set while bumping recorded metadata to 7.4.2
  (OpenAI ChatGPT)

## Version 7.4.1
- 2025-11-08: Thinned Stage 2 corner plots before rendering, wired the helper
  into Stage 5 output generation, refreshed documentation and bumped recorded
  version metadata to 7.4.1 so long chains no longer stall during plotting
  (OpenAI ChatGPT)

## Version 7.4.0
- 2025-11-08: Added a corner plot to the plotting suite so Stage 2 runs expose
  sampler geometry with Copernican footers, introduced automated filename
  handling, refreshed documentation and bumped the recorded version metadata
  (OpenAI ChatGPT)

## Version 7.3.2
- 2025-11-08: Rebuilt the Quantum Relational Scale Field model with dual
  entanglement and relational-fluid channels so BAO and Supernova datasets fit
  alongside the already-strong CMB results, promoted the speed of light to a
  fixed parameter for cleaner LaTeX output and refreshed documentation to match
  the new description (OpenAI ChatGPT)
- 2025-11-07: Expanded the Quantum Relational Scale Field model description and
  abstract, refreshed the README model overview and documented the entanglement
  and relational release mechanisms so QRSF stands alone without USMF context
  (OpenAI ChatGPT)
- 2025-11-07: Consolidated the gravitational-wave standard siren placeholder
  under the GW loader, retired the redundant siren registry and refreshed
  documentation to frame the update as placeholder management ahead of the next
  dataset rollout (OpenAI ChatGPT)

## Version 7.3.1
- 2025-11-07: Replaced the sampler confirmation and post-run prompts with
  numbered menus aligned with the Copernican console style. Expanded Stage 2
  documentation to describe the clearer flows and added regression coverage for
  the new helper before bumping the recorded version to 7.3.1 (OpenAI ChatGPT)

## Version 7.3.0
- 2025-11-07: Rewrote the README introduction to highlight the suite's mission, components and supported datasets, synced the design overview summary and relocated release notes from the README into the changelog (OpenAI ChatGPT)

- 2025-11-07: Integrated ArviZ convergence diagnostics into the ensemble MCMC
  engine, logging compact :math:`\hat{R}` and effective sample size summaries,
  returning the metrics alongside sampler results, extending the regression
  suite to assert finite diagnostics and documenting publication guidance for
  the new statistics (OpenAI ChatGPT)

## Version 7.2.10
- 2025-11-07: Seeded the MCMC engine's NumPy generator from the shared
  ``copernican_lib.utils.get_random_seed`` value, added regression coverage that
  replays ``fit_sne_parameters`` with a fixed seed to confirm the resulting
  chains and log-probabilities remain identical, and documented the deterministic
  contract across the run manifest and design overview guides (OpenAI ChatGPT)

## Version 7.2.9
- 2025-11-06: Extended the setuptools include guard to cover the ``models.*``
  namespace so nested plugins remain packageable and tightened the regression
  test to assert both the include and exclude tuples stay aligned with the
  documented packaging policy (OpenAI ChatGPT)

## Version 7.2.8
- 2025-11-05: Scoped setuptools package discovery to the ``copernican_lib``,
  ``engines`` and ``models`` namespaces so macOS launchers running under the
  bundled setuptools 79.0.1 release stop failing with the "Multiple top-level
  packages discovered" error during ``pip install --no-deps .``; refreshed the
  packaging guide, bumped user-facing metadata to 7.2.8 and added regression
  coverage that enforces the include list (OpenAI ChatGPT)

## Version 7.2.7
- 2025-11-05: Deferred the ``piptools`` check in ``tools/update_lock.py`` so
  importing the helper in regression tests no longer
  triggers an unconditional ``SystemExit`` while preserving the actionable
  guidance when ``pip-compile`` genuinely runs; expanded the accompanying test
  suite and documentation to cover the lazy guard (OpenAI ChatGPT)

## Version 7.2.6
- 2025-11-05: Rebuilt the lockfile workflow around `tools/update_lock.py`,
  regenerating dependencies in a temporary workspace, preserving existing
  banners when the body is unchanged, documenting the process across the
  toolkit and adding regression tests for the helper so the `make-lock` hook
  remains deterministic (OpenAI ChatGPT)

## Version 7.2.5
- 2025-11-02: Raised a dedicated ``SoundHorizonComputationError`` when robust
  quadrature exhausts its retries, taught the BAO likelihood to stop plotting
  ratios once ``rs_expression`` integrals diverge, added regression tests that
  integrate ``\int_{z_{rec}}^{\infty} dz/(1+z)`` to ensure the failure
  propagates, refreshed documentation to describe the guardrails and bumped the
  recorded version to 7.2.5 (OpenAI ChatGPT)
- 2025-11-02: Realigned the metadata validation reference date with the updated
  documentation timestamps so CI recognizes the refreshed release metadata
  (OpenAI ChatGPT)
- 2025-11-02: Updated the metadata regression tests to read the UTC-normalised
  clock from ``tools.check_meta`` and documented the workflow for running the
  validator alongside documentation updates (OpenAI ChatGPT)

## Version 7.2.4
- 2025-11-01: Guarded autocorrelation estimation against undersized chains in
  the MCMC engine, added a regression test covering the short-chain scenario,
  refreshed diagnostics documentation and bumped the recorded version to 7.2.4
  (OpenAI ChatGPT)

## Version 7.2.3
- 2025-11-01: Synced the functional CAMB regression test with the restored
  neutrino-sector pass-through so cached :math:`D_\ell` spectra match direct
  solver calls, refreshed documentation to describe the alignment and bumped
  project metadata to 7.2.3 (OpenAI ChatGPT)

## Version 7.2.2
- 2025-11-01: Restored the full neutrino-sector mapping for the CAMB helpers,
  mirrored the configuration across the cached background observables, added
  regression coverage that compares helper outputs against direct CAMB calls and
  refreshed the architecture notes to highlight the restored pass-through
  (OpenAI ChatGPT)

## Version 7.2.1
- 2025-11-01: Returned :math:`D_\ell` spectra from the CAMB helper, restored a
  controlled BAO background fallback that reuses model distance functions when
  CAMB parameters are unavailable, relaxed BAO covariance validation to fall
  back to diagonal errors for trusted datasets and bumped the recorded version
  to 7.2.1 (OpenAI ChatGPT)
- 2025-11-01: Added regression coverage confirming the BAO likelihood falls back
  to model distance functions when CAMB parameters are unavailable (OpenAI
  ChatGPT)

## Version 7.2.0
- 2025-11-01: Routed BAO likelihood distances and sound-horizon evaluations
  through the CAMB helpers shared with the CMB module, enforced positive-
  definite BAO covariance matrices with condition-number reporting, validated
  CAMB parameter maps in the engine interface, recorded CAMB configuration
  details in run manifests, refreshed the sample models with explicit neutrino
  sector parameters, added dedicated CAMB background tests and bumped the suite
  version to 7.2.0 (OpenAI ChatGPT)

## Version 7.1.4
- 2025-11-01: Extended the resilient quadrature helper with logistic
  remapping for infinite bounds, automatic breakpoint seeding and expanded
  regression coverage so USMFv2-class models complete without repeated
  fallback warnings, and refreshed the documentation plus recorded version
  metadata (OpenAI ChatGPT)

## Version 7.1.3
- 2025-11-01: Hardened the symbolic quadrature pipeline with automatic limit
  escalation, interval subdivision and targeted logging so wild theories such
  as USMFv2 complete without SciPy ``IntegrationWarning`` spam, refreshed
  documentation to describe the resilience improvements, bumped the recorded
  version and added regression tests for the new helper (OpenAI ChatGPT)

## Version 7.1.2
- 2025-11-01: Refreshed every launcher with a concise primary menu, an
  environment-management submenu and blank-line separators, added a guided
  sampler questionnaire after CMB loading, updated documentation, synced the
  recorded version and adjusted start-script tests for the new flows (OpenAI
  ChatGPT)

## Version 7.1.1
- 2025-11-01: Normalised every runtime timestamp to Coordinated Universal Time
  (UTC) across logging, manifests and filenames, updated metadata
  validators and pre-commit checks to read the UTC clock, added targeted
  unit coverage for the new helpers, refreshed documentation, and bumped
  the recorded version (OpenAI ChatGPT)

## Version 7.1.0
- 2025-11-01: Added an interactive Stage 2 sampler configuration menu that
  records production steps, burn-in length, walker counts and pool sizes,
  ensured the MCMC engine honours explicit pool selections when sizing the
  ensemble, persisted the sampler plan in parameter summaries, refreshed
  documentation, bumped the recorded version and extended regression tests
  for the new metadata (OpenAI ChatGPT)

## Version 7.0.6
- 2025-10-31: Retired the sound-horizon fallback, enforced explicit
  ``rs_expression`` definitions, updated bundled models with integral
  expressions, expanded unit tests, refreshed documentation and bumped the
  recorded version (OpenAI ChatGPT)

## Version 7.0.5
- 2025-10-31: Cached SNe, BAO and CMB likelihood inputs as immutable NumPy
  arrays to remove per-call DataFrame conversions, reusing residual buffers to
  accelerate multiprocessing, added regression tests covering the caching
  behaviour and refreshed documentation and metadata (OpenAI ChatGPT)

## Version 7.0.4
- 2025-10-31: Hardened runtime version discovery so the macOS launcher and
  plotting stack tolerate missing ``copernican_lib.version.get_version`` during
  partial upgrades, added regression tests covering the new fallbacks and
  refreshed documentation and metadata (OpenAI ChatGPT)

## Version 7.0.3
- 2025-10-31: Wrapped SymPy-generated distance helpers in self-reconstructing
  callables so spawn-based multiprocessing workers rebuild them from cached
  expressions, refreshed the regression tests and documentation, and bumped
  suite metadata (OpenAI ChatGPT)

## Version 7.0.2
- 2025-10-31: Replaced ``MappingProxyType`` wrappers inside engine plugins with
  a picklable ``FrozenMapping`` helper, restored spawn-pool compatibility,
  added regression coverage for plugin pickling and refreshed suite metadata
  (OpenAI ChatGPT)

## Version 7.0.1
- 2025-10-31: Registered SymPy-generated distance helpers as module-level
  callables so spawn-based pools launched from the macOS bootstrap
  script remain stable, restored start.command usability, added
  regression tests, and updated documentation and metadata (OpenAI
  ChatGPT)

## Version 7.0.0
- 2025-10-31: Replaced the legacy engine interface with the picklable
  `copernican_lib.plugins` package and a standalone posterior module, ensured
  log-uniform transforms serialise cleanly, refreshed validation and
  documentation, added regression tests covering posterior pickling and bumped
  suite metadata (OpenAI ChatGPT)

## Version 6.7.4
- 2025-10-31: Made joint likelihood adapters and generated distance functions
  picklable so spawn-based pools no longer crash, relaxed plugin validation when
  distance metrics are disabled, added an optional burn-in override to
  ``fit_sne_parameters`` and trimmed MCMC-heavy tests to keep CI fast. Updated
  documentation and metadata accordingly (OpenAI ChatGPT)

## Version 6.7.3
- 2025-10-31: Replaced the nested posterior closure with a picklable adapter so
  spawn-based multiprocessing pools can evaluate it, tightened unit coverage and
  refreshed documentation and metadata (OpenAI ChatGPT)

## Version 6.7.2
- 2025-10-31: Removed `pip-tools` from runtime installs while retaining the
  familiar developer workflow, refactored the Stage 2 log-probability adapter
  so multiprocessing workers can pickle it reliably, added regression tests
  for the new helper, refreshed dependency documentation and bumped suite
  metadata (OpenAI ChatGPT)

## Version 6.7.1
- 2025-10-31: Ensured sampler progress logs enumerate every parameter, reused
  diagnostic buffers to cut callback overhead, wrapped walker snapshots, updated
  documentation, extended regression coverage and fixed lint issues (OpenAI
  ChatGPT)

## Version 6.7.0
- 2025-10-31: Added granular sampler diagnostics with walker snapshots, auto-
  configured multiprocessing, live BAO/CMB residual logging, regression tests,
  documentation refreshes and bumped suite metadata (OpenAI ChatGPT)

## Version 6.6.0
- 2025-10-31: Enabled joint SNe/BAO/CMB sampling in the MCMC engine, updated
  Stage 2 orchestration and downstream reporting to reuse the combined
  likelihood diagnostics, refreshed documentation, expanded regression tests
  and bumped suite metadata (OpenAI ChatGPT)

## Version 6.5.4
- 2025-10-31: Allowed "Last Updated" markers within the first three lines of
  tracked files, removed time components from metadata fields, updated the CI
  checks accordingly, refreshed documentation, and bumped suite metadata
  (OpenAI ChatGPT)

## Version 6.5.3
- 2025-10-30: Ensured the managed launchers bootstrap `pip` with
  `ensurepip` and a `get-pip.py` fallback so dependency installations never
  fail on fresh interpreters, refreshed the quick-start documentation, and
  bumped suite metadata (OpenAI ChatGPT)

## Version 6.5.2
- 2025-10-30: Hardened all launchers to purge Python 3.12 interpreters, added
  explicit range guards to the bootstrap tests, refreshed documentation and
  metadata, and bumped the recorded suite version (OpenAI ChatGPT)

## Version 6.5.1
- 2025-10-30: Reverted the managed interpreter to Python 3.11 across all
  launchers so CAMB wheels install on macOS again, tightened packaging
  metadata to block Python 3.12 environments until upstream wheels ship,
  updated CI matrices, documentation and metadata, and bumped the suite
  version (OpenAI ChatGPT)

## Version 6.5.0
- 2025-10-30: Centralised SNe/BAO/CMB dataset loading, recorded dataset names,
  versions and independence statements in manifests, documented the new
  `run_config` schema, refreshed metadata and bumped suite metadata (OpenAI
  ChatGPT)

## Version 6.4.0
- 2025-10-30: Added an explicit `fixed` prior class with canonical
  normalisation, enforced strict `type` fields in the model schema, promoted
  equal-bound parameters to deterministic metadata in plugins, refreshed
  models, documentation and regression tests, and bumped suite metadata
  accordingly (OpenAI ChatGPT)

## Version 6.3.1
- 2025-10-30: Normalised parameter prior mappings during model parsing,
  tightened validation errors, refreshed documentation, expanded regression
  tests and bumped suite metadata (OpenAI ChatGPT)

## Version 6.3.0
- 2025-10-30: Added `copernican_lib/priors.py` with reusable prior classes,
  extended model validation with log-uniform support, refreshed documentation,
  expanded prior tests and bumped the suite version (OpenAI ChatGPT)

## Version 6.2.0
- 2025-10-30: Rewrote development laws to enforce chronological date checks,
  normalised incorrect timestamps across documentation, and refreshed metadata
  that slipped into the future (OpenAI ChatGPT)
- 2025-10-30: Integrated JointLike-powered posterior assembly in the MCMC
  engine, exposed `engine_interface.make_logposterior` for reusable prior
  handling, expanded smoke tests with likelihood diagnostics, refreshed
  documentation metadata and bumped the suite version (OpenAI ChatGPT)

## Version 6.1.1
- 2025-02-14: Restored import ordering in the likelihood package to satisfy
  style linters, refreshed documentation metadata, and bumped the suite
  version (OpenAI ChatGPT)

## Version 6.1.0
- 2025-10-30: Introduced the `copernican_lib/likelihoods` package with reusable
  dataset log-likelihood helpers, migrated χ² logic out of `statistics.py`,
  added a configurable joint likelihood aggregator, refreshed documentation
  and bumped suite metadata (OpenAI ChatGPT)

## Version 6.0.14
- 2025-10-30: Normalised the dependency lock workflow by dropping the
  Python interpreter banner, ensured the `make lock` helper keeps
  cross-platform runs byte-identical, refreshed documentation and
  bumped suite metadata (OpenAI ChatGPT)

## Version 6.0.13
- 2025-10-30: Normalised metadata and policy check outputs across Windows and
  POSIX paths, pinned the lint workflow to pip-tools 7.4.1, made the lock
  target explicit about --strip-extras and bumped suite metadata (OpenAI
  ChatGPT)

## Version 6.0.12
- 2025-10-30: Added repository policy pre-commit checks for metadata dates,
  version synchronisation and print-free libraries, expanded lint hooks and
  documented the CI `pre-commit run --all-files` enforcement (OpenAI ChatGPT)

## Version 6.0.11
- 2025-10-30: Removed `pip` and `pip-tools` from the runtime lock so Windows
  runs no longer attempt to replace the active installer, regenerated the
  dependency snapshot, refreshed CI and developer guidance, and bumped the
  recorded suite metadata (OpenAI ChatGPT)

## Version 6.0.10
- 2025-10-30: Rebuilt the dependency lock against currently published
  wheels, pinned the bootstrapper to `pip==24.2`, updated CI workflows to
  honour the stable installer and refreshed documentation and metadata so
  Windows, macOS and Linux jobs all resolve packages without source builds
  (OpenAI ChatGPT)

## Version 6.0.9
- 2025-10-30: Added a cross-platform GitHub Actions CI matrix for Python 3.12,
  cached pip and CAMB assets, automated testing, packaging artifact uploads,
  refreshed the documentation, stabilised the dependency lock hook by pinning
  its pip toolchain and bumped the recorded suite version (OpenAI ChatGPT)

## Version 6.0.8
- 2025-10-30: Enforced Python 3.12+ across all start launchers, rebuilt the
  dependency lock with the released ArviZ 0.22.0 for NumPy 2 support,
  refreshed documentation and bumped suite metadata (OpenAI ChatGPT)

## Version 6.0.7
- 2025-10-30: Added a metadata validation script that enforces synchronized
  release numbers and prevents future-dated documentation, refreshed release
  notes and normalized Last Updated timestamps across the suite (OpenAI
  ChatGPT)

## Version 6.0.6
- 2025-10-29: Added a guarded parameter extraction helper so BAO and CMB
              stages skip models whose SNe fits fail instead of raising
              KeyError, updated documentation and added regression tests for
              the fallback path (OpenAI ChatGPT)

## Version 6.0.5
- 2025-10-30: Classified numerically locked parameters before sampling,
              introduced adaptive walker initialisation to defeat emcee's
              condition-number guard and added regression tests covering the
              helper utilities so arbitrary YAML models remain supported
              (OpenAI ChatGPT)

## Version 6.0.4
- 2025-10-29: Hardened the MCMC sampler to exclude fixed-bound parameters
              from the active subspace so constant entries no longer trigger
              emcee's condition-number guard and added regression coverage for
              the Conformal Stationary Field Cosmology plugin (OpenAI ChatGPT)

## Version 6.0.3
- 2025-10-29: Rebuilt all non-\LambdaCDM model YAMLs with explicit
              `python_var` mappings, safe expressions and documentation links
              so they load without parser errors and serve as future-ready
              examples (OpenAI ChatGPT)

## Version 6.0.2
- 2025-10-29: Removed the tracked dependency cache directory and documented
              the `.cache/` workflow so Git only sees per-user data
              (OpenAI ChatGPT)

## Version 6.0.1
- 2025-10-29: Restored the README `Last Updated` value to the human-specified
              date, codified the timestamp verification guideline in
              `AGENTS.md` and reaffirmed the need to understand prior human
              changes before altering them (OpenAI ChatGPT)
- 2025-10-29: Added a README banner reference for the refreshed
              Copernican Suite artwork so the documentation opens
              with the updated visual identity once the asset is
              supplied (OpenAI ChatGPT)
- 2025-10-30: Added a tracked VERSION file, taught the runtime helper to read
              it before falling back to setuptools_scm, embedded the suite
              version in run manifests, expanded packaging guidance and
              refreshed documentation for the new workflow (OpenAI ChatGPT)
- 2025-10-29: Retired the repository roadmap formerly stored in `PLAN.md`,
              confirmed no remaining references and documented the removal
              (OpenAI ChatGPT)

## Version 6.0.0
- 2025-10-28: Retired the combined optimiser module, promoted the MCMC sampler
              to the default pluggable engine, updated the CLI, tests and
              documentation to reflect the single-engine architecture and
              reiterated verbose progress reporting (OpenAI ChatGPT)

## Version 5.0.0
- 2025-10-27: Replaced the legacy combined optimiser with
              ``engines.cosmo_engine``, added verbose percentage-based
              progress reporting to the MCMC backend, refreshed all
              documentation and bumped suite metadata (OpenAI ChatGPT)

## Version 4.3.26
- 2025-10-26: Reseeded invalid MCMC walkers to eliminate emcee warnings,
              copied SNe chi-squared totals into summary outputs, reused
              posterior chains when `MODEL_FILENAME` matches so BAO/CMB
              overlays and χ² values stay aligned during LCDM self-tests,
              refreshed documentation and hardened tests for the new helper
              (OpenAI ChatGPT)

## Version 4.3.25
- 2025-10-25: Extracted shared chi-squared helpers into
              ``copernican_lib.statistics``, overhauled the MCMC engine to
              initialise walkers uniformly, run a dedicated burn-in and record
              diagnostics, reused SNe chains when models match so BAO/CMB
              overlays align during self-comparisons, refreshed documentation
              across the suite and bumped metadata (OpenAI ChatGPT)

## Version 4.3.24
- 2025-10-23: Hardened plot summaries against missing chi-squared totals,
              added regression tests, refreshed documentation and bumped the
              suite metadata (OpenAI ChatGPT)

## Version 4.3.23
- 2025-10-23: Replaced the MCMC penalty sentinel with ``-np.inf``, updated
              tests, documentation and metadata to describe the deterministic
              rejection behaviour (OpenAI ChatGPT)

## Version 4.3.22
- 2025-10-23: Added a cached dependency scan so repeated launches skip costly
              AST parsing, introduced targeted tests, refreshed documentation
              and metadata across the suite (OpenAI ChatGPT)

## Version 4.3.21
- 2025-10-22: Precomputed Windows bootstrap release metadata outside
              conditional blocks so `%DOWNLOAD_URL%` expands reliably,
              kept the empty-URL guard, verified the other launchers
              remain stable and refreshed suite documentation
              (OpenAI ChatGPT)

## Version 4.3.20
- 2025-10-05: Moved the Windows launcher PowerShell invocations into helper
              subroutines to avoid `cmd.exe` parsing bugs, confirmed the
              bootstrap menu launches cleanly and refreshed documentation and
              metadata (OpenAI ChatGPT)

## Version 4.3.19
- 2025-09-30: Hardened the launchers to validate the Python download URL, pass
              strict arguments to PowerShell and surface empty URL errors on
              all platforms; documented the guard and bumped suite metadata
              (OpenAI ChatGPT)

## Version 4.3.18
- 2025-09-28: Guarded the Windows launcher download flow by exporting the
              URL through environment variables, validating it before the
              PowerShell download step and extending documentation to
              explain the hardened bootstrap (OpenAI ChatGPT)

## Version 4.3.17
- 2025-09-26: Repaired the Windows launcher so it builds a valid
              Python download URL, pre-creates the `.python` directory,
              documents the fix and bumps the suite metadata
              (OpenAI ChatGPT)

## Version 4.3.16
- 2025-09-22: Reconfigured the pre-commit `make lock` hook to provision
              `pip-tools` automatically so dependency refreshes succeed in CI
              and during local linting (OpenAI ChatGPT)

## Version 4.3.15
- 2025-09-22: Switched the dependency lock automation to
              `python -m piptools compile`, refreshed documentation and
              regenerated the lock file to keep the managed environment
              reproducible (OpenAI ChatGPT)

## Version 4.3.14
- 2025-09-22: Bundled pip-tools with locked dependencies, refreshed the lock
               file, documentation and licensing metadata so `make lock`
               always succeeds inside the managed environment (OpenAI ChatGPT)

## Version 4.3.13
- 2025-09-03: Closed NetCDF handle in MCMC test to resolve Windows temp file cleanup (OpenAI ChatGPT)
- 2025-09-03: Installed pre-commit with dependencies in CI to fix missing cfgv import (OpenAI ChatGPT)

## Version 4.3.12
- 2025-09-02: Removed dependency hash verification and related tooling, tests and documentation (OpenAI ChatGPT)


## Version 4.3.11
- 2025-09-02: Derived wheel tags from the running Python version to drop
               hard-coded cp311 references in hash refresher and tests
               (OpenAI ChatGPT)

## Version 4.3.10
- 2025-09-01: Pinned setuptools and extended hash refresher to cover cp311
              wheels and other unsafe packages, preventing hash-mode
              install failures (OpenAI ChatGPT)

## Version 4.3.9
- 2025-09-01: Added pytest and Windows colorama dependency to lock file and
              refreshed hashes to fix failing tests (OpenAI ChatGPT)

## Version 4.3.8
- 2025-09-01: Included stable-ABI wheels in hash refresher and refreshed
              pyerfa hashes for all platforms (OpenAI ChatGPT)

## Version 4.3.7
- 2025-09-01: Added universal2 wheel support in hash helper and refreshed
              dependency hashes (OpenAI ChatGPT)

## Version 4.3.6
- 2025-09-01: Automated wheel hash recreation and fixed contourpy macOS ARM
              hash to unblock CI (OpenAI ChatGPT)

## Version 4.3.5
- 2025-09-01: Added macOS and Windows wheel hashes for contourpy==1.3.3
              to support cross-platform installs (OpenAI ChatGPT)

## Version 4.3.4
- 2025-09-01: Refreshed dependency lock file (OpenAI ChatGPT)


## Version 4.3.3
- 2025-09-01: Added automated hash locking and pre-commit hook for dependency
              updates; documented new workflow (OpenAI ChatGPT)

## Version 4.3.2
- 2025-09-01: start scripts fetch Python 3.12.11 from astral-sh releases
              (OpenAI ChatGPT)

## Version 4.3.1
- 2025-08-30: Removed outdated CLI examples, revised menu and seed tests,
              and clarified external authentication prompts in LICENSE
              (OpenAI ChatGPT)
- 2025-08-30: Split CI into dedicated lint and test workflows using
              Python 3.12 (OpenAI ChatGPT)

## Version 4.3.0
- 2025-08-30: Removed the command-line seed flag in favour of an interactive
              seed prompt with manual and random options; updated manifest,
              utilities, tests and documentation (OpenAI ChatGPT)

## Version 4.2.1
- 2025-08-30: Added package manager password notices in launchers and
              updated README and LICENSE (OpenAI ChatGPT)

## Version 4.2.0
- 2025-08-31: Replaced CLI flags with menu-driven launchers and environment
               variables; updated tests and documentation (OpenAI ChatGPT)

## Version 4.1.0
- 2025-08-30: Launchers bootstrap a private Python 3.12+ and ignore system
              interpreters; updated documentation (OpenAI ChatGPT)

## Version 4.0.0
- 2025-08-31: Require Python 3.12+, updated launchers and docs, added 3.12 wheel hashes (OpenAI ChatGPT)
- 2025-08-30: Added dependency update law and synced policies (OpenAI ChatGPT)

## Version 3.13.11
- 2025-08-30: Added macOS NumPy hash to fix start script installs
              (OpenAI ChatGPT)
## Version 3.13.10
- 2025-08-30: Vectorised distance integrals and finite penalties in MCMC
              engine to prevent hangs and warnings (OpenAI ChatGPT)
## Version 3.13.9
- 2025-08-30: Pinned typing_extensions and dependency tree for hash-locked installs (OpenAI ChatGPT)
## Version 3.13.8
- 2025-08-29: Pinned h5py dependency for hash-locked installs (OpenAI ChatGPT)

## Version 3.13.7
- 2025-08-29: Pinned xarray-einstats dependency to satisfy hash-locked installs (OpenAI ChatGPT)

## Version 3.13.6
- 2025-08-29: Allowed `COPERNICAN_VERSION` to override runtime version and
               documented custom prerelease builds (OpenAI ChatGPT)

## Version 3.13.5
- 2025-08-28: Pinned h5netcdf dependency for ArviZ to satisfy
              hash-locked installs (OpenAI ChatGPT)

## Version 3.13.4
- 2025-08-28: Pinned packaging dependency with hashes for reproducible
              installs (OpenAI ChatGPT)

## Version 3.13.3
- 2025-08-28: Added cross-platform wheel hashes and fixed Windows pip upgrade
              in CI (OpenAI ChatGPT)

## Version 3.13.2
- 2025-08-28: Replaced ArviZ VCS dependency with pinned commit archive
              (OpenAI ChatGPT)

## Version 3.13.1
- 2025-08-28: Pinned ArviZ to upstream commit and simplified dependency
              installation across launchers and CI (OpenAI ChatGPT)

## Version 3.13.0
- 2025-08-28: Added result writer for parameter summaries and exposed
              covariance matrices from optimisation and MCMC engines
              (OpenAI ChatGPT)

## Version 3.12.0
- 2025-08-27: Added a command-line seed flag, seeded Python and engine RNGs
               and logged the value in manifest and logs (OpenAI ChatGPT)

## Version 3.12.1
- 2025-08-28: Enforced use of repository virtual environment, added laws on
  testing and dependency licensing, and worked around ArviZ's NumPy pin in
  start scripts (OpenAI ChatGPT)

## Version 3.11.2
- 2025-08-27: Logged SHA256 digests for dataset files and propagated them
               through the run manifest (OpenAI ChatGPT)

## Version 3.11.1
- 2025-08-27: Added xarray to locked dependencies and documented automatic
  installation of emcee, xarray and ArviZ (OpenAI ChatGPT)

## Version 3.11.0
- 2025-08-27: Added emcee-based MCMC engine, per-run output folders and
              NetCDF chain writer (OpenAI ChatGPT)

## Version 3.10.0
- 2025-08-28: Added run manifest with dataset hashes and Git metadata, SHA256
  helper and accompanying tests and documentation (OpenAI ChatGPT)
- 2025-08-27: Added parameter priors with parser and engine support (OpenAI ChatGPT)

## Version 3.9.31
- 2025-08-27: Parallelised combined χ² computation and added tests and
              docs (OpenAI ChatGPT)
- 2025-08-27: Added Last Updated fields and clarified development rules
              (OpenAI ChatGPT)

## Version 3.9.30
- 2025-08-27: Refactored BAO χ² to accept arrays and updated tests
              (OpenAI ChatGPT)

## Version 3.9.29
- 2025-08-26: Externalised BAO plugin validation and updated tests
              (OpenAI ChatGPT)

## Version 3.9.28
- 2025-08-26: Throttled optimisation progress updates and added tests
              (OpenAI ChatGPT)

## Version 3.9.27
- 2025-08-26: start.command handles unset VIRTUAL_ENV (OpenAI ChatGPT)
- 2025-08-26: documented start script parity law (OpenAI ChatGPT)
- 2025-08-26: added strict security compliance law (OpenAI ChatGPT)

## Version 3.9.26
- 2025-08-25: Routed optimisation progress to ``stdout`` so runs no longer
              appear to hang on Linux (OpenAI ChatGPT)

## Version 3.9.25
- 2025-08-25: Flushed console output to prevent apparent hangs on Linux and
              restricted detailed environment information to the log file
              (OpenAI ChatGPT)

## Version 3.9.24
- 2025-08-25: start.sh guards against unset VIRTUAL_ENV to prevent
              startup errors (OpenAI ChatGPT)

## Version 3.9.23
- 2025-08-25: Hardened parser discovery against symlink escapes and expanded
              security tests (OpenAI ChatGPT)

## Version 3.9.22
- 2025-08-26: Capped expression complexity in get_camb_params and added
              stress tests (OpenAI ChatGPT)

## Version 3.9.21
- 2025-08-25: Prepended license notice to test modules (OpenAI ChatGPT)

## Version 3.9.20
- 2025-08-25: start.sh installs project with --no-deps to avoid implicit
              dependency resolution (OpenAI ChatGPT)

## Version 3.9.20
- 2025-08-28: Added cross-platform wheel hashes for NumPy and SciPy in requirements.lock (AI assistant)

## Version 3.9.19
- 2025-08-24: start.command exits on unset variables for stricter error
              handling (OpenAI ChatGPT)

## Version 3.9.18
- 2025-08-24: start.bat and start.command install hashed dependencies and
              isolate project install with --no-deps (OpenAI ChatGPT)

## Version 3.9.17
- 2025-08-24: Normalised parser hash computation for cross-platform
               verification and refreshed trusted hashes (OpenAI ChatGPT)
- 2025-08-24: Clarified data directory policy to allow parser and metadata
               edits (OpenAI ChatGPT)

## Version 3.9.16
- 2025-08-24: Fixed BAO compound parser registration to honour dataset_id
               (OpenAI ChatGPT)

## Version 3.9.15
- 2025-08-23: Nested developer guide sections and required tests to pass
               before commits (OpenAI ChatGPT)

## Version 3.9.14
- 2025-08-23: Clarified development laws section link and heading
               (OpenAI ChatGPT)
- 2025-08-23: Established documentation refresh policy and aligned version
               numbers across metadata (OpenAI ChatGPT)

## Version 3.9.13
- 2025-08-23: start.sh exits on unset variables for stricter error handling
               (AI assistant)

## Version 3.9.12
- 2025-08-23: Added security test ensuring rogue parsers are ignored
               (AI assistant)

## Version 3.9.11
- 2025-08-23: Linted dataset parsers and removed data exclusion from
  pre-commit (AI assistant)

## Version 3.9.10
- 2025-08-23: Pinned pyproject dependencies to requirements.lock and
  documented joint regeneration (AI assistant)

## Version 3.9.9
- 2025-08-23: start.sh installs dependencies with hash verification before
  package installation (AI assistant)

## Version 3.9.8
- 2025-08-23: Added CITATION.cff and referenced it from README (AI assistant)
- 2025-08-23: Embedded third-party license texts and documented CAMB LGPL
  obligations (AI assistant)

## Version 3.9.7
- 2025-08-23: Normalized parser path separators so trusted hashes work on all
  platforms (AI assistant)

## Version 3.9.6
- 2025-08-23: Verified parser modules against trusted hashes and skipped
  untrusted files (AI assistant)

## Version 3.9.5
- 2025-08-23: Replaced ``eval`` in model compilation with AST-based execution
  and expanded tests for integral handling (AI assistant)

## Version 3.9.4
- 2025-08-23: Prepended license notices to start scripts (AI assistant)
- 2025-08-23: Locked runtime dependencies and enforced hash-verified
  installation (AI assistant)
- 2025-08-23: Expanded documentation and updated dependency instructions
  (AI assistant)

## Version 3.9.3
- 2025-08-23: Replaced ad-hoc metadata parser with strict YAML loader and
  added tests rejecting invalid YAML (AI assistant)

## Version 3.9.2
- 2025-08-23: Updated README version and Last Updated date (AI assistant)
- 2025-08-22: Wrapped metadata citations with YAML folded blocks and line
  breaks (AI assistant)
- 2025-08-22: Updated licenses for GW and siren placeholders (AI assistant)

## Version 3.9.1
- 2025-08-22: Replaced ``eval`` in CAMB parameter parsing with a safe
  AST-based evaluator and added malicious expression tests (AI assistant)

## Version 3.9.0
- 2025-08-22: Documented third-party licenses and linked from README
  (AI assistant)
- 2025-08-22: Added LICENSE.md references to module headers (AI assistant)
- 2025-08-22: Prompted before installing dependencies and added `--yes` flag
  for CI automation (AI assistant)

## Version 3.8.4
- 2025-08-22: Added dataset license references and updated documentation
  (AI assistant)

## Version 3.8.3
- 2025-08-22: Updated README version to 3.8.3 (AI assistant)
- 2025-08-22: Dropped JSON input from the compound BAO parser and updated
  documentation to reference YAML only (AI assistant)

## Version 3.8.2
- 2025-08-21: Logged previously silent exceptions in `copernican.py`,
  `copernican_lib/utils.py` and `engines/cosmo_engine_comb.py` (AI assistant)

## Version 3.8.1
- 2025-08-21: Removed unused `get_user_input_filepath` and `parse_model_header`
  helpers from `copernican.py` (AI assistant)

## Version 3.8.0
- 2025-08-21: Added NumPy/SciPy sanity checks before heavy computations to
  diagnose CPU feature mismatches (AI assistant)

## Version 3.7.0
- 2025-08-21: Forwarded Python warnings to logger and added strict warning flag
  for CI reproducibility (AI assistant)

## Version 3.6.27
- 2025-08-21: Logged Python version, OS, CPU and package versions
  after logging setup (AI assistant)

## Version 3.6.26
- 2025-08-21: Added crash signal handlers dumping stack traces to log and
  console (AI assistant)

## Version 3.6.25
- 2025-08-20: start.command recreates missing virtual environments and
  advises reinstalling Python when activation scripts remain absent (AI
  assistant)

## Version 3.6.24
- 2025-08-19: start.bat verifies '.venv\Scripts\activate.bat' exists,
  recreating the environment once and advising on missing 'venv' support
  before exiting (AI assistant)
- 2025-08-19: start.sh retries virtual environment creation when the
  activation script is missing and advises installing 'python3.11-venv'
  if the second attempt fails (AI assistant)

## Version 3.6.23
- 2025-08-19: Read ``latex_mappings.yml`` using UTF-8 for cross-platform
  Unicode safety (AI assistant)

## Version 3.6.22
- 2025-08-19: Replace legacy CI with pull-request-only ``Tests`` workflow and
  document behaviour (AI assistant)
- 2025-08-19: ``console_output.write`` now degrades gracefully on consoles
  lacking Unicode support (AI assistant)

## Version 3.6.21
- 2025-08-19: Rename CI job to 'test' for clarity (AI assistant)
- 2025-08-19: Remove 'build/' before and after 'pip install .' in start
  scripts, document cleanup and ignore the directory (AI assistant)

## Version 3.6.20
- 2025-08-19: start.sh checks for missing 'python3.11-venv' after creating
  '.venv' and prints installation hint (AI assistant)

## Version 3.6.19
- 2025-08-18: start.sh resolves absolute path before re-executing
  (AI assistant)

## Version 3.6.18
- 2025-08-18: start.command resolves absolute path; README notes macOS should
  run `./start.command` (AI assistant)

## Version 3.6.17
- 2025-08-18: Document launcher enforcement of Python 3.11+ with automatic
  `.venv` setup and OS install hints (AI assistant)

## Version 3.6.16
- 2025-08-18: Parse interpreter '--version' in start scripts and use
  'py -3.11' for virtual environments (AI assistant)

## Version 3.6.15
- 2025-08-18: start.command and start.bat check for Python 3.11+ and show
  install hints before creating the virtual environment (AI assistant)

## Version 3.6.14
- 2025-08-17: start.sh enforces Python 3.11+ and prints OS install hints
  (AI assistant)

## Version 3.6.13
- 2025-08-16: Expanded README and in-source docstrings; broadened
  documentation across `docs/` (AI assistant)
- 2025-08-15: Expanded packaging guide with Python 3.11 install and build docs
  (AI assistant)
- 2025-08-15: Archived PyInstaller spec files and streamlined CI to use a
  cached `.venv` for linting and tests (AI assistant)
- 2025-08-15: Replaced PyInstaller references with start script and `.venv`
  instructions in documentation (AI assistant)
- 2025-08-15: Automatically install missing packages and enforce `.venv`
  usage during dependency checks (AI assistant)
- 2025-08-15: Start scripts now create and reuse a local virtual environment,
  installing dependencies automatically (AI assistant)
- 2025-08-13: Added regression tests for BOSS DR12 BAO parsing and LCDM
  chi-squared residuals (AI assistant)

- 2025-08-13: Use full BAO covariance when available and test coverage
  (AI assistant)
- 2025-08-12: Forward CLI args in start.command; wrap comments (AI assistant)
- 2025-08-11: Improve CI to export Python path, build universal2 macOS
  binaries and verify Copernican.app artifact (AI assistant)
- 2025-08-11: Wrapped long lines across docs and scripts for readability (AI
  assistant)

- 2025-08-11: Specify OS shells in CI, validate binaries with --help and
  enumerate hidden imports in spec files (AI assistant)

- 2025-08-12: Expanded dataset overview with parser and covariance details,
  documenting the compound BAO dataset (AI assistant)
- 2025-08-12: Revamped test suite with verbose logging, bounded optimiser
  iterations and explicit dataset paths (AI assistant)
- 2025-08-13: Standardised `dataset_id` metadata and output filenames
  (AI assistant)

- 2025-08-14: Require `dataset_id` for data loaders, revamp registries,
  update tests and documentation (AI assistant)

## Version 3.6.12

- 2025-08-11: Update documentation version strings to 3.6.12 (AI assistant)
- 2025-08-11: Prepared 3.6.12 release and opened new Unreleased section (AI
  assistant)
- 2025-08-11: Set formatter line length to 79 and wrap existing lines (AI
  assistant)
- 2025-08-11: Load BAO parser via importlib in tests to avoid package import
  errors (AI assistant)

## Version 3.6.11

- 2025-08-11: Update documentation version strings to 3.6.11 (AI assistant)
- 2025-08-11: Remove ``target_arch`` from macOS spec on non-mac systems to
  fix Linux and Windows CI builds (AI assistant)
- 2025-08-11: Make macOS PyInstaller spec use universal2 only on macOS to
  prevent CI failures (AI assistant)
- 2025-08-11: Propagate ``target_arch`` to the macOS bundle to keep universal2
  builds working (AI assistant)

## Version 3.6.10

- 2025-08-10: Fix CI pre-commit invocation to use correct module name (AI
  assistant)
- 2025-08-10: Ensure macOS build uses universal2 Python and document
  requirement (AI assistant)

## Version 3.6.9

- 2025-08-10: Use per-OS PyInstaller specs and archive dist/ (AI assistant)

## Version 3.6.8

- 2025-08-10: Prepared 3.6.8 release and opened new Unreleased section (AI
  assistant)
- 2025-08-10: Added 79-char line-length rule to development laws (AI
  assistant)
- 2025-08-11: Declared `setuptools_scm` as a runtime dependency (AI assistant)
- 2025-08-10: Updated README version and clarified `setuptools_scm`-based
  versioning (AI assistant)
- 2025-08-10: Gracefully handle missing `setuptools_scm` by importing it
  lazily
  (AI assistant)
- 2025-08-10: Removed tracked `copernican_suite.egg-info` and added to
  `.gitignore` (AI assistant)
- 2025-08-10: Derived fallback version from Git worktree using
  `setuptools_scm`
  (AI assistant)
- 2025-08-09: Formatted version and engine exports for style (AI assistant)
- 2025-08-09: Wrapped test file imports, docstrings and assertions for 79-char
  compliance (AI assistant)
- 2025-08-09: Shortened lines in `engines/cosmo_engine_comb.py` (AI assistant)
- 2025-08-09: Wrapped long lines across data parsers for 79-character
  compliance (AI assistant)
- 2025-08-09: Added `psutil` dependency and ensured CI installs project before
  running tests (AI assistant)

### Version Bump Rules
- **MAJOR**: incompatible API changes.
- **MINOR**: backward-compatible feature additions.
- **PATCH**: backward-compatible bug fixes and documentation updates.

## Version 3.6.7
- 2025-08-09: Refactored `model_coder` to replace lambda assignments,
  aligned Flake8 line length with Black and shortened long lines for
  lint compliance (AI assistant)
- 2025-08-09: Wrapped long lines in `copernican_lib/csv_writer.py`,
  `model_coder.py`, `model_parser.py`, `optim_utils.py` and `utils.py`
  for 79-column compliance (AI assistant)
- 2025-08-09: Wrapped `generate_filename` for 79-char limit (AI assistant)

## Version 3.6.6
- 2025-08-09: Wrapped long lines in `copernican_lib/optim_utils.py` for
  79-column compliance (AI assistant)

## Version 3.6.5
- 2025-08-09: Wrapped long line in `copernican_lib/model_parser.py` to
  enforce 79-character limit (AI assistant)

## Version 3.6.4
- 2025-08-09: Wrapped long lines in `copernican_lib/csv_writer.py` for
  79-column compliance (AI assistant)

## Version 3.6.3
- 2025-08-09: Wrapped long lines across `copernican_lib` modules and
  `copernican.py` for 79-column compliance (AI assistant)
- 2025-08-09: Lowered minimum Python version to 3.11, pinned `camb` to 1.6.2,
  updated CI and documentation (AI assistant)

## Version 3.6.2
- 2025-08-09: Configured pre-commit with Black, Isort, Ruff and Flake8 and
  added licensing reminders to contributor docs (AI assistant)

## Version 3.6.1
- 2025-08-09: Delegated the test-suite menu option to `python -m unittest
              discover`, expanded regression and interface tests, and
              updated CI to run the full suite on every push (AI assistant)

## Version 3.6.0
- 2025-08-09: Centralised version handling via `copernican_lib.version`,
  routed
  modules through the helper, configured `setuptools_scm` fallback and
  documented SemVer bump rules (AI assistant)

## Version 3.5.3
- 2025-08-09: Added PyInstaller build specifications for Windows, macOS and
  Linux, bundled project sources and documented macOS signing (AI assistant)

## Version 3.5.2
- 2025-08-09: Added cross-platform CI workflow using GitHub Actions (AI
  assistant)

## Version 3.5.1
- 2025-08-08: Expanded comments across codebase, restructured plot footer
  documentation and enlarged technical docs (AI assistant)

## Version 3.5.0
- 2025-08-07: Added comprehensive development plan summarizing project goals
  (AI assistant)
- 2025-08-05: Expanded subscript and superscript tables to cover full Latin
  and
  Greek alphabets, digits and common operators; updated docs and bumped
  version (AI assistant)

## Version 3.4.4
- 2025-08-04: Replaced unsupported ``\textbf`` footer styling with ``\mathbf``
  and preserved spaces to prevent plot save failures (AI assistant)

## Version 3.4.3
- 2025-08-04: Dropped HTML tags from plot footers, centralised footer
  generation and kept dataset names spaced; bumped version (AI assistant)

## Version 3.4.2
- 2025-08-04: Adopted HTML footer template preserving dataset spacing and
  bumped version (AI assistant)

## Version 3.4.1
- 2025-08-04: Added rule requiring concise, descriptive function and
  identifier
  names and synchronized documentation (AI assistant)

## Version 3.4.0
- 2025-08-04: Centralised dataset metadata loading in `data_loaders.py`,
  removed metadata handling from parsers and updated documentation (AI
  assistant)

## Version 3.3.8
- 2025-08-04: Replaced dataset name attributes with `dataset_name_sanitized`,
  preserved original `dataset_name`, and refreshed documentation (AI
  assistant)

## Version 3.3.7
- 2025-08-04: Updated metadata key references to use `author` and refreshed
  documentation; bumped version (AI assistant)

## Version 3.3.6
- 2025-08-04: Added BibTeX metadata fields and updated citations across public
  datasets; refreshed documentation and version numbers (AI assistant)

## Version 3.3.5
- 2025-08-03: Documented absence of joint covariance for BOSS DR12 data and
  parser's block-diagonal approach in docs and README (AI assistant)

## Version 3.3.4
- 2025-08-03: Added regression test for BOSS DR12 BAO parser validating
  covariance handling and error paths (AI assistant)

## Version 3.3.3
- 2025-08-03: Integrated full BOSS DR12 BAO covariance by combining dM/Hz and
  D_V/F_AP inputs; updated documentation and version (AI assistant)

## Version 3.3.2
- 2025-08-03: Corrected BOSS DR12 BAO conversion to include redshift scaling,
  fixed compound parser scaling bug and added escape-sequence guideline;
  bumped version (AI assistant)

## Version 3.3.1
- 2025-08-03: Renamed BAO test dataset to compound dataset, improved BAO
  parsers and documentation, and bumped version (AI assistant)

## Version 3.3.0
- 2025-07-31: Added BOSS DR12 BAO consensus dataset with full covariance and
  skipped placeholder folders; bumped version (AI assistant)

## Version 3.2.1
- 2025-07-31: Reordered Pantheon+ covariance matrix to match sorted data and
  updated documentation; bumped version (AI assistant)

## Version 3.2.0
- 2025-07-31: Standardized all console output through `console_output.py`,
  added automatic log renaming and bumped version (AI assistant)

## Version 3.1.1
- 2025-07-31: Updated JLA parser to use published SALT2 parameters and
  documented them; bumped project version (AI assistant)

## Version 3.1.0
- 2025-07-31: Reverted project to version 3.1.0 state and removed universal
  constants (AI assistant)
- 2025-07-30: Replaced `^` with `**` for exponentiation across all model YAML
  files and documented LaTeX syntax (AI assistant)

## Version 3.0.1
- 2025-07-31: Fixed CAMB parameter map exponent syntax in
  cosmo_model_usmf2.yml
  to prevent runtime errors (AI assistant)

## Version 3.0.0
- 2025-07-30: Dropped all remaining JSON dataset support and expanded
  documentation (AI assistant)

## Version 2.1.0
- 2025-07-30: Switched cached models and LaTeX mappings to YAML and removed
  JSON usage across the codebase (AI assistant)
- 2025-07-30: Converted all dataset metadata and the BAO compound dataset to
  YAML (AI assistant)

## Version 2.0.7
- 2025-07-30: Corrected malformed tab in USMFv2 description to pass YAML
  parsing (AI assistant)
- 2025-07-30: Expanded inline comments and documentation to clarify workflow
  logic (AI assistant)
- 2025-07-30: Synchronized development laws between README.md and AGENTS.md
  (AI
  assistant)
- 2025-07-30: Removed unused JLA covariance fallback logic (AI assistant)

## Version 2.0.6
- 2025-07-30: Expanded comments across the codebase and added a session-start
  reminder in AGENTS (AI assistant)
- 2025-07-30: Added RNG seeding, improved SNe chi-squared validation and
  expanded tests (AI assistant)
- 2025-07-30: Consolidated AI development guidelines into a single README
  section (AI assistant)

## Version 2.0.5
- 2025-07-30: Verified latex_mappings.json validity and kept fallback;
  reordered changelog and clarified instructions (AI assistant)

## Version 2.0.4
- 2025-07-30: Documented stub GW and siren parsers returning None (AI
  assistant)

## Version 2.0.3
- 2025-07-30: Removed Unicode escape sequences from model YAML files and
  converted abstracts and descriptions to block scalars (AI assistant)

## Version 2.0.2
- 2025-07-30: Console output now renders parameter names with Unicode Greek
  letters and subscripts (AI assistant)

## Version 2.0.1
- 2025-07-30: Vectorised BAO chi-squared and updated YAML documentation (AI
  assistant)

## Version 2.0.0
- 2025-07-30: Migrated all models to YAML and removed JSON support (AI
  assistant)

## Version 2.0.3
- 2025-07-30: Removed Unicode escape sequences from model YAML files and
  converted abstracts and descriptions to block scalars (AI assistant)


## Version 1.19.3
- 2025-07-29: Fixed parsing of LaTeX names containing `\rm` and bumped version
  (AI assistant)

## Version 1.19.2
- 2025-07-29: Normalized LaTeX parameter names in all models and updated
  example docs (AI assistant)

## Version 1.19.1
- 2025-07-29: Added missing LaTeX names to LCDM parameters and bumped version
  (AI assistant)

## Version 1.19.0
- 2025-07-29: Removed parameter-name fallback and made `latex_name` mandatory
  in all models (AI assistant)

## Version 1.18.3
- 2025-07-29: Fallback sound-horizon integral now looks for `Omega_b`,
  `Omega_gamma` and `z_rec`/`z_recomb` instead of legacy aliases (AI
  assistant)

## Version 1.18.2
- 2025-07-29: Fixed parsing failures by removing \rm from parameter names in
  expressions and bumped versions (AI assistant)

## Version 1.18.1
- 2025-07-29: Replaced legacy parameter aliases with full LaTeX names across
  models and documentation (AI assistant)

## Version 1.18.0
- 2025-07-28: Removed math delimiters and double backslash requirement in
  model
  files; added implicit multiplication (AI assistant)

## Version 1.17.0
- 2025-07-28: Extended latex_mappings with extra symbols, functions and
  macros;
  bumped version (AI assistant)

## Version 1.16.0
- 2025-07-28: Centralized LaTeX mappings and added latex_utils module (AI
  assistant)

## Version 1.15.0
- 2025-07-28: Added automatic python_var generation and improved LaTeX
  handling
  (AI assistant)

## Version 1.14.11
- 2025-07-28: Stripped size macros from plot labels and bumped version to
  1.14.11 (AI assistant)

## Version 1.14.10
- 2025-07-28: Expanded model JSON guide with supported functions and common
  mistakes (AI assistant)

## Version 1.14.9
- 2025-07-26: Reduced CMB title padding to avoid overlap with residual plots
  (AI assistant)
- 2025-07-27: Improved LaTeX parsing for additional macros (AI assistant)
- 2025-07-27: Fixed bracket handling in LaTeX parser to avoid parse failures
  (AI assistant)
- 2025-07-27: Documented JSON escape requirement for LaTeX macros (AI
  assistant)

## Version 1.14.8
- 2025-07-26: Improved footer spacing, unified CMB legends and added verbose
  dataset summaries (AI assistant)

## Version 1.14.7
- 2025-07-26: Combined JLA systematic and statistical covariances and updated
  parser logic (AI assistant)

## Version 1.14.6
- 2025-07-26: Unified info box spacing with margins, adjusted footer placement
  and fixed CMB title overlap (AI assistant)

## Version 1.14.5
- 2025-07-26: Documented JLA covariance fallback and tightened info box layout
  (AI assistant)

## Version 1.14.4
- 2025-07-27: Handled near-singular JLA covariance by falling back to diagonal
  errors (AI assistant)

## Version 1.14.3
- 2025-07-27: Removed deprecated UniStra SNe data and fixed JLA covariance
  handling (AI assistant)
- 2025-07-27: Improved fit report outputs and enlarged plot dimensions (AI
  assistant)

## Version 1.14.2
- 2025-07-26: Lightened grid lines, widened plot margins and fixed BAO info
  box
  equation parsing (AI assistant)

## Version 1.14.1
- 2025-07-26: Human intervention in CHANGELOG.md due to messed up order, dates
  and lack of template (Apostol Apostolov)
- 2025-07-26: Unified plot style and improved info boxes across all data types
  (AI assistant)

## Version 1.14.0
- 2025-07-25: Added JLA 2014 dataset with full covariance matrix and new
  metadata field `authors_all` (AI assistant)
- 2025-07-25: Fixed version string handling and updated documentation (AI
  assistant)

## Version 1.13.1
- 2025-07-25: Renamed test BAO dataset and updated documentation (AI
  assistant)

## Version 1.13.0
- 2025-07-24: Enforced automatic SemVer bumps and updated version references
  (AI assistant)

## Version 1.12.9
- 2025-07-19: Expanded and clarified documentation; explained `.egg-info`
  folder and added CONTRIBUTING guide (AI assistant)

## Version 1.12.8
- 2025-07-19: Updated logger to avoid duplicate console output and capture
  user
  input (AI assistant)
- 2025-07-19: Footer lines now rendered with smaller font to prevent overlap
  (AI assistant)

## Version 1.12.7
- 2025-07-16: Log now records console output verbatim and strips absolute
  paths
  (AI assistant)

## Version 1.12.6
- 2025-07-16: Improved footer wrapping, plot legends and info boxes with
  combined chi2; tweaked BAO residuals (AI assistant)

## Version 1.12.5
- 2025-07-16: Ignored virtual env directories when scanning imports for
  dependency check (AI assistant)
- 2025-07-16: Removed automatic dependency installation and virtual
  environment
  logic (AI assistant)
- 2025-07-16: Implemented BAO residual plots with smoothed averages (AI
  assistant)
- 2025-07-16: Added smoothed residual averages to all plots and extended
  footer
  wrapping (AI assistant)
- 2025-07-16: Dependency check now prints install command with only missing
  packages (AI assistant)
- 2025-07-16: Dependency checker parses imports via AST and prints OS-aware
  install instructions (AI assistant)
- 2025-07-16: Fixed logger crash and missing AST import in dependency check
  (AI
  assistant)

## Version 1.12.4
- 2025-07-15: Fixed CMB spectrum scaling bug and added Dl verification test
  (AI
  assistant)
- 2025-07-15: Updated documentation and developer guide with raw string rule
  (AI assistant)
- 2025-07-15: Converted math docstrings to raw strings to silence escape
  warnings (AI assistant)
- 2025-07-15: Fixed dependency check for Python 3.13 `find_spec` ValueError
  (AI
  assistant)

## Version 1.12.3
- 2025-07-13: Unified timestamp handling and console output format updated (AI
  assistant)

## Version 1.12.2
- 2025-07-10: Unified dataset metadata files and expanded plot footers (AI
  assistant)
- 2025-07-10: Fixed file name sanitization for Planck dataset (AI assistant)

## Version 1.12.1
- 2025-07-10: Dynamic BAO metadata parsing and verbose fit summaries (AI
  assistant)

## Version 1.11.9
- 2025-07-10: Automatic virtual environment setup and start scripts for
  Windows, macOS and Linux. Cancelling a run now removes its log file (AI
  assistant)

## Version 1.11.8
- 2025-07-09: Added official JLA and Pantheon+ dataset names and short
  identifiers (AI assistant)
- 2025-07-09: Simplified plot footers and updated documentation (AI assistant)

## Version 1.11.7
- 2025-07-09: Renamed Pantheon+ files and made parser auto-detect dataset
  names
  (AI assistant)
- 2025-07-09: Moved chi-squared helpers back into the engine and removed
  chi2_helper module (AI assistant)

## Version 1.11.6
- 2025-07-09: Removed deprecated 1.4b and numba engines and set combined
  engine
  as default (AI assistant)

## Version 1.11.5
- 2025-07-09: Documented SNe refinement step in workflow section of README (AI
  assistant)
- 2025-07-08: Added SNe pre-fit step to combined engine to improve convergence
  and updated documentation (AI assistant)
- 2025-07-08: Updated minimum Python version to 3.12 and synced README (AI
  assistant)
- 2025-07-08: Added runtime check for Python version and documented exit
  behavior (AI assistant)

## Version 1.11.4
- 2025-07-08: Expressions in all cosmo_model JSON files converted to LaTeX and
  parser updated (AI assistant)

## Version 1.11.3
- 2025-07-07: Fixed missing extra CMB parameters in run_cmb_analysis and
  bumped
  version (AI assistant)

## Version 1.11.2
- 2025-07-07: Moved chi-squared helpers to chi2_helper module and updated docs
  (AI assistant)

## Version 1.11.1
- 2025-07-07: Unified SNe data processing and chi-squared helpers (AI
  assistant)


## Version 1.10.1-beta (Development Release)
- 2025-07-07: Unified CMB handling with SNe and BAO, removed engine interface
  fallbacks, updated docs (AI assistant)

## Version 1.9.3-beta (Development Release)
- 2025-07-07: Fixed parameter list mutation in combined engine and bumped
  version (AI assistant)
- 2025-07-07: Removed deprecated L-BFGS-B solver options to silence SciPy
  warnings (AI assistant)
- 2025-07-07: Increased CMB cache precision to six significant digits (AI
  assistant)

## Version 1.9.2-beta (Development Release)
- 2025-07-07: Bumped version to 1.9.2-beta and expanded code comments (AI
  assistant)

## Version 1.9.1-beta (Development Release)
- 2025-07-07: Renamed scripts package to copernican_lib and updated
  documentation (AI assistant)

## Version 1.9.0-beta (Development Release)
- 2025-07-07: Centralized optimization wrappers and updated documentation (AI
  assistant)

## Version 1.8.5-beta (Development Release)
- 2025-07-07: Enforced spawn start method and restricted JSON validation to
  main process (AI assistant)

## Version 1.8.4-beta (Development Release)
- 2025-07-07: Restored compatibility of chi_squared_cmb with plugin interface
  (AI assistant)
- 2025-07-07: Bumped development version and updated documentation (AI
  assistant)
- 2025-07-07: Documented engine-plugin architecture and updated JSON example
  (AI assistant)
- 2025-07-07: Revised AGENTS overview and expanded README with developer guide
  (AI assistant)
- 2025-07-07: Fixed test discovery and Matplotlib cleanup when running the
              test suite via the menu option (AI assistant)
  (AI
  assistant)

## Version 1.8.3-beta (Development Release)
- 2025-07-06: Rewrote combined engine for true joint optimisation (AI
  assistant)
- 2025-07-06: Fixed CMB chi-squared interface and allowed fitting of CAMB
  parameters (AI assistant)

## Version 1.8.2-beta (Development Release)
- 2025-07-06: Optimized CMB evaluation with cached CAMB calls (AI assistant)
- 2025-07-06: Enabled true joint fitting with optional SALT2 parameters (AI
  assistant)

## Version 1.8.1-beta (Development Release)
- 2025-07-06: Made combined-fit engine verbose and fixed docstring escape
  warning (AI assistant)

## Version 1.8.0-beta (Development Release)
- 2025-07-06: Added combined-fit engine and optional test execution (AI
  assistant)
- 2025-07-06: Bumped version to 1.8.0-beta (AI assistant)
- 2025-07-06: Integrated combined-fit workflow and updated documentation (AI
  assistant)

## Version 1.7.12-beta (Development Release)
- 2025-07-06: Added TE/EE spectrum handling and improved cosmic variance
  plotting (AI assistant)
- 2025-07-06: Bumped version to 1.7.12-beta (AI assistant)

## Version 1.7.11-beta (Development Release)
- 2025-07-06: Fixed Planck 2018 lite parser and trimmed covariance to TT block
  (AI assistant)
- 2025-07-06: Bumped version to 1.7.11-beta (AI assistant)

## Version 1.7.10-beta (Development Release)
- 2025-07-06: Corrected CAMB spectrum scaling and updated docs (AI assistant)
- 2025-07-06: Bumped version to 1.7.10-beta (AI assistant)

## Version 1.7.9-beta (Development Release)
- 2025-07-06: Fixed Planck lite scaling and covariance endianness (AI
  assistant)
- 2025-07-06: Enhanced default CMB wrapper and engine spectra output (AI
  assistant)
- 2025-07-06: Updated documentation and version bump to 1.7.9-beta (AI
  assistant)

## Version 1.7.8-beta (Development Release)
- 2025-07-06: Added dedicated CMB analysis stage with verbose logging (AI
  assistant)
- 2025-07-06: Updated documentation and version bump to 1.7.8-beta (AI
  assistant)

## Version 1.7.7-beta (Development Release)
- 2025-07-06: Overhauled Planck parser with µK² conversion and TE/EE support
  (AI assistant)
- 2025-07-06: Redesigned CMB plot with log scaling and variance shading (AI
  assistant)
- 2025-07-06: Documentation updates and version bump to 1.7.7-beta (AI
  assistant)

## Version 1.7.6-beta (Development Release)
- 2025-07-05: Bumped COPERNICAN_VERSION and docs to 1.7.6-beta. (AI assistant)
- 2025-07-06: Added TE/EE spectrum handling in parser, engine and plotter. (AI
  assistant)
- 2025-07-06: Improved Planck lite parser covariance checks with fallback
  warnings. (AI assistant)
- 2025-07-06: Fixed chi-squared label formatting warnings in plotter. (AI
  assistant)

## Version 1.7.5-beta (Development Release)
- 2025-07-05: Removal of user-selectable test mode. (AI assistant)
- 2025-07-05: Automatic functional tests run at startup. (AI assistant)
- 2025-07-05: Updated documentation and model guide. (AI assistant)
- 2025-07-05: Clarified CMB requirements in cosmo_model_guide and bumped guide
  version. (AI assistant)
- 2025-07-05: Documented automatic startup test suite in README. (AI
  assistant)

## Version 1.7.4-beta (Development Release)
- 2025-07-05: Fixed unit conversion (K\u00b2 \u2192 \u03bcK\u00b2) by applying
  a 1e12 scale factor (AI assistant)
- 2025-07-05: Added neutrino density mapping (`omnuh2`) to the \u039bCDM
  parameter map (AI assistant)

## Version 1.7.3-beta (Development Release)
- 2025-07-05: Fixed Planck covariance reader for ASCII data and ensured CMB
  parameters use SNe best-fit values (AI assistant)
- 2025-07-05: Corrected Planck covariance parsing for binary Fortran record
  (AI
  assistant)
- 2025-07-05: Re-added integral expression support using numerical quadrature
  (AI assistant)
- 2025-07-05: Added `_wrap_math` helper and updated parameter label rendering
  (AI assistant)
- 2025-07-05: Updated LICENSE.md with new definitions and effective date (AI
  assistant)
- 2025-07-05: Restored 1.6.4 and 1.6.5 changelog entries (AI assistant)

## Version 1.7.2-beta (Development Release)
- 2025-07-05: Fixed Planck covariance parser using np.loadtxt (AI assistant)
- 2025-07-05: Added default CAMB parameter mapping from SNe fits (AI
  assistant)
- 2025-07-05: Handled binary Planck covariance matrix fallback (AI assistant)

## Version 1.7.1-beta (Development Release)
- 2025-07-05: Updated version references to 1.7.1-beta (AI assistant)
- 2025-07-05: Implemented Planck 2018 lite CMB parser (AI assistant)
- 2025-07-05: Added `valid_for_cmb` flag and updated plugin validation (AI
  assistant)
- 2025-07-05: Added CAMB-based CMB analysis and chi-squared routines (AI
  assistant)
- 2025-07-05: Added cmb.param_map metadata to models and documentation (AI
  assistant)
- 2025-07-05: Stored CAMB parameter order in Planck 2018 parser (AI assistant)
- 2025-07-05: Added automatic CMB wrapper and parameter mapping helper (AI
  assistant)
- 2025-07-05: run_cmb_analysis now converts fitted parameters with
  get_camb_params (AI assistant)

## Version 1.7.0-beta (Development Release)
- 2025-07-05: Skip CMB evaluation when model sets valid_for_cmb=false (AI
  assistant)
- 2025-07-05: Implemented CMB spectrum plotting (AI assistant)
- 2025-07-05: Added CMB residual CSV export (AI assistant)
- 2025-07-05: Documented cmb.param_map usage and parser param_names attribute
  (AI assistant)
- 2025-07-05: Bumped version to 1.7.0 and reorganized changelog (AI assistant)
- 2025-07-05: Removed obsolete CMB placeholder parser and dataset (AI
  assistant)
- 2025-07-05: Added CAMB dependency to pyproject and updated docs (AI
  assistant)
- 2025-07-05: Corrected CMB spectrum units and Planck parser to use D_l (AI
  assistant)
- 2025-07-05: Removed DEV NOTE headers from pyproject.toml (AI assistant)

## Version 1.6.5 (Patch Release)
- 2025-06-23: Fixed plot info boxes to display equations from the selected
  alternative theory and ensured Greek letters render correctly (AI
  assistant)
- 2025-06-23: Updated README and AGENTS documentation for corrected JSON
  schema
  and version bump (AI assistant)

## Version 1.6.4 (Patch Release)
- 2025-06-23: Added numerical quadrature support for Integral expressions (AI
  assistant)

## Version 1.6.3 (Patch Release)
- 2025-06-22: Restored `pyproject.toml` and silenced Pandas whitespace warning
  (AI assistant)
- 2025-06-22: Declared Python 3.13.1+ requirement in pyproject and README (AI
  assistant)

## Version 1.6.2 (Patch Release)
- 2025-06-22: Added LCDM equations and sound horizon formula (AI assistant)

## Version 1.6.1 (Patch Release)
- Restored model equations in plot info boxes.
- 2025-06-22: Fixed plot crashes when model equations used display-mode dollar
  signs (AI assistant)
- Added standardized plot footer with run metadata.
- start.command cleaned up.
- 2025-06-21: Documented stable plotting style and algorithms (AI assistant)
- 2025-06-21: Clarified when MINOR vs PATCH increments occur in README (AI
  assistant)

## Version 1.6 (Stable Release)
- 2025-06-21: Fixed trailing text in start.command and ensured newline (AI
  assistant)
- 2025-06-21: First stable release with reliable SNe Ia and BAO calculations
  (AI assistant)
- 2025-06-21: Legacy DEV NOTE headers removed from source files and notes
  migrated to `CHANGELOG.md` (AI assistant)
- 2025-06-21: Plugin now exposes model equations and filename (AI assistant)
- 2025-06-21: Plugin filename stored during JSON loading (AI assistant)
- 2025-06-21: Plots now include a timestamped footer with comparison details
  (AI assistant)

## Version 1.5.1 (Development Release)
- 2025-06-20: Added CHANGELOG template and updated docs to reference it (AI
  assistant)
- Removed ``initial_guess`` from JSON models; parameter guesses now computed
  automatically from bounds.
- Consolidated model metadata: ``theory`` block removed and equations moved
  under ``equations``.
- Documentation updated to reflect declarative model design.
- Development protocol revised: DEV NOTE markers removed in favor of
  documenting changes in `CHANGELOG.md` or `AGENTS.md`.
- Schema documentation updated: `abstract` and `description` are now mandatory
  and all contributors summarize updates in `CHANGELOG.md`.
- 2025-06-20: Added explicit `rs_expression` to `cosmo_model_lcdm.json` and
  migrated legacy documentation notes to `CHANGELOG.md` (AI assistant)

## Version 1.5.0 (Development Release)
- Data files and parsers reorganized under ``data/<type>/<source>/``.
- Parser selection now based on data source only.
- Removed deprecated `parsers/` directory and UniStra h2 parser.
- Updated documentation for version 1.5.0.
- Hotfix: Prompts list friendly dataset names with a clear title for every
  selection.

## Version 1.5f (Development Release)
- Completed Phase 6: JSON schema extended with optional fields for CMB and
  gravitational-wave standard siren inputs. Added placeholder parser coverage
  and loader functions for these data types.
- Updated documentation for version 1.5f.
- Hotfix 5: Removed automatic dependency installer. Users are now instructed
  to
  run a printed `pip install` command when packages are missing.
- Hotfix 7: `Hz_expression` added to JSON models and compiled automatically
  for
  distance predictions.
- Hotfix 8: Sound horizon `r_s` is now computed automatically when possible
  using
  a fallback integral if `rs_expression` is missing.
- Hotfix 9: Parser auto-discovery now searches the project's top-level
  `parsers`
  directory instead of a nonexistent `scripts/parsers` folder.
- Hotfix 10: Fixed BAO smooth curve generation by allowing `_dm` to accept
  array
  redshift values.

## Version 1.5e (Development Release)
- Added Numba-based engine and modular utility wrappers.
- Updated documentation for version 1.5e.

## Version 1.5d (Development Release)
- Completed Phase 4: all models converted to JSON and legacy plugins removed.
- Updated documentation and headers for version 1.5d.
- Automatic dependency installer added and invoked by `copernican.py` when
  packages are missing.

## Version 1.5c (Development Release)
- Completed Phase 3: engine_interface now validates plugins and engines use
  the
  new abstraction layer.
- Updated documentation and headers for version 1.5c.

## Version 1.5b (Development Release)
- Completed Phase 2: parser caches validated JSON and coder generates
  callables
  with sanity checks.
- Updated documentation and headers for version 1.5b.

## Version 1.5a (Development Release)
- Introduced JSON-based model pipeline and new `scripts/` modules.
- Added example JSON model and updated documentation for version 1.5a.

## Version 1.4.1 (Maintenance Release)
- LCDM model separated into lcdm.py plugin.
- Added splash screen and improved logging with per-run timestamps.


## Version 1.4 (Stable Release)
- Refactored into a fully pluggable architecture with discoverable engines,
  parsers and models.
- Migrated specification into `AGENTS.md` and cleaned documentation.
- Added modular data and model directories.
- Finalized engine and model interfaces for long-term stability.

## Version 1.3 (Stable Release)
- CRITICAL BUG FIX - BAO plotting restored (fixed multiprocessing issue).
- Added developer specification `doc.json`.
- BAO plot clarity improved with transparency.
- Streamlined CSV outputs to detailed files only.

## Version 1.2 (Major Refactor)
- Removed GPU code for stability.
- Implemented robust multiprocessing using `psutil`.
- Added test mode and cache cleanup loop.
