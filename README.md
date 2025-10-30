**Version:** 6.0.14
**Last Updated:** 2025-10-30

![Copernican Suite banner](docs/banner_github.png)

The Copernican Suite is a Python toolkit for testing cosmological models
against Supernovae Type Ia (SNe Ia), Baryon Acoustic Oscillation (BAO), and
Cosmic Microwave Background (CMB) data.
Support for gravitational waves and standard siren events is planned for
future
releases.
The suite provides a modular architecture so new models, data parsers and
computational engines can be plugged in with minimal effort.
Additional design notes can be found under the `docs/` directory.
Citation details are provided in [CITATION.cff](CITATION.cff).

Plot summaries now report ``N/A`` for missing chi-squared totals so the
visualisation workflow remains stable even when a dataset omits cross-dataset
statistics. Supernova-only MCMC runs now populate ``χ²_Total`` with the SNe
value so downstream summaries never display ``N/A`` when only the sampling
engine is in play.

---

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Directory Layout](#directory-layout)
4. [Design Overview](docs/design_overview.md)
5. [Data Directory Overview](docs/data_overview.md)
6. [BAO Compound Dataset Format](docs/bao_compound_dataset_format.md)
7. [Dataset Metadata Fields](docs/dataset_metadata.md)
8. [LaTeX Syntax Guide](docs/latex_syntax.md)
9. [Using the Suite](#using-the-suite)
10. [Plot Footers and Metadata](#plot-footers-and-metadata)
11. [Logging and Caching](#logging-and-caching)
12. [Creating New Models](#creating-new-models)
13. [Developer Guide](#developer-guide)
    - [Workflow Overview](#workflow-overview)
    - [Development History & Roadmap](#development-history--roadmap)
    - [AI-driven and human development laws and
      protocols](#ai-driven-and-human-development-laws-and-protocols)
14. [License](#license)
15. [Versioning Policy](#versioning-policy)
16. [API Overview](docs/api_overview.md)
17. [Packaging Guide](docs/packaging.md)
18. [Documentation Policy](docs/documentation_policy.md)
19. [Run Manifest](docs/run_manifest.md)

---

## Overview
The suite compares the reference ΛCDM model with alternative theories
provided by the user. Each model is defined entirely by a YAML file
`cosmo_model_*.yml` under `./models/`. This YAML stores all theory text,
equations and parameters and serves as the sole source of truth. Optional
Markdown summaries may exist for human readers but are ignored by the
software.
Users select models, datasets, and computational engines at runtime through a
simple command line interface. Each execution creates a dedicated
`output/copernican-run_YYYYMMDD_HHMMSS` folder that stores plots, CSV tables
and posterior chains in NetCDF format.
Under the hood the program follows a clear pipeline:
1. **Dependency Check** – `copernican.py` scans for required packages,
   installs missing ones and verifies each import. The scan now caches its
   import list in `.cache/dependency_scan.json` whenever the source tree is
   unchanged, so repeated launches skip the expensive AST parse and return to
   the menu promptly. A tiny NumPy/SciPy calculation then runs to catch CPU
   feature mismatches before heavy computation begins.
2. **Initialization** – a run-specific output directory is created and
   logging begins.
3. **Configuration** – the user chooses a model and a computation engine
   from `./engines/`. Engines are discovered dynamically by the
   `cosmo_engine_*.py` naming convention so additional deterministic or
   stochastic solvers can be dropped in later without touching the launcher.
   The current default, `cosmo_engine_mcmc.py`, uses an `emcee` ensemble
   sampler to explore the SNe posterior. Its distance calculations are
   vectorised for responsiveness and invalid proposals return ``-np.inf``
   directly so walkers outside the allowed region or producing non-finite
   chi-squared values are rejected unambiguously. The sampler automatically
   classifies parameters whose bounds collapse to a point—or a numerically
   indistinguishable sliver—as fixed so arbitrary model plugins remain
   compatible. Initial walkers now expand adaptively until their condition
   number falls below the guardrail enforced by ``emcee``, ensuring even wildly
   scaled models and exotic bound combinations initialise without manual
   tuning. After burn-in any walkers
   that drift into ``nan`` coordinates are reseeded near the ensemble mean,
   removing the `RuntimeWarning: invalid value encountered in scalar subtract`
   messages recorded in previous ΛCDM self-tests. The sampler draws its
   initial ensemble uniformly inside the declared bounds, performs an
   explicit burn-in phase before production sampling, records acceptance
   fractions, autocorrelation estimates and log-probability traces and emits
   progress updates with percentage indicators for both burn-in and
   production stages. Shared chi-squared helpers live in
   `copernican_lib/statistics.py` so every backend calls the same routines
   without cross-importing engine modules. Constant values in a model's
   `cmb.param_map` are treated as additional fit parameters so CMB spectra
   can be matched precisely. Data
   parsers are discovered automatically under
  `data/<type>/<source>` and models are loaded from `cosmo_model_*.yml`.
  Only parser modules whose SHA256 digest matches a vetted list are imported,
  ensuring untrusted files are ignored. Symbolic links are rejected and any
  path that resolves outside the repository is skipped. Folders named
  `placeholder` are ignored so unfinished datasets do not appear in the
  selection menus.
4. **Parameter Fitting** – the MCMC engine samples the SNe posterior for the
   ΛCDM baseline and the chosen alternative model. When both theories share
   the same plugin (for example when testing ΛCDM against itself) the Stage 2
   workflow compares `MODEL_FILENAME` values, reuses the first chain and
   copies the chi-squared totals so BAO and CMB comparisons draw identical
   predictions. Otherwise the ΛCDM reference and the alternative model are
   sampled in turn with independent random seeds.
5. **BAO Analysis** – BAO observables are computed using the MAP parameters
   returned by the sampler and chi-squared statistics are reported. Shared
   helpers from `copernican_lib.statistics` keep the calculations identical
   regardless of the active engine, ensuring future backends remain
   drop-in replacements. When a sampler reports failure or omits fitted
   parameters the suite now skips BAO calculations for that model instead of
   crashing, logging a warning so users can revisit their priors.
6. **CMB Analysis** – CMB power spectra are generated using the fitted
   cosmological parameters and any extra CMB-specific values supplied by the
   engine. The chi-squared contribution is then calculated. As with the BAO
   stage the orchestrator bypasses CMB processing gracefully when the
   underlying fit does not yield cosmological parameters, preventing
   `KeyError` exceptions at the end of long runs.
7. **Spectra Caching** – unlensed CAMB spectra are cached using parameter
   keys rounded to six significant digits.
8. **Output Generation** – `copernican_lib/logger.py`,
   `copernican_lib/plotter.py` and `copernican_lib/csv_writer.py` handle
   logs, plots and tables. The log file is renamed at the end of each run to
   match the output timestamp.
9. **Loop or Exit** – the user may evaluate another model or quit, at which
   point temporary cache files are cleaned automatically.

## Quick Start
1. Run the platform-specific `start` script. macOS users should run
   `./start.command`, Windows users open `start.bat`, and Linux users can
   execute `./start.sh`. The launcher downloads a private Python 3.12+ into
   `.python`, removes any bundled interpreter older than 3.12 and recreates
   `.venv` automatically when its Python falls below the minimum supported
   version. It pins `pip` to 24.2 before installing dependencies so
   Windows jobs no longer attempt to grab unreleased wheels, installs the
   locked stack and installs the project with `pip install --no-deps .`. The
   helpers
   skip errors when
   `VIRTUAL_ENV` is unset and delete any `build/` directory before and after
   installation to avoid stale artifacts. If the activation script is missing
   the launcher recreates `.venv` once before exiting with an error. Each
   launcher prints a notice before invoking `sudo`, `brew` or `winget` so
   users know any password prompt originates from the package manager and is
   never read or stored. `sudo -k` and explicit prompts keep password
   handling within the operating system. On Windows the launcher now
   delegates the download and extraction steps to dedicated helper routines
   so the PowerShell commands execute outside the bootstrap condition,
   preventing `cmd.exe` from mis-parsing closing parentheses and restoring
   the interactive menu.
2. Follow the interactive prompts to choose a model, preferred data sources
   and
   computation engine.
3. Choose "Run the unit test suite" from the launcher's menu or execute
   `python -m unittest discover -v` directly. The test runner reports
   informational messages, warnings and errors while verifying the
   reference model and parsers. Toggle strict warning mode from the menu
   or set `COPERNICAN_STRICT_WARNINGS=1` to upgrade warnings to errors for
   reproducible CI runs.
4. When prompted, choose an RNG seed or set `COPERNICAN_SEED=<n>` in the
   environment to skip the prompt. The seed defaults to `0` and is applied to
   NumPy, Python's ``random`` module and supported engines.
5. Results, including posterior chains, will appear inside a timestamped
   folder under `output/` when the run completes.

## Dependencies
The launchers automatically bootstrap a dedicated Python 3.12+ into
`.python`, delete any interpreter older than 3.12 and rebuild `.venv`
whenever its Python falls below the supported floor, so no pre-existing
Python installation is needed. They verify that `.venv/bin/activate` exists
and retry once before aborting. Inside the virtual environment this project
relies on `numpy==1.26.4`, `scipy==1.12.0`, `matplotlib==3.8.2`,
`pandas==2.2.1`, `sympy==1.13.0`, `jsonschema==4.21.1`,
`camb==1.6.3`, `emcee==3.1.4`, `h5netcdf==1.3.0`,
`h5py==3.10.0`, `xarray==2023.12.0`, `typing_extensions==4.10.0`
and the widely available `arviz==0.16.1` release
so wheels exist on every platform. The launchers refuse to run when another
virtual environment is active and reinstall pinned dependencies on every
start so the suite always uses its managed `.venv`.

Versions for all runtime dependencies are pinned in
`requirements.lock`. The manifest lists the same wheel-friendly releases as
`pyproject.toml`, and the CI bootstrap upgrades `pip` to 24.2 before it
resolves the lock so Windows installers never attempt to overwrite the
running binary. Helper libraries such as `xarray-einstats==0.6.0`,
`typing_extensions==4.10.0`, Matplotlib's rendering stack
(`contourpy==1.2.0`, `cycler==0.12.1`, `fonttools==4.51.0`,
`kiwisolver==1.4.5`, `pillow==10.3.0`, `pyparsing==3.1.1`), the timezone
tooling (`python-dateutil==2.9.0.post0`, `six==1.16.0`, `pytz==2024.1`,
`tzdata==2024.1`) and supporting libraries such as
`packaging==24.2`, `attrs==23.2.0`, `jsonschema-specifications==2023.12.1`,
`referencing==0.34.0`, `rpds-py==0.18.0`, `pyerfa==2.0.1.1` and
`astropy-iers-data==0.2024.10.28.0.34.7` remain pinned to the published
wheels. When a package is missing the program asks before running
`pip install -r requirements.lock` and verifies each import. Set
`COPERNICAN_AUTO_INSTALL=1` to skip the prompt in automated environments.
Regenerate both files together whenever dependencies change so the suite and
published wheels remain in sync. The pre-commit hook provisions
`pip-tools==7.4.1` on demand before it runs `make lock`, so the runtime
environment no longer carries `pip-tools` or `pip` in the lock file.
Use the bundled start scripts to enter the managed environment before
regenerating locks; they guarantee the module resolves to the pinned
version of `pip-tools`. The `make lock` target wraps
`python -m piptools compile --allow-unsafe`, so law 22 under
"AI-driven and human development" covers this workflow explicitly.
To keep CI reproducible across Python 3.11 and 3.12, the target
normalises the generated header so reruns stop rewriting the
interpreter banner.
Running `copernican.py` directly now fails with a message directing you to
use the `start.*` helpers. Future engines may also depend on `numba` or GPU
libraries.

## Building & Installation
Windows users should open `start.bat`, macOS users should run
`./start.command`, and Linux users can execute `./start.sh`. These helpers
create a local virtual environment, upgrade `pip` and install the package
automatically before launching the suite. Running `copernican.py` outside
this environment prompts you to use the appropriate start script.

The launchers now assemble the Python download URL once, validate that it
is non-empty and halt with an explicit "Copernican Suite download URL is
empty." error when the release metadata is missing. The Windows helper
passes the URL and archive path to PowerShell as strict parameters so
`Invoke-WebRequest` cannot start with undefined values, while the Unix
launchers rely on `curl -fL` to surface HTTP failures immediately.

To install the package system-wide run:

```bash
pip install .    # regular install
pip install -e . # editable for development
```

Installing the suite with `pip` creates a `copernican_suite.egg-info`
directory. The launchers also delete any temporary `build/` directory before
and after installation so build artifacts are never tracked. This folder
contains package metadata such as the version number, dependency list and
entry points used by Python's packaging tools. It is generated automatically
and does not need to be tracked in version control.

The suite no longer ships standalone binaries. Launch with `start.bat`,
`start.command` or `start.sh` to create a local `.venv` and install all
dependencies automatically. Only a system-wide Python 3.12+ installation is
required. See [docs/packaging.md](docs/packaging.md) for launcher details.

## Continuous Integration
The GitHub Actions workflow named **CI** validates every pull request and each
push to the `main` branch across `ubuntu-latest`, `macos-latest` and
`windows-latest` runners using Python 3.12. The job checks out the repository,
restores cached pip wheels through `actions/setup-python`, optionally reuses
CAMB background data from `~/.camb`, installs the pinned dependencies from
`requirements.lock`, executes `pytest -q` and then builds both the source
distribution and wheel via `python -m build`. The resulting `dist/` directory
is uploaded as a workflow artifact with `actions/upload-artifact` so
maintainers can inspect the exact packages produced by CI. Branch protection
requires the CI job to succeed before merges complete, so contributors should
replicate this sequence locally to avoid surprises.

Windows bootstrap reliability received an extra safeguard in 4.3.21.
`start.bat` now computes release metadata such as the Python version,
architecture tag and release identifier before the interpreter bootstrap
conditional executes. This keeps `%DOWNLOAD_URL%` stable on stock
`cmd.exe` builds without requiring delayed expansion. The empty-URL guard
remains in place, and the macOS and Linux launchers continue to function
unchanged.


## Directory Layout
```
models/           - YAML model definitions containing all theory text and
                    equations. Optional `.md` files may provide human-readable
                    summaries but are not required.
engines/          - Computational backends (e.g. `cosmo_engine_mcmc.py` for
                    ensemble sampling)
data/             - Observation data organized as ``data/<type>/<source>/``
  cmb/planck2018lite/ - Planck 2018 lite TT/TE/EE spectra and covariance
                         (binary Fortran matrix)
output/           - All generated results
AGENTS.md         - Development specification and contributor rules
CONTRIBUTING.md   - Quick checklist for pull requests
CHANGELOG.md      - Release history
copernican_lib/          - Helper modules
  logger.py         - Logging setup and helpers
  console_output.py - Console output helpers
  plotter.py        - Plotting functions
  csv_writer.py     - CSV output helpers
  data_loaders.py   - Data loading utilities
  utils.py          - Common helpers
  optim_utils.py    - Shared optimisation wrappers used by engines
```
All dataset tables and metadata are provided **only** as YAML files. JSON
input is no longer supported as of version 3.0.0.
**Note:** Tables in `data/` are read-only reference datasets.  Parser `.py`
files and accompanying `metadata_*.yml` files within that tree may be
modified when necessary.

## Engine and Plugin Architecture
The program compiles model equations into Python functions at runtime. When a
`cosmo_model_*.yml` file is selected, `copernican_lib/model_parser.py`
validates the
content and `copernican_lib/model_coder.py` converts the symbolic expressions
into
NumPy-ready callables. `copernican_lib/engine_interface.build_plugin` attaches
these
functions to a lightweight plugin object that exposes a stable API. Every
engine
operates solely through this plugin and decides how parameters are fitted. The
main workflow simply loads the plugin, selects an engine from `./engines/` and
invokes its functions. New engines can therefore implement alternate
strategies
—such as SNe-only sampling or future joint optimisations—without modifying the
rest of the codebase.
Generic chi-squared helpers live in `copernican_lib/statistics.py` and are
re-exported by each engine module, keeping `model_coder.py` focused on
translating models.

The helper `chi_squared_cmb` now accepts either a plugin and parameter
vector or a ready CAMB dictionary. This flexibility lets future engines reuse
the same CMB calculation regardless of their own fitting scheme.

## Using the Suite
- The program discovers available models from `models/cosmo_model_*.yml`.
 - Data sources for SNe, BAO and CMB are chosen interactively. Once a source
   is
   selected, its parser and files are loaded automatically from
   `data/<type>/<source>/`. The CMB loader now understands TT, TE and EE
   spectra with full covariance so additional datasets can be dropped in with
  minimal effort. The BOSS DR12 BAO parser combines the consensus dM, Hz,
  $D_V$ and $F_{AP}$ measurements with their covariance matrices to yield
  $D_M/r_s$, $D_H/r_s$ and $D_V/r_s$. The public [SDSS DR12
archive](https://data.sdss.org/sas/dr12/boss/) does not provide a
  joint covariance matrix for these observables, so `cosmo_parser_bossdr12.py`
  follows a block-diagonal approach that assumes the $dM/Hz$ and $D_V/F_{AP}$
  sets are uncorrelated.
- Engines are selected interactively from the `engines/` directory. Parsers
  are
  discovered automatically when their source folders are imported.
- Model plugins must be validated once using
  `engine_interface.validate_plugin` before passing them to engine
  routines or chi-squared helpers.
- After each run you may choose to evaluate another model or exit. Cache files
  are cleaned automatically.
- When a run finishes the suite prints the abstract text from each model along
  with a summary of the best-fit parameters and individual chi-squared values
for
  SNe, BAO and CMB.

## Plot Footers and Metadata
Each generated plot includes a centered footer that documents the run.
The first line shows the model comparison, Copernican Suite version and a
timestamp. The second line lists the observational dataset and processing
notes, and the third line provides the citation. The first and third lines are
bold, while the dataset name on the second line retains its original spacing
via MathText's ``\mathbf`` command.

Metadata values are read from ``metadata_*.yml`` files stored next to each
dataset. These files include a ``license`` field pointing to usage terms.
``copernican_lib/data_loaders.py`` attaches this metadata to the DataFrame
returned by each parser so both plot footers and CSV headers reflect the
official dataset description and citation. Individual parsers never access
metadata files directly.

During configuration each loader prints a summary indicating whether the
dataset's covariance matrix was inverted successfully or if diagonal errors
are
being used. When generating file names the suite sanitizes dataset names,
replacing spaces and characters like ``/`` with hyphens so output paths remain
portable across operating systems.

## Logging and Caching
All console output and user prompts are captured in a timestamped log file in
`./output/`. After initialisation the suite logs the Python version, OS, CPU
model and key package versions. A short summary appears on the console while
full details are stored in the log file. The logger shortens absolute paths so
logs remain portable and records the final filenames used for plots and tables.
Progress indicators print to ``stdout`` and flush on every update so long
optimisations do not appear stalled on Linux terminals.
Dependency checks reuse a cached import list stored in
`.cache/dependency_scan.json`. The cache records the absolute path, size and
modification time of every parsed module so unchanged worktrees skip the AST
walk entirely. The `.cache/` directory is created on demand and is now ignored
by Git so each contributor keeps a private cache that never pollutes commits.
Set `COPERNICAN_DEP_CACHE_DIR` to point the cache at a custom location when
running the suite from read-only media or temporary clones.
Model YAML files are
sanitised and cached under `models/cache/` for the duration of the session,
avoiding repeated schema validation. For CMB analyses unlensed CAMB spectra
are cached by rounded parameter tuples which keeps successive evaluations
fast during optimisation loops.

Each run directory also contains a YAML manifest named
`run_manifest_<timestamp>.yml` capturing the suite version, chosen models,
engine, parameter priors, dataset hashes and the Git commit. See
[docs/run_manifest.md](docs/run_manifest.md) for details on using this file to
reproduce analyses.

Fatal signals such as ``SIGILL``, ``SIGSEGV`` or ``SIGFPE`` trigger handlers
that dump stack traces to the console and active log file before termination.

## Creating New Models
All model details, including theory text and equations, must be stored in a
single YAML file. Markdown summaries are optional and have no effect on the
software. To create a new model:
See `cosmo_model_template.yml` for a detailed template.
1. Copy an existing `cosmo_model_*.yml` file and edit the fields to describe
   your theory.
2. *(Optional)* Create `cosmo_model_name.md` if you want a human-friendly
   summary of the same content. The suite does not read this file.
3. Include an `Hz_expression` written in LaTeX math form defining `H(z)` using
   your model parameters. Explicit `*` is optional since implicit
multiplication
   is now supported, though adding it can improve readability.
4. Optionally provide an `rs_expression` in LaTeX for the sound horizon at
   recombination or include the parameters `Omega_b`, `Omega_gamma` and either
   `z_rec` or `z_recomb`. The suite will then
   derive `r_s` automatically using a numerical integral. Use `\infty` when an
   integral extends to infinity.
5. Python code must never appear in `cosmo_model_*.yml`; all expressions are
   written in LaTeX.
6. Backslashes may be written normally; the parser automatically escapes them
   so
   LaTeX commands like `\frac` work without doubled characters. Prefer YAML
   block scalars (`|` or `>`) for long expressions instead of quoting strings;
   this avoids accidental escape sequences such as `\beta` becoming a
   backspace.
7. Expressions may include `Integral(...)` terms with explicit limits. They
   are
   evaluated numerically with SciPy's `quad` when the model is loaded.
8. Parameter initial guesses are calculated automatically as the midpoint of
   each parameter's bounds.
9. Each parameter may define a `prior` block describing sampling assumptions.
   `type: gaussian` requires `mean` and `sigma` while `type: uniform` needs
   `lower` and `upper`. Engines expose these via `PARAMETER_PRIORS`.
10. Every parameter must define a `latex_name`. When a `python_var` field is
    omitted, a valid identifier is derived automatically from this LaTeX
    name. Provide an explicit `python_var` when you want short variable names
    such as `Omega_b` or `A1`; the refreshed sample models show this pattern.
11. `latex_name` values do not require `$` delimiters. Plots automatically
    wrap parameter names in math mode.
12. Console and log outputs display parameter names with Greek letters,
    subscripts and superscripts when possible for easier reading. The
    conversion tables cover every Latin and Greek letter, digits and common
    operators.

### Updated example models
The non-\LambdaCDM samples now demonstrate several design patterns:

* `cosmo_model_cfsc.yml` shows how to drive sound-horizon fits with explicit
  phenomenological parameters.
* `cosmo_model_cpc.yml` illustrates a compact f(R) toy model using explicit
  `python_var` names instead of escaped LaTeX.
* `cosmo_model_qauc.yml` and the refreshed `cosmo_model_usmf{3..7}.yml` files
  document different shrink-based expansion laws while keeping YAML easy to
  extend. These examples double as regression fixtures that should parse
  cleanly without hand-editing cached outputs.

**Common mistakes**
* Missing `*` between variables and parentheses results in a `'Symbol' object
  is not callable` error.
* Using `oo` for infinite limits fails; write `\infty` instead.
* Referencing `H(z)` inside `rs_expression` is unsupported—repeat the formula
  or rely on the fallback parameters.

The LaTeX parser supports a subset of math syntax including `\frac`,
subscripts and superscripts, common functions (`\log`, `\ln`, `\exp`, `\sin`,
`\cos`, `\tan`, `\csc`, `\sec`, `\cot`, `\arcsin`, `\arccos`, `\arctan`,
`\sinh`, `\cosh`, `\tanh`, `\coth`, `\sech`, `\csch`, `\arcsinh`, `\arccosh`,
`\arctanh`, `\sqrt`, `\abs`, `\floor`, `\ceil`), Greek letters such as
`\alpha`
and `\beta`, and
macros that adjust bracket size like `\left`, `\right`, `\bigl` and `\bigr`.
Thin spaces (`\,`) and font switches (`\rm`) are ignored. Unsupported sizing
macros are removed from plot labels to keep Matplotlib's MathText parser
happy.
All sanitisation rules now live in `copernican_lib/latex_utils.py` with
extensible mappings stored in `latex_mappings.yml`. Expressions may also
contain `Integral` constructs with explicit limits which are numerically
evaluated with SciPy. Use `\infty` for an infinite upper bound and avoid
referencing `H(z)` inside other expressions—repeat the formula instead.
The suite validates the YAML, stores a sanitized copy under `models/cache/` as
YAML, and auto-generates the necessary Python functions.

Initial guesses are derived automatically from each parameter's bounds.
### YAML Schema
The required top-level keys are `model_name`, `version`, `parameters`,
`equations`, `abstract` and `description`.
```yaml
model_name: My Model
version: "1.0"
parameters:
  - name: H0
    bounds: [50, 100]
    latex_name: H_0
  - name: Omega_m0
    bounds: [0.1, 0.5]
    latex_name: \Omega_{m0}
Hz_expression: "H(z) = H_0 * \sqrt{Om0*(1+z)^3 + Ol0}"
rs_expression: "r_s = custom_expression"
equations:
  sne:
    - "d_L(z) = (1+z) \int_0^z \frac{c\,dz'}{H(z')}"
    - "\mu(z) = 5\log_{10}[d_L(z)/{\rm Mpc}] + 25"
  bao:
    - "D_M(z) = \int_0^z \frac{c\,dz'}{H(z')}"
    - "D_H(z) = \frac{c}{H(z)}"
    - "D_V(z) = [D_M(z)^2 D_H(z)]^{1/3}"
valid_for_cmb: true
cmb:
  param_map:
    H0: H_0
    ombh2: "\Omega_{b0} * (H_0/100)**2"
    omch2: "(\Omega_{m0} - \Omega_{b0}) * (H_0/100)**2"
    tau: 0.054
    As: 2.1e-9
    ns: 0.965
gravitational_waves: {}
standard_sirens: {}
abstract: short overview text
description: longer explanation
notes: any additional remarks
```
When a `cmb.param_map` object is provided, the mapping is stored on the plugin
as `CMB_PARAM_MAP`. Call `plugin.get_camb_params(values)` to convert a list of
cosmological parameters into a dictionary for CAMB. Constant numeric values in
the mapping are interpreted as extra fit parameters by engines that expose
them so the CMB spectrum can be adjusted independently. The engines themselves
call
CAMB using this mapping; the plugin no longer provides a fallback
`compute_cmb_spectrum` implementation. When `valid_for_cmb` is `false` the
suite skips the CMB evaluation stage for that model. Expressions are parsed
by a restricted interpreter that aborts when recursion depth or AST node
count limits are exceeded to block runaway evaluation.
CMB data parsers attach a `param_names` attribute to the returned DataFrame
listing the CAMB parameter order—including `omnuh2` when relevant. The engine
combines this list with `get_camb_params` to evaluate the power spectrum and
chi-squared. The CMB plotter draws separate TT, TE and EE panels with
residuals, uses a logarithmic scale for temperature and $E$-mode spectra and
shows cosmic-variance and observational uncertainty bands. Titles now use
minimal padding so each label fits neatly between CMB subplots.
`model_parser.py` accepts unknown keys and simply copies them to the sanitized
cache. This allows the domain-specific YAML language to evolve while remaining
compatible with older models.
`model_parser.py` validates this structure and `model_coder.py` translates the
LaTeX expressions into NumPy-ready callables. When `Hz_expression` is present
it is
compiled into `get_Hz_per_Mpc` and related distance functions used by
`engine_interface.py`. If an `rs_expression` or the parameters `Omega_b`,
`Omega_gamma` and either `z_rec` or `z_recomb` are provided, a callable
`get_sound_horizon_rs_Mpc` is also generated.

## Developer Guide
Document every change in `CHANGELOG.md`. Each substantive update must add an
entry using the template `- YYYY-MM-DD: short summary (author)`.
Legacy `dev_note` headers embedded in source files have been removed in favour
of changelog entries.
Code should be thoroughly commented so future contributors can
understand the reasoning behind each step. The documentation in `README.md`
and
`AGENTS.md` must be updated whenever behavior or structure changes.
See `CHANGELOG.md` for the complete project history.
The short file `CONTRIBUTING.md` summarises the basic workflow for submitting
patches and links back to these guidelines.

The Copernican Suite License forbids redistributing the full suite and
prohibits patent filings or assertions. All contributions must adhere to these
restrictions.

To start developing, install the suite in editable mode:

```bash
pip install -e .
```

Install and run the pre-commit hooks to apply Black, Isort, Ruff, Flake8 and
the Copernican policy checks:

```bash
pre-commit install
pre-commit run --all-files
```

The local `copernican-policy` hook verifies that no file declares a future
"Last Updated" date, enforces version synchronisation between `README.md`,
`CITATION.cff` and `copernican_lib/VERSION`, and forbids direct `print()`
calls inside `copernican_lib/` modules outside the console helpers. The
standard whitespace fixers, Ruff auto-fixes and formatting hooks run before
the custom policy check to keep style adjustments automated.

The local `make-lock` hook now bootstraps a dedicated Python environment and
installs `pip-tools==7.4.1` before executing `make lock`. This keeps
`pip-compile` available for both developer workflows and CI runs without
requiring manual package management outside the managed stack.

Run the tests with:

```bash
python -m unittest discover -v
```

Set `COPERNICAN_STRICT_WARNINGS=1` to treat all warnings as errors during
any run. Set `COPERNICAN_AUTO_INSTALL=1` to install missing dependencies
without prompting.

Pull requests trigger the ``Lint`` workflow, which executes `pre-commit run
--all-files`, and the ``Tests`` workflow, which runs the unit suite across
Windows, macOS and Debian-based Linux. Each job executes inside a cached
virtual environment for reproducibility and speed.

Multiprocessing is used by several engines. The program enforces the `spawn`
start method when it launches so that each worker process begins with a fresh
Python interpreter. Model YAML files are validated with `jsonschema` only in
the
main process; child processes simply read the sanitized cache.
All engines import progress helpers from `copernican_lib/optim_utils.py` so
that
evaluation counting and reporting remain consistent across backends. The
helpers update the console at most once every ten evaluations or half a
second to keep progress readable.

New models are described entirely by YAML. Copy an existing file from
`models/`
and consult `cosmo_model_template.yml` for the full schema. Additional engines
may
be placed under `engines/` and must follow the interface in
`copernican_lib/engine_interface.py`.

**Note:** The current plotting style and algorithms are considered stable. Do
not modify them unless explicitly instructed.

### Workflow Overview

1.  **Dependency Check**: `copernican.py` scans for missing packages,
    prompts before installing them with `pip` and verifies the environment.
    Set `COPERNICAN_AUTO_INSTALL=1` to skip the prompt in automated runs.
2.  **Optional Tests**: Choose "Run the unit test suite" from the launcher
    or run `python -m unittest discover -v` to verify that the LCDM model
    and data parsers work as expected. This command performs unittest
    discovery over the `tests` package and streams informational messages,
    warnings and errors. Combine with `COPERNICAN_STRICT_WARNINGS=1` to fail
    on any warning.
3.  **Initialization**: The script starts and creates the `./output/`
    directory
    for all results.
4.  **Random Seed Setup**: Early in the run the suite either reads
    ``COPERNICAN_SEED`` or prompts for a seed. NumPy, Python's ``random``
    module and supported engines are seeded with this value, which is written
    to the log and run manifest.
5.  **Configuration**: The user specifies the file paths for the model and
    data files.
6.  **SNe Ia Sampling**: The active engine—currently `cosmo_engine_mcmc`—
    samples the parameters of both the ΛCDM model and the alternative model
    against the SNe Ia data. Matching models reuse the first chain to avoid
    redundant computation.
7.  **BAO Analysis**: Using the MAP parameters returned by the sampler, the
    engine calculates BAO observables for each model.
8.  **CMB Analysis**: Each model's CMB spectrum is evaluated against the
    selected dataset with any engine-provided auxiliary parameters folded
    into the CAMB call.
9.  **Output Generation**: `plotter`, `csv_writer` and `logger` save plots,
    tables and logs using a consistent format. Plots now use a white
    background with very light grey, solid grid lines for clarity.
10. **Loop or Exit**: The user is prompted to run another evaluation or exit.

### Development History & Roadmap

See `CHANGELOG.md` for complete version history.

### AI-driven and human development laws and protocols

> **To any AI or human developer, including my future self, that modifies this
codebase:**
>
> This project is developed through a combination of human direction and AI
implementation. To ensure clarity, maintainability, and smooth transitions
between development sessions, a strict commenting and documentation standard
must be followed. The `AGENTS.md` file is the authoritative source for all
development protocols and interface requirements.
>
> 1. **Summarize every change in `CHANGELOG.md` using the changelog**
> template.** Legacy `dev_note` headers should be migrated to the changelog
> when touched.
> 2. **Comment the code extensively.** Explain the "why" as well as the
> "what", clarifying both obvious and non-obvious, simple or complex logic or
> algorithms.
> 3. **Keep comments synchronized with the actual code.** Whenever behaviour
> changes, update all nearby comments immediately so future contributors can
> rely on them.
> 4. **Update documentation**, including this `AGENTS.md`, `README.md` and the
> `docs/` directory, whenever behaviour or structure changes. Each task must
> expand the documentation's scope and size, refresh version strings and
> ensure every file carries a `Last Updated` field. Update that field on
> every edit and add one when missing.
> 5. **Keep these laws synchronized across `README.md` and `AGENTS.md`.**
> Amendments to any rule require an explicit human request.
> 6. **Bump the project version according to Semantic Versioning whenever**
> changes introduce new features, fixes or breaking changes.
> 7. **Never insert Git conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) in**
> any file.
> 8. **Re-read the "AI-driven and human development laws and protocols"
> section in `README.md` at the start of every development session.**
> 9. **Document every module, function and class with clear "what" and "why"
> explanations.** Comments and docstrings should describe not only the
> behaviour but also the rationale behind it.
> 10. **Use concise, descriptive function and identifier names that
accurately** convey their purpose without unnecessary length.
> 11. **Use raw strings or escape backslashes explicitly to avoid invalid**
> escape sequence warnings in docstrings or string literals.
> 12. **Run `pre-commit` on all modified files before committing to enforce**
> Black, Isort, Ruff and Flake8 checks.
> 13. **Do not redistribute the Copernican Suite in full or assert patent**
> claims; the license forbids these actions.
> 14. **Keep individual lines under 79 characters to maintain readability.**
> 15. **Treat documentation refresh as integral to every task.** No change is
> complete until all relevant texts reflect the update and version numbers
> remain in sync.
> 16. **Commit changes only after all tests pass on every supported platform.**
> 17. **Treat `start.command`, `start.bat` and `start.sh` equally.** When one
> launcher is fixed, examine the others for the same issue and update them as
> needed. Consider how code changes impact these launchers and modify them when
> required.
> 18. **Follow current compliance and security requirements for all work.** The
> suite processes user-provided files, so every change must meet the latest
> security guidelines and account for their effect on the `start.*` scripts.
> 19. **Add tests alongside new functionality or behaviour changes.** Each
> feature or fix must include unit tests demonstrating the intended
> behaviour.
> 20. **Audit licenses for new dependencies.** Ensure added packages are
> license-compatible and update `THIRD_PARTY_LICENSES.md` and the
> `licenses/` directory accordingly.
> 21. **Run the suite exclusively through the managed virtual environment.**
> Always launch via `start.sh`, `start.command` or `start.bat` so the
> repository's `.venv` is created or updated automatically; other Python
> environments must be ignored.
> 22. **Refresh dependencies whenever packages are added or changed.**
>    Run `python -m piptools compile requirements.in --allow-unsafe
>    --output-file requirements.lock` (or simply `make lock`), commit the
>    updated `requirements.lock`, and audit `THIRD_PARTY_LICENSES.md`.
> 23. **Verify every `Last Updated` field reflects the actual current date**
>     before committing. Investigate prior human edits to understand their
>     intent and never overwrite those timestamps without confirming they are
>     genuinely stale.
>
> Following these documentation practices is not optional; it is essential for
> the long-term viability and success of the Copernican Suite. Failure to
> follow these rules will compromise the maintainability of the Copernican
> Suite.

See [docs/api_overview.md](docs/api_overview.md) for the scripting API.
All contributors must re-read this section at the beginning of every
development session. The AGENTS.md file now instructs this explicitly.

## License
The Copernican Suite is distributed under the terms of the [Copernican Suite
License (CSL)](LICENSE.md). The license forbids redistributing the software in
full and disallows patent filings or assertions. Licenses for runtime
dependencies are listed in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md), and the corresponding
license texts ship under [`licenses/`](licenses/). CAMB is licensed under
LGPL-3.0-or-later; you may relink the suite against a modified CAMB as
described in that license.

## Versioning Policy
The project now follows [Semantic Versioning](https://semver.org/). Versions
are listed as `MAJOR.MINOR.PATCH`, where breaking changes increment `MAJOR`,
new features increment `MINOR` and bug fixes increment `PATCH`. The canonical
version is stored in two places inside the repository: the heading at the top
of this README and the tracked `copernican_lib/VERSION` file. Keep both in
sync when preparing a release so runtime code and documentation report the
same identifier. Runtime code calls `copernican_lib.version.get_version` which
reads the version file before consulting installed package metadata or Git
tags.

Set ``COPERNICAN_VERSION`` during builds to supply custom prerelease
identifiers, for example in CI when building off feature branches. When
publishing wheels from a work-in-progress branch, export the value written to
`copernican_lib/VERSION` via ``COPERNICAN_VERSION`` so the package metadata
matches the runtime identifier.

The `MINOR` value only increases when the suite gains a new data type or a
similarly significant feature, such as introducing CMB support or a new
engine.
Routine bug fixes and small feature restorations bump the `PATCH` value
without
altering `MAJOR.MINOR`.
