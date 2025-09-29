# Copernican Suite Development Guide
**Last Updated:** 2025-10-05

Development notes were previously kept at the top of this file. That history
now
lives in `CHANGELOG.md`. New modifications must update the changelog, and
legacy
`dev_note` headers embedded in source files have been fully phased out.


## 1. Program Overview
The helper modules previously stored under `scripts/` now live in the
`copernican_lib/` package.
The suite evaluates cosmological models against SNe Ia, BAO and CMB data.
Support for additional observations such as gravitational waves and standard
sirens is
being prepared. Users interact with `copernican.py`, choose a model from
`./models/`, pick a computational engine from `./engines/` and choose data
sources. Parsers reside alongside their data but are imported only when their
SHA256 digest matches a vetted list to block untrusted files. Results are saved
under `./output/`, each run in a dedicated `copernican-run_YYYYMMDD_HHMMSS`
subdirectory. Each plot carries a centered footer with three lines: the
model comparison, dataset details and the citation. The first and third
lines are bold, while the dataset name on the second line is bolded
using Matplotlib's standard text rendering. Dataset names retain their
original spacing and the second line wraps after 190 characters when
necessary.
Each run directory also includes a `run_manifest_*.yml` file listing the
selected models, engine, dataset hashes and Git state to aid
reproducibility. The data loaders compute and log SHA256 digests for all
non-parser files in each dataset directory and store these hashes on the
returned DataFrames. The manifest copies this mapping verbatim. Parsers
must register under the `dataset_id` stated in their metadata so the
loaders can locate them directly without discovery.

A ``COPERNICAN_SEED`` environment variable overrides the interactive seed
prompt.  When unset, the program asks users to accept the default ``0``, enter
their own value or generate a random seed.  The final choice is stored in the
run manifest and logged so analyses can be reproduced.

The program enables Python's ``faulthandler`` at startup and registers
``SIGILL``, ``SIGSEGV`` and ``SIGFPE`` handlers. When triggered, they dump
stack traces to both the console and the active log file before exiting.
Immediately after logging initialises the suite records the Python version,
operating system, CPU model and key package versions. A short summary is
shown on the console while the log captures full details. Progress messages
print to ``stdout`` and flush on every update so lengthy optimisations still
display activity on Linux terminals.

All Python warnings are forwarded to the central logger. Set
``COPERNICAN_STRICT_WARNINGS=1`` to elevate warnings to errors during CI
runs.

Before any heavy computation, a tiny NumPy/SciPy calculation checks that the
installed binaries match the CPU. If this fails the log explains possible CPU
feature mismatches and suggests reinstalling with suitable wheels.

The default engine is `engines/cosmo_engine_comb.py`. All model plugins are
validated
through `copernican_lib/engine_interface.py` before being passed to the
engine. The BAO χ² helper accepts pre-extracted arrays so callers can
convert data frames once outside optimisation loops. This
ensures the expected functions are present and callable. Chi-squared
values for SNe, BAO and CMB are evaluated concurrently when multiple
datasets are supplied, using processes when objects are picklable and
threads otherwise. Starting with
version 1.11.4 the test suite no longer runs automatically. Launchers offer
a *Run the unit test suite* option which delegates to `python -m unittest
discover` and exits cleanly even when Matplotlib has not yet been imported.

## 2. Directory Layout
```
models/           - YAML model definitions with embedded theory text and
                    equations.
engines/          - Computational backends (SciPy CPU by default)
data/             - Observation files under ``data/<type>/<source>/``. Each
                    dataset directory includes a `metadata_*.yml` file with
                    `dataset_name`, `description`, `citation`, `license`, the
                    full `author` list and BibTeX keys such as `title`,
                    `volume`, `journal` and `DOI`. Metadata is loaded
                    exclusively by
                    `copernican_lib/data_loaders.py` after each parser runs.
  cmb/planck2018lite/ - Planck 2018 lite TT/TE/EE spectra and covariance
output/           - Per-run folders with plots, tables and NetCDF chains
AGENTS.md         - Development specification and contributor rules
CHANGELOG.md      - Release history
copernican_lib/optim_utils.py - Shared optimisation helpers used by engines
copernican_lib/latex_utils.py - LaTeX translation helpers using
                                latex_mappings.yml
copernican_lib/console_output.py - Central console output helpers
```
Installing the suite with `pip` produces a `copernican_suite.egg-info`
directory
containing build metadata. This folder can be safely removed and should not be
edited manually.
Tables under `data/` remain read-only, but parser `.py` files and
`metadata_*.yml` files within that tree may be updated when necessary.

The current plotting style and algorithms are considered stable. Do not alter
them without explicit instruction.

Multiprocessing is enabled across several engines. To guarantee a clean
environment for each worker, the program sets Python's multiprocessing start
method to `spawn` at entry. Model YAML files are validated via `jsonschema`
**only** in the main process; child processes read the sanitized cache without
re-validating to prevent occasional plugin failures under multiprocessing.

All engines should remain purely computational. Shared utilities such as
evaluation counters now live in ``copernican_lib/optim_utils.py`` and are
imported
by the engines instead of being reimplemented inside each backend.

The ``_eval_safe`` helper in ``engine_interface`` caps recursion depth and
AST node count when parsing expressions for ``get_camb_params`` to block
runaway evaluation on malicious or overly complex inputs.

## 3. Dependency Installation
`copernican.py` scans all project files for imported modules using Python's
AST parser to avoid false positives from comments. The `start.*` launchers
always download a private Python 3.12+ into ``.python`` and build ``.venv``
from that interpreter, ignoring any system-wide Python. If the download fails
they exit with guidance. When packages are missing the program asks before
installing them with `pip install -r requirements.lock` and
verifies the import before continuing. Set ``COPERNICAN_AUTO_INSTALL=1`` to
bypass the prompt in non-interactive environments. Running outside ``.venv``
prompts the user to
restart via the appropriate launcher. This lightweight approach works across
Windows, macOS and Linux while allowing new engines to introduce additional
dependencies without manual updates to the documentation.
The launchers print a notice before invoking `sudo`, `brew` or `winget` so
users know any password prompt originates from the package manager and is
never read or stored. `sudo -k` and explicit prompts ensure the operating
system handles all credential entry.
ArviZ is installed from commit
`01c8b9454349247eed2145a27b03f9231acb412f` of the upstream repository to
avoid its former `numpy<2` restriction without maintaining a fork.

`requirements.lock` pins exact versions for all runtime
packages, and `[project].dependencies` in `pyproject.toml` mirrors these pins.
Any dependency change must regenerate both files and update
`THIRD_PARTY_LICENSES.md` to keep license records current (see law 22).
To install the suite as a package, run `pip install .` at the repository root.
Use `pip install -e .` if you intend to develop the code. The start scripts
install pinned dependencies from `requirements.lock`
before running `pip install --no-deps .`. They delete any `build/` directory
before and after installing the project to prevent stale build artifacts and
recreate `.venv` once when the activation script is missing.

Pull requests run a GitHub Actions workflow named ``Tests`` that executes
pre-commit checks and the full unit suite on Ubuntu, macOS and Windows.
Only pull requests trigger the workflow to avoid duplicate push builds.

## 4. YAML Model System
As of version 2.0 every cosmological model is described by a single YAML file
`cosmo_model_*.yml`. All theory text, equations and parameters reside in this
file. Markdown files may mirror the YAML for readability, but models are
distributed only as YAML. No permanent Python plugins exist in the repository.
Models are automatically discovered
by scanning for `cosmo_model_*.yml` files in the `models/` directory.
All expressions inside these YAML files must be written in LaTeX math form;
raw Python code is not permitted.

### 4.1 YAML Model File
The schema requires `model_name`, `version`, `parameters`, `equations`,
`abstract` and `description`.
Optional fields such as `unit` and `latex_name` provide additional context.
`copernican_lib/model_parser.py` validates the YAML and writes a sanitized
copy to `models/cache/`. `copernican_lib/model_coder.py` transforms the
equations into NumPy callables. These callables are validated by
`copernican_lib/engine_interface.py` before being passed to the chosen
engine.
`model_parser.py` ignores unrecognized keys and copies them to the cache, so
new metadata can be added without breaking older YAML files.

## 5. Creating a New Model
1. Copy an existing `cosmo_model_*.yml` file such as `cosmo_model_lcdm.yml`.
2. Edit the YAML fields to describe your model, following the schema above.
3. *(Optional)* Create a Markdown file with the same base name if you want a
   human-readable summary. The YAML file remains the single source of truth.
See `cosmo_model_template.yml` for a complete template.

### 5.1 YAML Template
Use the following structure when creating new models:

```yaml
model_name: My Model
version: "1.0"
parameters:
  - name: H0
    bounds: [50, 100]
    definition: N/A
    latex_name: H_0
equations:
  sne:
    - "d_L(z) = (1+z) \int_0^z \frac{c\,dz'}{H(z')}"
    - "\mu(z) = 5\log_{10}[d_L(z)/{\rm Mpc}] + 25"
  bao:
    - "D_M(z) = \int_0^z \frac{c}{H(z')}"
    - "D_H(z) = \frac{c}{H(z)}"
    - "D_V(z) = [D_M(z)^2 D_H(z)]^{1/3}"
```
Initial guesses are computed automatically as the midpoint of each
parameter's bounds.

`model_parser.py` and `model_coder.py` handle validation and code generation
automatically; no manual Python implementation is required.
The parser keeps unknown keys intact, ensuring the DSL stays backward
compatible as new fields are introduced.

### 4.2 Dataset compatibility flags

Generated model plugins include boolean attributes
`valid_for_distance_metrics`,
`valid_for_bao` and `valid_for_cmb`. All default to `True` and signal which
datasets the model supports. When `valid_for_cmb` is `False` the engine does
not
require the optional `compute_cmb_spectrum` function during validation.
Models that can compute a CMB power spectrum should also define a
`cmb.param_map`
object describing how standard CAMB parameters such as `H0` and `ombh2` are
derived from the model's variables or constants.

## AI-driven and human development laws and protocols
To keep the project maintainable all contributors, human or AI, must follow
these rules:
1. **Summarize every change in `CHANGELOG.md` using the changelog template.**
   Legacy `dev_note` headers should be migrated to the changelog when touched.
2. **Comment the code extensively.** Explain the "why" as well as the "what",
   clarifying both obvious and non-obvious, simple or complex logic or
   algorithms.
3. **Keep comments synchronized with the actual code.** Whenever behaviour
   changes, update all nearby comments immediately so future contributors can
   rely on them.
4. **Update documentation**, including this `AGENTS.md`, `README.md` and the
   `docs/` directory, whenever behaviour or structure changes. Each task must
   expand the documentation's scope and size, refresh version strings and
   ensure every file carries a `Last Updated` field. Update that field on
   every edit and add one when missing.
5. **Keep these laws synchronized across `README.md` and `AGENTS.md`.**
   Amendments to any rule require an explicit human request.
6. **Bump the project version according to Semantic Versioning whenever
   changes introduce new features, fixes or breaking changes.**
7. **Never insert Git conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) in
   any file.**
8. **Re-read the "AI-driven and human development laws and protocols" section
   in `README.md` at the start of every development session.**
9. **Document every module, function and class with clear "what" and "why"
   explanations.** Comments and docstrings should describe not only the
   behaviour but also the rationale behind it.
10. **Use concise, descriptive function and identifier names that accurately
    convey their purpose without unnecessary length.**
11. **Use raw strings or escape backslashes explicitly to avoid invalid escape
    sequence warnings in docstrings or string literals.**
12. **Run `pre-commit` on all modified files before committing to enforce
    Black, Isort, Ruff and Flake8 checks.**
13. **Do not redistribute the Copernican Suite in full or assert patent
    claims; the license forbids these actions.**
14. **Keep individual lines under 79 characters to maintain readability.**
15. **Treat documentation refresh as integral to every task.** No change is
    complete until all relevant texts reflect the update and version numbers
    remain in sync.
16. **Commit changes only after all tests pass on every supported platform.**
17. **Treat `start.command`, `start.bat` and `start.sh` equally.** When one
    launcher is fixed, assess the other two for the same issue and update
    them as needed. Investigate how code changes affect the start scripts and
    adjust them accordingly. Keep multi-line PowerShell calls inside helper
    subroutines when editing `start.bat` so `cmd.exe` never mis-parses
    closing parentheses inside conditional blocks.
18. **Follow current compliance and security requirements for all work.** The
    suite processes user-provided files, so every change must meet the latest
    security guidelines and consider their impact on the `start.*` scripts.
19. **Add tests alongside new functionality or behaviour changes.** Each
    feature or fix must include unit tests demonstrating the intended
    behaviour.
20. **Audit licenses for new dependencies.** Ensure added packages are
    license-compatible and update `THIRD_PARTY_LICENSES.md` and the
    `licenses/` directory accordingly.
21. **Run the suite exclusively through the managed virtual environment.**
    Always launch via `start.sh`, `start.command` or `start.bat` so the
    repository's `.venv` is created or updated automatically; other Python
    environments must be ignored.
22. **Refresh dependencies whenever packages are added or changed.**
   Run `python -m piptools compile requirements.in --allow-unsafe
   --output-file requirements.lock` (or simply `make lock`), commit the
   updated `requirements.lock`, and audit `THIRD_PARTY_LICENSES.md`.
   The local pre-commit hook provisions `pip-tools==7.4.1` automatically
   before invoking `make lock` so the workflow succeeds even in clean CI
   environments.

Failure to follow these guidelines will compromise the Copernican Suite.

## 7. Versioning Policy
The project follows Semantic Versioning (`MAJOR.MINOR.PATCH`). Increment the
`MAJOR` number for breaking changes, the `MINOR` for new backward-compatible
features and the `PATCH` for bug fixes. Package versions are derived from Git
tags using `setuptools_scm`. Runtime code should obtain the current version
via ``copernican_lib.version.get_version`` rather than hard-coded strings.
Contributors must update the version whenever a pull request introduces a
change covered by these rules.
Setting ``COPERNICAN_VERSION`` in the environment overrides the derived
version so CI builds can embed custom prerelease identifiers.
