# Copernican Suite Development Guide

Development notes were previously kept at the top of this file. That history now
lives in `CHANGELOG.md`. New modifications must update the changelog, and legacy
`dev_note` headers embedded in source files have been fully phased out.

This document is the authoritative reference for contributors and AI systems working on the Copernican Suite. It replaces all previous specifications. The current development release is **version 1.13.0**.

## 1. Program Overview
The helper modules previously stored under `scripts/` now live in the `copernican_lib/` package.
The suite evaluates cosmological models against SNe Ia, BAO and CMB data.
Support for additional observations such as gravitational waves and standard sirens is
being prepared. Users interact with `copernican.py`, choose a model from
`./models/`, pick a computational engine from `./engines/` and choose data
sources. Parsers reside alongside their data. Results are saved under
`./output/`.

The default engine is `engines/cosmo_engine_comb.py`. All model plugins are validated
through `copernican_lib/engine_interface.py` before being passed to the engine. This
ensures the expected functions are present and callable. Starting with
version 1.11.4 the test suite no longer runs automatically. Execute
`copernican.py --run-tests` or run `python -m unittest discover` to verify that
the reference LCDM model and data parsers operate correctly. The `--run-tests`
flag now uses Python's built-in discovery to gather all tests from the `tests`
package and will exit cleanly even when Matplotlib has not yet been imported.

## 2. Directory Layout
```
models/           - JSON model definitions with embedded theory text and equations.
engines/          - Computational backends (SciPy CPU by default)
data/             - Observation files under ``data/<type>/<source>/``
  cmb/planck2018lite/ - Planck 2018 lite TT/TE/EE spectra and covariance
output/           - Generated plots and CSV tables (created automatically)
AGENTS.md         - Development specification and contributor rules
CHANGELOG.md      - Release history
copernican_lib/optim_utils.py - Shared optimisation helpers used by engines
```
Installing the suite with `pip` produces a `copernican_suite.egg-info` directory
containing build metadata. This folder can be safely removed and should not be
edited manually.
Files in `data/` are read-only and must not be modified by AI-driven changes.

The current plotting style and algorithms are considered stable. Do not alter
them without explicit instruction.

Multiprocessing is enabled across several engines. To guarantee a clean
environment for each worker, the program sets Python's multiprocessing start
method to `spawn` at entry. Model JSON files are validated via `jsonschema`
**only** in the main process; child processes read the sanitized cache without
re-validating to prevent occasional plugin failures under multiprocessing.

All engines should remain purely computational. Shared utilities such as
evaluation counters now live in ``copernican_lib/optim_utils.py`` and are imported
by the engines instead of being reimplemented inside each backend.

## 3. Dependency Installation
`copernican.py` scans all project files for imported modules using Python's AST
parser to avoid false positives from comments. If any required package is
missing, the program prints an install command tailored to the current operating
system and lists only the missing packages. Run that command manually to install
or upgrade packages (already installed libraries will be skipped). This
lightweight approach works across Windows, macOS and Linux while allowing new
engines to introduce additional dependencies without manual updates to the
documentation.
To install the suite as a package, run `pip install .` at the repository root. Use `pip install -e .` if you intend to develop the code.

## 4. JSON Model System
As of version 1.5f every cosmological model is described by a single JSON file
`cosmo_model_*.json`. All theory text, equations and parameters reside in this
file. Markdown files may mirror the JSON for readability, but models are
distributed only as JSON. No permanent Python plugins exist in the repository.
Models are automatically discovered
by scanning for `cosmo_model_*.json` files in the `models/` directory.
All expressions inside these JSON files must be written in LaTeX math form; raw Python code is not permitted.

### 4.1 JSON Model File
The schema requires `model_name`, `version`, `parameters`, `equations`, `abstract` and `description`.
Optional fields such as `unit` and `latex_name` provide additional context.
`copernican_lib/model_parser.py` validates the JSON and writes a sanitized copy to
`models/cache/`. `copernican_lib/model_coder.py` transforms the equations into NumPy
callables. These callables are validated by `copernican_lib/engine_interface.py` before
being passed to the chosen engine.
`model_parser.py` ignores unrecognized keys and copies them to the cache, so
new metadata can be added without breaking older JSON files.

## 5. Creating a New Model
1. Copy an existing `cosmo_model_*.json` file such as `cosmo_model_lcdm.json`.
2. Edit the JSON fields to describe your model, following the schema above.
3. *(Optional)* Create a Markdown file with the same base name if you want a
   human-readable summary. The JSON file remains the single source of truth.
See `cosmo_model_guide.json` for a complete template.

### 5.1 JSON Template
Use the following structure when creating new models:

```json
{
  "model_name": "My Model",
  "version": "1.0",
  "parameters": [
    {"name": "H0", "python_var": "H0", "bounds": [50, 100]}
  ],
  "equations": {
    "sne": [
      "$$d_L(z) = (1+z) \\int_0^z \\frac{c\\,dz'}{H(z')}$$",
      "$$\\mu(z) = 5\\log_{10}[d_L(z)/{\\rm Mpc}] + 25$$"
    ],
    "bao": [
      "$$D_M(z) = \\int_0^z \\frac{c\\,dz'}{H(z')}$$",
      "$$D_H(z) = \\frac{c}{H(z)}$$",
      "$$D_V(z) = [D_M(z)^2 D_H(z)]^{1/3}$$"
    ]
  }
}
```
Initial guesses are computed automatically as the midpoint of each
parameter's bounds.

`model_parser.py` and `model_coder.py` handle validation and code generation
automatically; no manual Python implementation is required.
The parser keeps unknown keys intact, ensuring the DSL stays backward
compatible as new fields are introduced.

### 4.2 Dataset compatibility flags

Generated model plugins include boolean attributes `valid_for_distance_metrics`,
`valid_for_bao` and `valid_for_cmb`. All default to `True` and signal which
datasets the model supports. When `valid_for_cmb` is `False` the engine does not
require the optional `compute_cmb_spectrum` function during validation.
Models that can compute a CMB power spectrum should also define a `cmb.param_map`
object describing how standard CAMB parameters such as `H0` and `ombh2` are
derived from the model's variables or constants.

## 6. Development Protocol
To keep the project maintainable all contributors, human or AI, must follow these rules:
1. **Summarize every change in `CHANGELOG.md`.** Use the template `- YYYY-MM-DD: short summary (author)` for each entry. Legacy `dev_note` headers should be migrated to the changelog when touched.
2. **Comment code extensively** to explain non-obvious logic or algorithms.
3. **Update documentation**, including this `AGENTS.md` and `README.md`, whenever behavior or structure changes.
4. **Bump the project version according to Semantic Versioning whenever changes introduce new features, fixes or breaking changes.**
5. **Never insert Git conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) in any file.**
6. **Use raw string literals for regular expressions, docstrings with LaTeX or backslashes, and Windows paths** to avoid Python's "invalid escape sequence" warnings.

Failure to follow these guidelines will compromise the Copernican Suite.

## 7. Versioning Policy
The project follows Semantic Versioning (`MAJOR.MINOR.PATCH`). Increment the
`MAJOR` number for breaking changes, the `MINOR` for new backward-compatible
features and the `PATCH` for bug fixes. Package versions are derived from Git
tags using `setuptools_scm`. Contributors must update the version whenever
a pull request introduces a change covered by these rules.
