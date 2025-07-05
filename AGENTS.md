# Copernican Suite Development Guide

Development notes were previously kept at the top of this file. That history now
lives in `CHANGELOG.md`. New modifications must update the changelog, and legacy
`dev_note` headers embedded in source files have been fully phased out.

This document is the authoritative reference for contributors and AI systems working on the Copernican Suite. It replaces all previous specifications. The current development release is **version 1.7.2-beta**.

## 1. Program Overview
The suite evaluates cosmological models against SNe Ia and BAO data. Support for
additional observations such as CMB, gravitational waves and standard sirens is
being prepared. Users interact with `copernican.py`, choose a model from
`./models/`, pick a computational engine from `./engines/` and choose data
sources. Parsers reside alongside their data. Results are saved under
`./output/`.

The default engine is `engines/cosmo_engine_1_4b.py`. All model plugins are validated
through `scripts/engine_interface.py` before being passed to the engine. This
ensures the expected functions are present and callable.

## 2. Directory Layout
```
models/           - JSON model definitions with embedded theory text and equations. Optional `.md` files may accompany a model for readability.
engines/          - Computational backends (SciPy CPU by default)
data/             - Observation files under ``data/<type>/<source>/``
  cmb/planck2018lite/ - Planck 2018 lite TT power spectrum
output/           - Generated plots and CSV tables
AGENTS.md         - Development specification and contributor rules
CHANGELOG.md      - Release history
```
Files in `data/` are read-only and must not be modified by AI-driven changes.

The current plotting style and algorithms are considered stable. Do not alter
them without explicit instruction.

## 3. Dependency Installation
`copernican.py` scans all project files for imported modules. If any required
package is missing, the program prints a `pip install` command listing **all**
detected dependencies and then exits. Run that command manually to install or
upgrade packages (already installed libraries will be skipped). This
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

### 4.1 JSON Model File
The schema requires `model_name`, `version`, `parameters`, `equations`, `abstract` and `description`.
Optional fields such as `unit` and `latex_name` provide additional context.
`scripts/model_parser.py` validates the JSON and writes a sanitized copy to
`models/cache/`. `scripts/model_coder.py` transforms the equations into NumPy
callables. These callables are validated by `scripts/engine_interface.py` before
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
4. **Do not change the project version number unless explicitly requested by a human contributor.**
5. **Never insert Git conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) in any file.**

Failure to follow these guidelines will compromise the Copernican Suite.

## 7. Versioning Policy
The project follows Semantic Versioning (`MAJOR.MINOR.PATCH`). Increment the
`MAJOR` number for breaking changes, the `MINOR` for new backward-compatible
features and the `PATCH` for bug fixes. Package versions are derived from Git
tags using `setuptools_scm`.
