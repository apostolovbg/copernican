<!-- Development History (for AI reference) -->
# DEV NOTE (v1.5g update): clarified that all model details reside in JSON files; Markdown is optional for readability only.
# DEV NOTE (v1.5.1): Removed legacy ``theory`` section from models and auto-generated
# parameter guesses from bounds midpoints.
- v1.5.0: Semantic versioning adopted; version constant updated.
- v1.5g: Added pyproject configuration and installation instructions.
- Hotfix 1: improved dependency scanner to skip relative imports and added SymPy aliasing in model_coder.
- Hotfix 2: `abstract` and `description` fields are now mandatory in JSON models; `notes` remains optional.
- Hotfix 3: `copernican.py` now performs the dependency check before importing third-party packages to avoid start-up failures. Style fixes applied across the codebase.
- Hotfix 4: Multiprocessing's `freeze_support` is now called using a local import after the dependency check to prevent NoneType errors.
- Hotfix 5: Removed automatic dependency installer. The suite now instructs users to run `pip install` manually when packages are missing.
- Hotfix 7: Models now provide a symbolic `Hz_expression` compiled at runtime for distance calculations.
- Hotfix 8: When `rs_expression` is absent but `Ob`, `Og` and `z_recomb` exist, the suite derives `get_sound_horizon_rs_Mpc` using SciPy's `quad` integral.
- Hotfix 9: Parser auto-discovery fixed to look in the top-level `parsers` directory.
- Phase 6 update: Added placeholder parsers for CMB, gravitational waves and standard sirens; expanded JSON schema.
- Data sources restructured as data/<type>/<source>; parsers reside with their data.
- Hotfix 10: Data source names are now human-readable and selection lists show a descriptive title.
- Hotfix 11: Added GPL-3.0 license and documented license section in README.
- Hotfix 12: Replaced GPL-3.0 with Copernican Suite License and updated README accordingly.
# Copernican Suite Development Guide

This document is the authoritative reference for contributors and AI systems working on the Copernican Suite. It replaces all previous specifications.

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
output/           - Generated plots and CSV tables
AGENTS.md         - Development specification and contributor rules
CHANGELOG.md      - Release history
```
Files in `data/` are read-only and must not be modified by AI-driven changes.

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
    "distance_modulus_model": "5*sympy.log(1+z,10)*H0"
  }
}
```
Initial guesses are computed automatically as the midpoint of each
parameter's bounds.

`model_parser.py` and `model_coder.py` handle validation and code generation
automatically; no manual Python implementation is required.
The parser keeps unknown keys intact, ensuring the DSL stays backward
compatible as new fields are introduced.

## 6. Development Protocol
To keep the project maintainable all contributors, human or AI, must follow these rules:
1. **Summarize every change in `CHANGELOG.md` or another central log agreed upon by the maintainers.** Use the template `- YYYY-MM-DD: short summary (author)` for each entry.
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
