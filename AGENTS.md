# DEV NOTE (v1.5g)
Hotfix: improved dependency scanner to skip relative imports and added SymPy aliasing in model_coder.
Hotfix 2: JSON models now contain optional abstract, description and notes fields.
Hotfix 3: `copernican.py` now performs the dependency check before importing third-party packages to avoid start-up failures. Style fixes applied across the codebase.
Hotfix 4: Multiprocessing's `freeze_support` is now called using a local import after the dependency check to prevent NoneType errors.
Hotfix 5: Removed automatic dependency installer. The suite now instructs users to run `pip install` manually when packages are missing.
Hotfix 7: Models now provide a symbolic `Hz_expression` compiled at runtime for distance calculations.
Hotfix 8: When `rs_expression` is absent but `Ob`, `Og` and `z_recomb` exist, the suite derives `get_sound_horizon_rs_Mpc` using SciPy's `quad` integral.
Hotfix 9: Parser auto-discovery fixed to look in the top-level `parsers` directory.
Updated for Phase 6. Added placeholder parsers for CMB, gravitational waves and standard sirens, and expanded JSON schema.
Data sources restructured as data/<type>/<source>; parsers reside with their data.
Hotfix 10: Data source names are now human-readable and selection lists show a descriptive title.
Hotfix 11: Added GPL-3.0 license and documented license section in README.
Hotfix 12: Replaced GPL-3.0 with Copernican Suite License and updated README accordingly.

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
models/           - JSON model definitions (Markdown files optional)
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

## 4. JSON Model System
As of version 1.5f every cosmological model is described by a single JSON file
`cosmo_model_*.json`. Markdown files may accompany the JSON for human
readability, but there are no permanent Python plugins in the repository.

### 4.1 JSON Model File
The schema requires `model_name`, `version`, `parameters` and `equations`.
Optional fields such as `unit` and `latex_name` provide additional context.
`scripts/model_parser.py` validates the JSON and writes a sanitized copy to
`models/cache/`. `scripts/model_coder.py` transforms the equations into NumPy
callables. These callables are validated by `scripts/engine_interface.py` before
being passed to the chosen engine.

## 5. Creating a New Model
1. Copy an existing `cosmo_model_*.json` file such as `cosmo_model_lcdm.json`.
2. Edit the JSON fields to describe your model, following the schema above.
3. Optionally provide a Markdown file with the same base name to document the
   model for human readers.

### 5.1 JSON Template
Use the following structure when creating new models:

```json
{
  "model_name": "My Model",
  "version": "1.0",
  "parameters": [
    {"name": "H0", "python_var": "H0", "initial_guess": 70.0, "bounds": [50, 100]}
  ],
  "equations": {
    "distance_modulus_model": "5*sympy.log(1+z,10)*H0"
  }
}
```

`model_parser.py` and `model_coder.py` handle validation and code generation
automatically; no manual Python implementation is required.

## 6. Development Protocol
To keep the project maintainable all contributors, human or AI, must follow these rules:
1. **Add a `DEV NOTE` at the top of each changed file** summarizing your modifications.
2. **Comment code extensively** to explain non-obvious logic or algorithms.
3. **Update documentation**, including this `AGENTS.md` and `README.md`, whenever behavior or structure changes.
4. **Do not change the project version number unless explicitly requested by a human contributor.**
5. **Never insert Git conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) in any file.**

Failure to follow these guidelines will compromise the Copernican Suite.
