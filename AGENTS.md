# DEV NOTE (v1.5e)
Added Numba and OpenCL engines for Phase 5 and bumped version references.

# Copernican Suite Development Guide

This document is the authoritative reference for contributors and AI systems working on the Copernican Suite. It replaces all previous specifications.

## 1. Program Overview
The suite evaluates cosmological models against SNe Ia and BAO data. Users interact with `copernican.py`, choose a model from `./models/`, pick a computational engine from `./engines/` and select data parsers from `./parsers/`. Results are saved under `./output/`.

The default engine is `engines/cosmo_engine_1_4b.py`. Starting with version 1.5e
an `engine_interface` module validates model callables before handing them to the
engine. Optional back ends `cosmo_engine_numba.py` and `cosmo_engine_opencl.py`
provide experimental acceleration. Legacy plugins are still supported through this layer.

## 2. Directory Layout
```
models/           - JSON model definitions and legacy Python plugins
engines/          - Computational backends (SciPy CPU by default)
parsers/          - Data format parsers for SNe and BAO
data/             - Example data files
output/           - Generated plots and CSV tables
AGENTS.md         - Development specification and contributor rules
CHANGELOG.md      - Release history
```
Files in `data/` are read-only and must not be modified by AI-driven changes.

## 3. Model Plugin System (introduced in v1.4b)
Each cosmological model consists of two files stored in `models/`:
1. **Markdown definition** (`cosmo_model_name.md`)
2. **Python plugin** (`name.py`)

### 3.1 Markdown Definition File
The Markdown file is the single source of truth for parameters and equations. It must begin with a YAML front matter block containing `title`, `version`, `date`, and `model_plugin`. Directly after, include a section titled:
```
## Quantitative Model Specification for Copernican Suite
```
Within this section provide:
- `### Key Equations` describing SNe and BAO formulas in LaTeX.
- `### Model Parameters` containing a Markdown table with these headers exactly:
  `Parameter Name`, `Python Variable`, `Initial Guess`, `Bounds`, `Unit`, `LaTeX Name`.

The files `models/cosmo_model_usmf3b.md` and `models/cosmo_model_lcdm.md` show fully compliant examples.

### 3.2 Python Plugin File
The Python module referenced by `model_plugin` implements the numerical functions used by `cosmo_engine_1_4b.py`. It must define the following global variables:
- `MODEL_NAME`
- `MODEL_DESCRIPTION`
- `MODEL_EQUATIONS_LATEX_SN`
- `MODEL_EQUATIONS_LATEX_BAO`
- `PARAMETER_NAMES`
- `PARAMETER_LATEX_NAMES`
- `PARAMETER_UNITS`
- `INITIAL_GUESSES`
- `PARAMETER_BOUNDS`
- `FIXED_PARAMS`

And it must implement these functions with the exact names and signatures:
```python
def distance_modulus_model(z_array, *cosmo_params):
    ...

def get_comoving_distance_Mpc(z_array, *cosmo_params):
    ...

def get_luminosity_distance_Mpc(z_array, *cosmo_params):
    ...

def get_angular_diameter_distance_Mpc(z_array, *cosmo_params):
    ...

def get_Hz_per_Mpc(z_array, *cosmo_params):
    ...

def get_DV_Mpc(z_array, *cosmo_params):
    ...

def get_sound_horizon_rs_Mpc(*cosmo_params):
    ...
```
Refer to `models/usmf3b.py` for a concise analytic implementation and `models/lcdm.py` for a more complex numerical example.

## 3.3 JSON Model DSL (introduced in v1.5d)
Models can now be defined entirely in JSON without accompanying Python code. A
`cosmo_model_name.json` file contains:
```
{
  "model_name": "Example",
  "version": "1.0",
  "date": "2025-06-18",
  "parameters": [{"name": "H0", "latex": "$H_0$", "guess": 70.0,
                   "bounds": [50,100], "unit": "km/s/Mpc"}],
  "equations": {"sne": ["mu = ..."], "bao": ["D_V = ..."]},
  "constants": {"C_LIGHT_KM_S": 299792.458}
}
```
The parser validates this schema and caches a sanitized copy in `models/cache/`.
`model_compiler.py` then converts the equations into callables consumed by the
engines.

## 4. Creating a New Model
1. Copy an existing `cosmo_model_*.json` as a template and edit the metadata,
   parameter list and equation strings.
2. Place the JSON file in the `models/` directory. Legacy Markdown and plugin
   pairs are still loaded if present but are no longer required.
3. Run `model_parser.py` to validate the file and generate a cache entry.
4. Execute `model_compiler.py` on the cached file to produce engine-ready
   callables.
5. Verify the new model using the checklist below.

### 4.1 JSON Template
Use the following structure when creating `cosmo_model_*.json` files.

```json
{
  "model_name": "Model Name",
  "version": "1.0",
  "date": "2025-06-18",
  "parameters": [
    {"name": "H0", "latex": "$H_0$", "guess": 70.0,
     "bounds": [50.0, 100.0], "unit": "km/s/Mpc"}
  ],
  "equations": {
    "sne": ["mu = ..."],
    "bao": ["D_V = ..."]
  },
  "constants": {
    "C_LIGHT_KM_S": 299792.458
  }
}
```

The parser will reject files missing required fields or malformed equations.

### Verification Checklist
- Does the JSON file include `model_name`, `version`, `date`, `parameters`, and `equations`?
- Do parameter objects provide `name`, `latex`, `guess`, `bounds`, and `unit`?
- Does `model_compiler.py` successfully generate callables from the cached JSON?
- Can `copernican.py` run with the compiled model without errors?

## 5. Development Protocol
To keep the project maintainable all contributors, human or AI, must follow these rules:
1. **Add a `DEV NOTE` at the top of each changed file** summarizing your modifications.
2. **Comment code extensively** to explain non-obvious logic or algorithms.
3. **Update documentation**, including this `AGENTS.md` and `README.md`, whenever behavior or structure changes.
4. **Version updates require explicit human request.** The version number of the suite may not be changed autonomously.
5. **Never insert Git conflict markers** (`<<<<<<<`, `=======`, `>>>>>>>`) in any file.

Failure to follow these guidelines will compromise the Copernican Suite.
