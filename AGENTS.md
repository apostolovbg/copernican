# DEV NOTE (v1.5a)
This file was rewritten entirely to document the current Copernican Suite structure and the model plugin system introduced in version 1.4b.

# Copernican Suite Development Guide

This document is the authoritative reference for contributors and AI systems working on the Copernican Suite. It replaces all previous specifications.

## 1. Program Overview
The suite evaluates cosmological models against SNe Ia and BAO data. Users interact with `copernican.py`, choose a model from `./models/`, pick a computational engine from `./engines/` and select data parsers from `./parsers/`. Results are saved under `./output/`.

The default engine is `engines/cosmo_engine_1_4b.py`. It imports a model's Python plugin based on the `model_plugin` field of a Markdown definition file.

## 2. Directory Layout
```
models/           - Markdown definitions and Python plugins
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

## 4. Creating a New Model
1. Copy `cosmo_model_usmf3b.md` and `usmf3b.py` as templates.
2. Edit the Markdown file's YAML block and parameter table to describe the new model.
3. Implement the Python plugin using the required variables and functions. The parameter lists must correspond exactly to the table in the Markdown file.
4. Place both files in the `models/` directory. `copernican.py` will automatically discover them.
5. Verify your plugin by running the checklist below.

### 4.1 Markdown Template
Use the following structure when creating `cosmo_model_*.md` files. The
`model_plugin` field must reference the Python plugin and the
`## Quantitative Model Specification for Copernican Suite` section must
contain `### Key Equations` and a parameter table.

```markdown
---
title: "Model Name"
version: "1.0"
date: "2025-06-14"
model_plugin: "model_name.py"
---

## Quantitative Model Specification for Copernican Suite

### Key Equations
Provide LaTeX equations for SNe Ia and BAO here.

### Model Parameters
| Parameter Name | Python Variable | Initial Guess | Bounds | Unit | LaTeX Name |
| :--- | :--- | :--- | :--- | :--- | :--- |
| H0 | `H0` | 70.0 | (50.0, 100.0) | km/s/Mpc | `$H_0$` |
```

Append the **Internal Formatting Guide for Model Definition Files** after
your model description so future developers understand the format.

### 4.2 Python Plugin Skeleton
The matching Python module must define all global metadata variables and
implement the interface functions exactly as shown below:

```python
MODEL_NAME = "My Model"
MODEL_DESCRIPTION = "Short summary."
MODEL_EQUATIONS_LATEX_SN = []
MODEL_EQUATIONS_LATEX_BAO = []
PARAMETER_NAMES = []
PARAMETER_LATEX_NAMES = []
PARAMETER_UNITS = []
INITIAL_GUESSES = []
PARAMETER_BOUNDS = []
FIXED_PARAMS = {}

def distance_modulus_model(z_array, *params):
    pass

def get_comoving_distance_Mpc(z_array, *params):
    pass

def get_luminosity_distance_Mpc(z_array, *params):
    pass

def get_angular_diameter_distance_Mpc(z_array, *params):
    pass

def get_Hz_per_Mpc(z_array, *params):
    pass

def get_DV_Mpc(z_array, *params):
    pass

def get_sound_horizon_rs_Mpc(*params):
    pass
```

Ensure the parameter order matches the Markdown table exactly.

### Verification Checklist
- Does the `.py` file define all required global variables?
- Are `distance_modulus_model`, `get_comoving_distance_Mpc`, `get_luminosity_distance_Mpc`, `get_angular_diameter_distance_Mpc`, `get_Hz_per_Mpc`, `get_DV_Mpc`, and `get_sound_horizon_rs_Mpc` implemented?
- Do the parameter lists match the Markdown table?
- Can `copernican.py` run with the new model without raising import errors?

## 5. Development Protocol
To keep the project maintainable all contributors, human or AI, must follow these rules:
1. **Add a `DEV NOTE` at the top of each changed file** summarizing your modifications.
2. **Comment code extensively** to explain non-obvious logic or algorithms.
3. **Update documentation**, including this `AGENTS.md` and `README.md`, whenever behavior or structure changes.
4. **Version updates require explicit human request.** The version number of the suite may not be changed autonomously.
5. **Never insert Git conflict markers** (`<<<<<<<`, `=======`, `>>>>>>>`) in any file.

Failure to follow these guidelines will compromise the Copernican Suite.
