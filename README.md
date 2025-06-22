**Version:** 1.6.3
**Last Updated:** 2025-06-22

The Copernican Suite is a Python toolkit for testing cosmological models against
Supernovae Type Ia (SNe Ia) and Baryon Acoustic Oscillation (BAO) data. Future
releases will also handle Cosmic Microwave Background (CMB) measurements,
gravitational waves and standard siren events. The suite provides a modular
architecture so new models, data parsers and computational engines can be
plugged in with minimal effort.

---

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Directory Layout](#directory-layout)
4. [Using the Suite](#using-the-suite)
5. [Creating New Models](#creating-new-models)
6. [Development Notes](#development-notes)
7. [AI Development Laws](#ai-development-laws)
8. [License](#license)
9. [Versioning Policy](#versioning-policy)

---

## Overview
The suite compares the reference ΛCDM model with alternative theories
provided by the user. Each model is defined entirely by a JSON file
`cosmo_model_*.json` under `./models/`. This JSON stores all theory text,
equations and parameters and serves as the sole source of truth. Optional
Markdown summaries may exist for human readers but are ignored by the
software.
Users select models, datasets, and computational engines at runtime through a
simple command line interface. Results are saved as plots and CSV files in the
`./output/` directory.
Each generated plot now includes a footer noting the comparison details,
Copernican Suite version and a timestamp.

Under the hood the program follows a clear pipeline:
1. **Dependency Check** – `copernican.py` scans for required packages and
   prints a `pip install` command if any are missing.
2. **Initialization** – the output directory is created and logging begins.
3. **Configuration** – the user chooses a model, an engine from `./engines/`,
  and data parsers for SNe Ia and BAO. Models are discovered from
  `cosmo_model_*.json` files which are converted into Python code on the fly.
4. **SNe Ia Fitting** – the selected engine estimates cosmological parameters
   for both the ΛCDM reference and the alternative model.
5. **BAO Analysis** – using the best-fit parameters the engine predicts BAO
   observables and computes chi-squared statistics.
6. **Output Generation** – `scripts/logger.py`, `scripts/plotter.py` and `scripts/csv_writer.py` handle logs, plots and tables.
7. **Loop or Exit** – the user may evaluate another model or quit, at which
   point temporary cache files are cleaned automatically.

## Quick Start
1. Ensure Python 3 is available. The suite requires `numpy`, `scipy`,
   `matplotlib`, `pandas`, `sympy` and `jsonschema`. If any package is
   missing the program will print the command to install them.
2. Run `python3 copernican.py` and follow the prompts to choose a model,
   preferred data sources and computation engine.
3. Plots and CSV results will appear in the `output/` folder when the run
   completes.

## Dependencies
The program relies on `numpy`, `scipy`, `matplotlib`, `pandas`, `sympy`,
`jsonschema`. If any of these are missing the dependency check
will print the full installation command `pip install numpy scipy matplotlib
pandas sympy jsonschema`. Future engines may also depend on `numba` or
GPU libraries.
 
## Building & Installation
Run `pip install .` from the repository root to build and install the `copernican` command. Use `pip install -e .` for editable installs.

## Directory Layout
```
models/           - JSON model definitions containing all theory text and
                    equations. Optional `.md` files may provide human-readable
                    summaries but are not required.
engines/          - Computational backends (SciPy CPU and Numba with automatic fallback)
data/             - Observation data organized as ``data/<type>/<source>/``
output/           - All generated results
AGENTS.md         - Development specification and contributor rules
CHANGELOG.md      - Release history
scripts/          - Helper modules
  logger.py         - Logging setup and helpers
  plotter.py        - Plotting functions
  csv_writer.py     - CSV output helpers
  data_loaders.py   - Data loading utilities
  utils.py          - Common helpers
```
**Note:** Files in `data/` are treated as read-only reference datasets and
should not be modified by AI-driven code changes.

## Using the Suite
- The program discovers available models from `models/cosmo_model_*.json`.
- Data sources for SNe and BAO are chosen interactively. Once a source is
  selected, its parser and files are loaded automatically from
  `data/<type>/<source>/`. Future datasets will follow the same structure.
- Engines are selected interactively from the `engines/` directory. Parsers are
  discovered automatically when their source folders are imported.
- After each run you may choose to evaluate another model or exit. Cache files
  are cleaned automatically.

## Creating New Models
All model details, including theory text and equations, must be stored in a
single JSON file. Markdown summaries are optional and have no effect on the
software. To create a new model:
See `cosmo_model_guide.json` for a detailed template.
1. Copy an existing `cosmo_model_*.json` file and edit the fields to describe
   your theory.
2. *(Optional)* Create `cosmo_model_name.md` if you want a human-friendly
   summary of the same content. The suite does not read this file.
3. Include an `Hz_expression` string defining `H(z)` in terms of your model
   parameters. This enables BAO and distance-based predictions.
4. Optionally provide an `rs_expression` for the sound horizon at recombination
   or include the parameters `Ob`, `Og` and `z_recomb`. The suite will then
   derive `r_s` automatically using a numerical integral.
5. Parameter initial guesses are calculated automatically as the midpoint of
   each parameter's bounds.
The suite validates the JSON, stores a sanitized copy under `models/cache/`, and
auto-generates the necessary Python functions.

### JSON Schema
The schema requires `model_name`, `version`, `parameters`, `equations`, `abstract` and `description`.
```json
{
  "model_name": "My Model",
  "version": "1.0",
  "Hz_expression": "H0 * sympy.sqrt(Om*(1+z)**3 + Ol)",
  "rs_expression": "integrate(c_s/H, (z, z_recomb, inf))",
  "parameters": [
    {"name": "H0", "python_var": "H0", "bounds": [50, 100]},
    {"name": "Ob", "python_var": "Ob", "bounds": [0.01, 0.1]},
    {"name": "Og", "python_var": "Og", "bounds": [1e-5, 1e-4]},
    {"name": "z_recomb", "python_var": "z_recomb", "bounds": [1000, 1200]}
  ],
  "equations": {
    "distance_modulus_model": "5*sympy.log(1+z,10)*H0"
  },
  "cmb": {},
  "gravitational_waves": {},
  "standard_sirens": {},
  "abstract": "short overview text",
  "description": "longer explanation with optional equations",
  "notes": "any additional remarks",
  "title": "Human friendly model title",
  "date": "2025-06-20"
}
```
Initial guesses are derived automatically from each parameter's bounds.
`model_parser.py` accepts unknown keys and simply copies them to the sanitized
cache. This allows the domain-specific JSON language to evolve while remaining
compatible with older models.
`model_parser.py` validates this structure and `model_coder.py` translates the
equations into NumPy-ready callables. When `Hz_expression` is present it is
compiled into `get_Hz_per_Mpc` and related distance functions used by
`engine_interface.py`. If an `rs_expression` or the parameters `Ob`, `Og` and
`z_recomb` are provided, a callable `get_sound_horizon_rs_Mpc` is also generated.

## Development Notes
Document every change in `CHANGELOG.md`. Each substantive update must add an entry using the template `- YYYY-MM-DD: short summary (author)`.
Legacy `dev_note` headers embedded in source files have been removed in favour of changelog entries.
Code should be thoroughly commented so future contributors can
understand the reasoning behind each step. The documentation in `README.md` and
`AGENTS.md` must be updated whenever behavior or structure changes.
See `CHANGELOG.md` for the complete project history.

**Note:** The current plotting style and algorithms are considered stable. Do
not modify them unless explicitly instructed.

## AI Development Laws
1. **Record each modification in `CHANGELOG.md` using the changelog template.**
2. **Comment code extensively** to clarify complex logic or algorithms.
3. **Update all documentation**, including this `README.md` and `AGENTS.md`,
   whenever the codebase changes.
4. **Never add Git conflict markers** (`<<<<<<<`, `=======`, `>>>>>>>`) in any file. These break automated merges and will be rejected.

Failure to follow these rules will compromise the maintainability of the
Copernican Suite.

## License
The Copernican Suite is distributed under the terms of the [Copernican Suite License (CSL)](LICENSE.md).

## Versioning Policy
The project now follows [Semantic Versioning](https://semver.org/). Versions are
listed as `MAJOR.MINOR.PATCH`, where breaking changes increment `MAJOR`, new
features increment `MINOR` and bug fixes increment `PATCH`. Package builds use
`setuptools_scm` to derive the version from Git tags.

The `MINOR` value only increases when the suite gains a new data type or a
similarly significant feature, such as introducing CMB support or a new engine.
Routine bug fixes and small feature restorations bump the `PATCH` value without
altering `MAJOR.MINOR`.

## 4. Workflow Overview

1.  **Dependency Check**: `copernican.py` scans for missing packages and
    instructs you to run a `pip install` command if any are absent.
2.  **Initialization**: The script starts and creates the `./output/` directory for all results.
3.  **Configuration**: The user specifies the file paths for the model and data files.
    -   **Test Mode**: A user can enter `test` to run ΛCDM against itself, providing a quick way to test the full analysis pipeline.
4.  **SNe Ia Fitting**: The `cosmo_engine` fits the parameters of both the ΛCDM model and the alternative model to the SNe Ia data.
5.  **BAO Analysis**: Using the best-fit parameters, the engine calculates BAO observables for each model.
6.  **Output Generation**: `plotter`, `csv_writer` and `logger` save plots, tables and logs using a consistent format.
7.  **Loop or Exit**: The user is prompted to run another evaluation or exit.

---

## 5. Development History & Roadmap

See `CHANGELOG.md` for complete version history.

## 6. A Note on AI-Driven Development

> **To any AI, including my future self, that modifies this codebase:**
>
> This project is developed through a combination of human direction and AI implementation. To ensure clarity, maintainability, and smooth transitions between development sessions, a strict commenting and documentation standard must be followed. The `AGENTS.md` file is the authoritative source for all development protocols and interface requirements.
>
> **When modifying any file, you are required to:**
> 1.  **Document all modifications in `CHANGELOG.md` using the changelog template.**
> 2.  **Comment the code extensively.** Explain the "why" behind your code, not just the "what".
> 3.  **Update this README file and `AGENTS.md`**. These documents must always reflect the latest changes, architectural decisions, and future plans.
>
> Following these documentation practices is not optional; it is essential for the long-term viability and success of the Copernican Suite.
