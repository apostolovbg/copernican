# Copernican Suite
<!-- DEV NOTE (v1.5f): Updated for Phase 6 with new data-type placeholders and schema fields. -->
<!-- DEV NOTE (v1.5f hotfix): Dependency scanner ignores relative imports; JSON models now support "sympy." prefix. -->

**Version:** 1.5f
**Last Updated:** 2025-06-20
engines/          - Computational backends (SciPy CPU by default, plus Numba)

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

---

## Overview
The suite compares the reference \(\Lambda\)CDM model with alternative theories
provided by the user. Each model is defined by a Markdown file under
`./models/` and a matching Python implementation residing in the same package.
Users select models, datasets, and computational engines at runtime through a
simple command line interface. Results are saved as plots and CSV files in the
`./output/` directory.

Under the hood the program follows a clear pipeline:
1. **Dependency Check** – `copernican.py` scans for required packages and
   installs any that are missing before continuing.
2. **Initialization** – the output directory is created and logging begins.
3. **Configuration** – the user chooses a model, an engine from `./engines/`,
  and data parsers for SNe Ia and BAO. Models are discovered from
  `cosmo_model_*.json` files which are converted into Python code on the fly.
4. **SNe Ia Fitting** – the selected engine estimates cosmological parameters
   for both the ΛCDM reference and the alternative model.
5. **BAO Analysis** – using the best-fit parameters the engine predicts BAO
   observables and computes chi-squared statistics.
6. **Output Generation** – `logger.py`, `plotter.py` and `csv_writer.py` handle logs, plots and tables.
7. **Loop or Exit** – the user may evaluate another model or quit, at which
   point temporary cache files are cleaned automatically.

## Quick Start
1. Ensure Python 3 is available. The suite requires `numpy`, `scipy`,
   `matplotlib`, `pandas`, `sympy`, `psutil` and `jsonschema`. Missing packages
   are installed automatically when the program starts.
2. Run `python3 copernican.py` and follow the prompts to choose a model, data
   files and engine.
3. Plots and CSV results will appear in the `output/` folder when the run
   completes.

## Dependencies
The suite automatically installs any missing Python packages at startup. Current
packages include `numpy`, `scipy`, `matplotlib`, `pandas`, `sympy`, `psutil` and
`jsonschema`. Future engines may also depend on `numba` or GPU libraries.

## Directory Layout
```
models/           - JSON model definitions (Markdown optional)
engines/          - Computational backends (SciPy CPU and Numba)
parsers/          - Data format parsers for SNe, BAO, CMB, gravitational waves and standard sirens
data/             - Example data files
output/           - All generated results
AGENTS.md         - Development specification and contributor rules
CHANGELOG.md      - Release history
logger.py         - Logging setup and helpers
plotter.py       - Plotting functions
csv_writer.py    - CSV output helpers
```
**Note:** Files in `data/` are treated as read-only reference datasets and
should not be modified by AI-driven code changes.

## Using the Suite
- The program discovers available models from `models/cosmo_model_*.md`.
- Data files for SNe and BAO are chosen interactively from `data/sne` and
  `data/bao`. Future datasets such as CMB or gravitational waves will use
  their own folders.
- Parsers and engines are also selected interactively from their respective
  directories.
- After each run you may choose to evaluate another model or exit. Cache files
  are cleaned automatically.

## Creating New Models
All models are now provided as a single JSON file. Markdown files can still be
included for explanatory text but are not required. To create a new model:
1. Copy an existing `cosmo_model_*.json` file and edit the fields to describe
   your theory.
2. Optionally create `cosmo_model_name.md` to document the equations in LaTeX so
   other researchers can read them easily.
The suite validates the JSON, stores a sanitized copy under `models/cache/`, and
auto-generates the necessary Python functions.

### JSON Schema
```json
{
  "model_name": "My Model",
  "version": "1.0",
  "parameters": [
    {"name": "H0", "python_var": "H0", "initial_guess": 70.0, "bounds": [50, 100]}
  ],
  "equations": {
    "distance_modulus_model": "5*sympy.log(1+z,10)*H0"
  },
  "cmb": {},
  "gravitational_waves": {},
  "standard_sirens": {}
}
```
`model_parser.py` validates this structure and `model_coder.py` translates the
equations into NumPy-ready callables used by `engine_interface.py`.

## Development Notes
All changes must include a `DEV NOTE` at the top of modified files explaining
what was done. Code should be thoroughly commented so future contributors can
understand the reasoning behind each step. The documentation in `README.md` and
`AGENTS.md` must be updated whenever behavior or structure changes.
See `CHANGELOG.md` for the complete project history.

## AI Development Laws
1. **Add a `DEV NOTE` to every changed file** summarizing modifications.
2. **Comment code extensively** to clarify complex logic or algorithms.
3. **Update all documentation**, including this `README.md` and `AGENTS.md`,
   whenever the codebase changes.

Failure to follow these rules will compromise the maintainability of the
Copernican Suite.
## 4. Workflow Overview

1.  **Dependency Check**: `copernican.py` scans for missing packages and launches
    an installer if necessary before continuing.
2.  **Initialization**: The script starts and creates the `./output/` directory for all results.
3.  **Configuration**: The user specifies the file paths for the model and data files.
    -   **Test Mode**: A user can enter `test` to run ΛCDM against itself, providing a quick way to test the full analysis pipeline.
4.  **SNe Ia Fitting**: The `cosmo_engine` fits the parameters of both the ΛCDM model and the alternative model to the SNe Ia data.
5.  **BAO Analysis**: Using the best-fit parameters, the engine calculates BAO observables for each model.
6.  **Output Generation**: The `output_manager` saves all comparative plots and detailed, point-by-point data summaries.
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
> 1.  **Add a `DEV NOTE` at the top of the file.** This note should summarize the changes made in the current version.
> 2.  **Comment the code extensively.** Explain the "why" behind your code, not just the "what".
> 3.  **Update this README file and `AGENTS.md`**. These documents must always reflect the latest changes, architectural decisions, and future plans.
>
> Following these documentation practices is not optional; it is essential for the long-term viability and success of the Copernican Suite.
