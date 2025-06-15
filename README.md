# Copernican Suite
# DEV NOTE (v1.5a): Updated pipeline references; model_converter.py renamed to
# model_encoder.py.

**Version:** 1.5a
**Last Updated:** 2025-06-16

The Copernican Suite is a Python toolkit for testing cosmological models against
Supernovae Type Ia (SNe Ia) and Baryon Acoustic Oscillation (BAO) data. It
provides a modular architecture that allows new models, data parsers and
computational engines to be plugged in with minimal effort.

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

Under the hood the program now follows a modular pipeline:
1. **Dependency Check** – `copernican.py` verifies all required Python
   libraries.
2. **Initialization** – logging via `scripts/logger.py` and the output directory
   are prepared.
3. **Model Parsing** – `scripts/model_parser.py` validates `cosmo_model_*.json`
   files and caches the sanitized content.
4. **Model Encoding** – `scripts/model_encoder.py` turns the cached data
   into Python callables used by the engines.
5. **Engine Execution** – `scripts/engine_interface.py` hands the callables and
   parsed data to the selected engine.
6. **Output Generation** – `scripts/plotter.py` and `scripts/csv_writer.py`
   produce plots and CSV summaries while `output_manager.py` coordinates file
   placement.
7. **Error Handling** – `scripts/error_handler.py` reports issues in a
   consistent format.
8. **Loop or Exit** – the user may evaluate another model or quit. Cache files
   remain until the end of the run.

## Quick Start
1. Ensure Python 3 with `numpy`, `scipy`, `matplotlib` and `psutil` is
   installed.
2. Run `python3 copernican.py` and follow the prompts to choose a model, data
   files and engine.
3. Plots and CSV results will appear in the `output/` folder when the run
   completes.

## Directory Layout
```
models/           - Markdown definitions and Python plugins
engines/          - Computational backends (SciPy CPU by default)
parsers/          - Data format parsers for SNe and BAO
data/             - Example data files
output/           - All generated results
AGENTS.md         - Development specification and contributor rules
CHANGELOG.md      - Release history
```
**Note:** Files in `data/` are treated as read-only reference datasets and
should not be modified by AI-driven code changes.

## Using the Suite
- The program discovers available models from `models/cosmo_model_*.md`.
- Data files for SNe and BAO are chosen interactively from `data/sne` and
  `data/bao`.
- Parsers and engines are also selected interactively from their respective
  directories.
- After each run you may choose to evaluate another model or exit. Cache files
  are cleaned automatically.

## Creating New Models
Model definition historically used a two-file system (Markdown + Python module).
The legacy approach is preserved here for reference and is documented in
`AGENTS.md`:
1. **Markdown file** (`cosmo_model_name.md`) describing equations and providing
   a table of parameters. Each model file should conclude with the *Internal
   Formatting Guide for Model Definition Files* so contributors understand the
   required structure.
2. **Python plugin** implementing the required functions listed in `AGENTS.md`.
   These plugins will be retired once every Markdown file is migrated to the JSON
   DSL, but they remain as examples for now.

Version 1.5a introduces an experimental **JSON DSL** for model definitions. A
`cosmo_model_name.json` file will eventually replace both the Markdown file and
the Python plugin. It contains:

- `model_name`, `version`, and `date` metadata
- a list of parameter objects with `name`, `latex`, `guess`, `bounds`, and
  `unit`
- LaTeX or SymPy strings for SNe Ia and BAO equations
- optional constants or fixed parameters

See `models/cosmo_model_lcdm.json` for a minimal example.

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

1.  **Dependency Check**: `copernican.py` verifies required Python libraries.
2.  **Initialization**: Logging via `scripts/logger.py` starts and the `./output/` directory is created.
3.  **Configuration**: The user specifies model and data paths. Test mode (`test`) runs ΛCDM against itself.
4.  **Model Parsing and Encoding**: The JSON DSL is processed by `model_parser.py` and `model_encoder.py`.
5.  **SNe Ia Fitting**: The selected engine receives callables through `engine_interface.py` and fits parameters.
6.  **BAO Analysis**: Using best-fit parameters, the engine computes BAO observables.
7.  **Output Generation**: Plots and CSVs are produced via `plotter.py`, `csv_writer.py`, and `output_manager.py`.
8.  **Loop or Exit**: The user may evaluate another model or exit.

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
