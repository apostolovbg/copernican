# Copernican Suite

**Version:** 1.4rc
**Last Updated:** 2025-06-13

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

## Using the Suite
- The program discovers available models from `models/cosmo_model_*.md`.
- Data files for SNe and BAO are chosen interactively from `data/sne` and
  `data/bao`.
- Parsers and engines are also selected interactively from their respective
  directories.
- After each run you may choose to evaluate another model or exit. Cache files
  are cleaned automatically.

## Creating New Models
Model definition follows a two-file system:
1. **Markdown file** (`cosmo_model_name.md`) describing equations and providing
   a table of parameters. See the formatting guide at the end of every model
   file.
2. **Python plugin** implementing the required functions listed in `AGENTS.md`.
   Place this module in the `models` package and reference its filename in the
   Markdown front matter under `model_plugin`.

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
