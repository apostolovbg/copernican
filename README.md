# Copernican Suite - A Modular Cosmology Framework

## Current Status (v2.0 - CosmoDSL Architecture)

codex/debug-copernican-suite-v2.0
**DEV NOTE (v2.0):** The Suite uses a declarative DSL for models and plugin-based engines. `main.py` is the sole entry point. A minimal `doc.json` is kept only for legacy merges and references `README.md` for full details.
**DEV NOTE (v2.0):** The Suite uses a declarative DSL for models and plugin-based engines. `main.py` is the sole entry point. A minimal `doc.json` is kept only for legacy merges and references `README.md` for full details.
**DEV NOTE (v2.0 cleanup):** The obsolete `legacy` folder has been removed and `plotter.py` restored to the project root.

j9ep1u-codex/refactor-copernican-suite-to-cosmodsl
**DEV NOTE (v2.0):** The Suite uses a declarative DSL for models and plugin-based engines. `main.py` is the sole entry point. A minimal `doc.json` is kept only for legacy merges and references `README.md` for full details.
2.0
**DEV NOTE (Session: 20250612_1530): This document has been updated to `v1.4g`. The UniStra parsers now replicate the successful fixed-width logic from v1.3 to fully load all 740 supernovae.**
**DEV NOTE (v2.0 bugfix):** Cleaned CLI functions and removed conflict markers.
**DEV NOTE (Session: 20250612_1600): CosmoDSL folders and plugin engine structure have been introduced.**


The full history of the project is maintained in `CHANGELOG.md`. Development guidelines are found in `AGENTS.md`. A tiny `doc.json` remains solely for merge compatibility with the historic `1.4g` branch.

---

## Overview

The Copernican Suite is a Python-based, modular framework designed for cosmological data analysis. It allows users to test different cosmological models against observational data, such as Type Ia Supernovae (SNe Ia) and Baryon Acoustic Oscillations (BAO). Its primary goal is to provide a flexible and extensible platform for researchers to compare theoretical models with empirical evidence.

## Installation & Usage

1. Ensure Python 3.10+ is installed with NumPy, SciPy, pandas and Matplotlib.
2. Place your data files in the `/data` folder.
3. Run `python main.py` (or use `start.command`, `run.bat`, or `Copernican.desktop`).
4. Select an engine, a model from `/models`, and your data files when prompted.

## Architecture

The suite is composed of several key modules that work in a pipeline:
* **`main.py`**: Entry point that performs dependency checks, shows the splash screen, and launches the menu-driven workflow.
* **`engines/`**: Folder of plugin engines automatically discovered at runtime.
* **`models/`**: CosmoDSL model files.
* **`data/`**: Default location for SNe Ia and BAO datasets.
* **`data_loaders.py`**: Parsers for supported data formats.
* **`cosmo_engine_*.py`**: Individual engine implementations.
The v2.0 release introduces **CosmoDSL** and a plugin-based architecture. `main.py` now handles all user interaction and dynamically loads engines and models from their respective folders.

## Plotting Style

Plots follow a unified theme based on the `seaborn-v0_8-colorblind` style with light backgrounds and readable font sizes. Info boxes and legend colors follow the guidelines from the old `doc.json` specification.
* **`data_loaders.py`**: Parsers for SNe Ia and BAO data.
* **`cosmo_engine_*.py`**: Plugin computational engines. The stable implementation is `cosmo_engine_1.4g.py`.
* **`copernican.py`**: The main orchestrator.
* **`input_aggregator.py`**: Assembles the `Job JSON`.
* **`data_loaders.py`**: Contains data parsers. **This is the current point of failure.** Its parsers for UniStra-type data **must** be rewritten to use the correct column indices and `NaN` handling from the v1.3 implementation to ensure all 740 SNe are loaded from `tablef3.dat`.
* **`cosmo_engine_*.py`**: The modular computational engine. The stable implementation is `cosmo_engine_1.4g.py` which works with the standard model plugins.
* **`output_manager.py`**: Dispatches output tasks.
* **`csv_writer.py`**: Writes tabular data.
* **`plotter.py`**: Generates plots based on the style guide.
The v1.4g refactor introduced **CosmoDSL** and a plugin-based engine folder (`/engines`). Models now live in `/models` as `.md` files written in the DSL, and data files are kept under `/data`. The new `main.py` script automatically discovers engines, models, and data at runtime.


---

## Development History

* **v1.3:** Stable version with a robust `data_loaders.py` that is the reference for the current bugfix.
* **v1.4rc (Initial):** A major refactor that broke the data pipeline.
* **v1.4rc2 - v1.4rc11:** A series of failed attempts to fix the data loading issue. These versions suffered from numerous cascading errors, including `KeyError`, `ValueError`, `TypeError` (string math), and incorrect data filtering, all stemming from the initial broken refactor.
* **v1.4rc12:** The last failed attempt. It incorrectly diagnosed the data loading issue, which was still only loading 33 supernovae. This version's failure made it clear a fundamental misunderstanding of the problem was occurring.
* **v1.4rc13:** Development reset with a focus on reproducing the v1.3 parsing logic. The bug source was confirmed to be incorrect column selection.
* **v1.4g:** The UniStra data loaders now use the v1.3 fixed-width parsing strategy, restoring all 740 SNe. The engine bug has been resolved with `cosmo_engine_1.4g.py`.

---

## Future Vision (v1.5 and beyond)

The long-term vision is to evolve the suite into a "Universal Math Engine" that can parse models directly from structured text files. This is contingent on first achieving a stable, working `v1.4`.

## Note on AI-Driven Development

This project is being developed with the assistance of a large language model (LLM). To ensure clarity, maintainability, and accountability, the following policies are enforced:
1.  **`DEV NOTE`s**: Any file modified by the AI must contain a `DEV NOTE` block at the top, explaining the version, the nature of the changes, and the reason for them.
2.  **Extensive Commenting**: All new or modified code must be commented clearly to explain its logic and purpose.
3.  **Documentation First**: Update `README.md` before adding new features so it always matches the codebase.
4.  **No Conflict Markers**: Avoid any merge conflict markers (such as `<<<<<<<`, `=======`, or `>>>>>>>`) in comments or documentation.

