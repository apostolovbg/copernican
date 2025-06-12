# Copernican Suite - A Modular Cosmology Framework

## Current Status (v1.4g - UniStra Parsing Restored)

**DEV NOTE (Session: 20250612_1530): This document has been updated to `v1.4g`. The UniStra parsers now replicate the successful fixed-width logic from v1.3 to fully load all 740 supernovae.**

**Version 1.4rc remains unstable and is not suitable for any use.**
The full history of the project is maintained in `CHANGELOG.md`. Development guidelines can be found in `AGENTS.md`.

The primary goal of the v1.4rc stabilization effort has been blocked by a single, persistent, and difficult bug in `data_loaders.py` that resulted in a failure to load the complete supernova dataset. All previous attempts to fix this have failed.

**Root Cause Analysis of the Data Loading Failure:**
* **The Problem:** The `_load_unistra_fixed_nuisance_h1` parser was consistently loading only 33 of 740 supernovae from `tablef3.dat`.
* **The Reason:** The parser was configured to read data from the **incorrect columns**. It was reading from columns that contained placeholders or non-essential data, which were then correctly identified as invalid and dropped, leading to the massive data loss.
* **The Law of the Land (`v1.3` Logic):** The stable `v1.3` version of the parser worked because it correctly targeted the columns for redshift, distance modulus, and error, and correctly handled placeholder values as `NaN`s *during* the initial read.

**The Path Forward:**
The UniStra parsers in `data_loaders.py` have been rewritten to **exactly replicate the column-targeting and NaN-handling logic of the `1.3` script.** The computational engine has also been updated (see `cosmo_engine_1.4g.py`) and now runs without the previous `TypeError`.

---

## Overview

The Copernican Suite is a Python-based, modular framework designed for cosmological data analysis. It allows users to test different cosmological models against observational data, such as Type Ia Supernovae (SNe Ia) and Baryon Acoustic Oscillations (BAO). Its primary goal is to provide a flexible and extensible platform for researchers to compare theoretical models with empirical evidence.

## Architecture

The suite is composed of several key modules that work in a pipeline:

* **`copernican.py`**: The main orchestrator.
* **`input_aggregator.py`**: Assembles the `Job JSON`.
* **`data_loaders.py`**: Contains data parsers. **This is the current point of failure.** Its parsers for UniStra-type data **must** be rewritten to use the correct column indices and `NaN` handling from the v1.3 implementation to ensure all 740 SNe are loaded from `tablef3.dat`.
* **`cosmo_engine_*.py`**: The modular computational engine. The stable implementation is `cosmo_engine_1.4g.py` which works with the standard model plugins.
* **`output_manager.py`**: Dispatches output tasks.
* **`csv_writer.py`**: Writes tabular data.
* **`plotter.py`**: Generates plots based on the style guide.

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
3.  **Documentation First**: Before implementing new features, the `README.md` and `doc.json` files should be updated to reflect the proposed changes, serving as a specification.
tkyfah-codex/fix-bug-in-data_readers.py-and-explain-long-term-solution
4.  **No Conflict Markers**: Avoid any merge conflict markers (the special sequences inserted by Git during merges) in comments or documentation.
4.  **No Conflict Markers**: Avoid sequences like "<<<<<<<", "======", or ">>>>>>>" in comments or documentation, as they appear as merge conflict markers.
1.4g

