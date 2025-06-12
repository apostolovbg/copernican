# Copernican Suite - A Modular Cosmology Framework

## Current Status (v1.4rc13 - Definitive Data Loader Fix)

**DEV NOTE (Session: 20250612_1530): This document has been updated to `v1.4rc13`. This version reflects a full stop and reassessment of the persistent data loading bug. The root cause has been definitively identified and the fix is specified herein.**

**Version 1.4rc remains unstable and is not suitable for any use.**

The primary goal of the v1.4rc stabilization effort has been blocked by a single, persistent, and difficult bug in `data_loaders.py` that resulted in a failure to load the complete supernova dataset. All previous attempts to fix this have failed.

**Root Cause Analysis of the Data Loading Failure:**
* **The Problem:** The `_load_unistra_fixed_nuisance_h1` parser was consistently loading only 33 of 740 supernovae from `tablef3.dat`.
* **The Reason:** The parser was configured to read data from the **incorrect columns**. It was reading from columns that contained placeholders or non-essential data, which were then correctly identified as invalid and dropped, leading to the massive data loss.
* **The Law of the Land (`v1.3` Logic):** The stable `v1.3` version of the parser worked because it correctly targeted the columns for redshift, distance modulus, and error, and correctly handled placeholder values as `NaN`s *during* the initial read.

**The Path Forward:**
The immediate and only priority is to rewrite the UniStra parsers in `data_loaders.py` to **exactly replicate the successful column-targeting and NaN-handling logic of the `1.3data_loaders.py` script.** A secondary `TypeError` is expected to appear in the `cosmo_engine` once the data is correctly loaded.

---

## Overview

The Copernican Suite is a Python-based, modular framework designed for cosmological data analysis. It allows users to test different cosmological models against observational data, such as Type Ia Supernovae (SNe Ia) and Baryon Acoustic Oscillations (BAO). Its primary goal is to provide a flexible and extensible platform for researchers to compare theoretical models with empirical evidence.

## Architecture

The suite is composed of several key modules that work in a pipeline:

* **`copernican.py`**: The main orchestrator.
* **`input_aggregator.py`**: Assembles the `Job JSON`.
* **`data_loaders.py`**: Contains data parsers. **This is the current point of failure.** Its parsers for UniStra-type data **must** be rewritten to use the correct column indices and `NaN` handling from the v1.3 implementation to ensure all 740 SNe are loaded from `tablef3.dat`.
* **`cosmo_engine_*.py`**: The computational engine. Will likely expose a secondary `TypeError` once data loading is fixed.
* **`output_manager.py`**: Dispatches output tasks.
* **`csv_writer.py`**: Writes tabular data.
* **`plotter.py`**: Generates plots based on the style guide.

---

## Development History

* **v1.3:** Stable version with a robust `data_loaders.py` that is the reference for the current bugfix.
* **v1.4rc (Initial):** A major refactor that broke the data pipeline.
* **v1.4rc2 - v1.4rc11:** A series of failed attempts to fix the data loading issue. These versions suffered from numerous cascading errors, including `KeyError`, `ValueError`, `TypeError` (string math), and incorrect data filtering, all stemming from the initial broken refactor.
* **v1.4rc12:** The last failed attempt. It incorrectly diagnosed the data loading issue, which was still only loading 33 supernovae. This version's failure made it clear a fundamental misunderstanding of the problem was occurring.
* **v1.4rc13 (This Version):** A full reset. The root cause of the data loss has been identified as reading from the wrong columns. The plan is to implement the correct `v1.3` logic.

---

## Future Vision (v1.5 and beyond)

The long-term vision is to evolve the suite into a "Universal Math Engine" that can parse models directly from structured text files. This is contingent on first achieving a stable, working `v1.4`.

## Note on AI-Driven Development

This project is being developed with the assistance of a large language model (LLM). To ensure clarity, maintainability, and accountability, the following policies are enforced:
1.  **`DEV NOTE`s**: Any file modified by the AI must contain a `DEV NOTE` block at the top, explaining the version, the nature of the changes, and the reason for them.
2.  **Extensive Commenting**: All new or modified code must be commented clearly to explain its logic and purpose.
3.  **Documentation First**: Before implementing new features, the `README.md` and `doc.json` files should be updated to reflect the proposed changes, serving as a specification.