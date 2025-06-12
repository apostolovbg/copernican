# Copernican Suite - A Modular Cosmology Framework

## Current Status (v1.4 Release Candidate - Under Active Debugging)

**Version 1.4rc is an unstable development build and is not suitable for production use.**

The original goal for the v1.4 release was to build upon v1.4b and finalize the plotting functionality, particularly for Baryon Acoustic Oscillation (BAO) data. However, development has been blocked by a series of critical, cascading errors that prevent the program from completing a single successful run.

Previous debugging sessions to fix the plotting were sidetracked, leading to a loss of context and further instability in the codebase. The current debugging effort is focused on restoring core functionality that was broken during a major refactoring between versions. Specifically, the data parsing algorithms in `data_loaders.py` were compromised. A working version of this file from `v1.3` is being used as a reference to restore the correct data handling logic.

The primary objective is to stabilize the application. Testing and development of the plotting features cannot resume until the program can execute a full analysis without crashing.

---

## Overview

The Copernican Suite is a Python-based, modular framework designed for cosmological data analysis. It allows users to test different cosmological models against observational data, such as Type Ia Supernovae (SNe Ia) and Baryon Acoustic Oscillations (BAO). Its primary goal is to provide a flexible and extensible platform for researchers and enthusiasts to explore cosmological paradigms.

## Architecture (v1.4)

The suite is designed with a decoupled architecture to promote modularity and ease of development.

-   **`copernican.py`**: The main orchestrator and user interface. It manages the overall workflow, from user input to dispatching jobs.
-   **`input_aggregator.py`**: Gathers all necessary inputs—model selections, data file paths, and parser choices—and assembles them into a standardized JSON job file.
-   **`cosmo_engine_*.py`**: The computational heart of the suite. These are swappable engines that perform the intensive calculations, such as parameter fitting (e.g., using `scipy.optimize.minimize`) and generating model predictions.
-   **Model Plugins (`.py` files)**: Self-contained Python files that define a specific cosmological model. Each plugin must provide a `METADATA` dictionary containing the model's name, equations (in LaTeX), parameter definitions, and the core physics functions (`get_distance_modulus_mu`, etc.).
-   **`data_loaders.py`**: Contains a library of parser functions for different observational data formats.
-   **`output_manager.py`**: Manages the post-processing stage. It receives a "Results JSON" from the engine and dispatches tasks to the appropriate output modules.
-   **`csv_writer.py`**: A submodule of the output manager that writes detailed numerical results to CSV files.
-   **`plotter.py`**: A submodule of the output manager that generates all plots (e.g., Hubble diagrams, BAO plots) based on the results and a strict style guide.

## Future Vision (v1.5 and beyond)

The long-term vision is to evolve the suite into a "Universal Math Engine". The current reliance on model-specific `.py` plugins will be deprecated in favor of a system that can parse mathematical models directly from structured text files (e.g., Markdown with a YAML header).

This would allow users to define new models by simply writing out their equations and parameter specifications in a `.md` file, making the suite accessible to users without Python programming experience. The engine would parse these files to dynamically generate the necessary computational objects, creating a truly universal tool for cosmological exploration.

## Note on AI-Driven Development

This project is being developed with the assistance of a large language model (LLM). To ensure clarity, maintainability, and accountability, the following policies are enforced:
1.  **`DEV NOTE`s**: Any file modified by the AI must contain a `DEV NOTE` block at the top, explaining the version, the nature of the changes, and the reason for them.
2.  **Extensive Commenting**: All new or modified code must be commented clearly to explain its logic and purpose.
3.  **Documentation First**: Before implementing new features, the `README.md` and `doc.json` files should be updated to reflect the proposed changes, serving as a specification.