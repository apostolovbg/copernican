# Copernican Suite - Full Documentation

**Version:** 1.2.1 (OpenCL Integration)
**Last Updated:** 2025-06-08

---

## 1. Project Overview

### Description

The **Copernican Suite** is a Python-based framework for the modular testing and comparison of cosmological models against observational data. This version removes the Numba dependency and introduces a new, user-selectable OpenCL backend for GPU acceleration, allowing for high-performance fitting of complex models.

### Goals

-   To provide a robust and extensible platform for fitting cosmological models to observational data.
-   To compare alternative cosmological models against the standard LambdaCDM model using consistent statistical measures.
-   To ensure consistent parameter handling when testing a model across different datasets.
-   To support various data formats through a modular data loading system.
-   To allow easy integration of new cosmological models via a standardized plugin architecture, with a clear path for GPU acceleration through OpenCL.

---

## 2. Architecture

The suite is designed with a primary project directory containing all core scripts and model plugins. All outputs (logs, plots, CSVs) are saved into a dedicated `output` subdirectory. The architecture is modular to facilitate easy extension and maintenance.

### Core Modules

-   **`copernican.py`**: The main orchestrator script. It handles user interaction, dynamically loads model plugins, manages the sequential workflow, and directs all file outputs to the `output/` subdirectory.
-   **`data_loaders.py`**: Manages the loading and initial parsing of various observational datasets through a registered parser system.
-   **`cosmo_engine.py`**: Contains the core physics and statistical logic. This includes chi-squared functions, the SNe parameter fitting algorithm, and a high-performance function dispatcher that prompts the user to select between the standard SciPy backend and a GPU-accelerated OpenCL backend.
-   **`output_manager.py`**: Handles all forms of output: logging, generating comparative plots (e.g., binned residual averages), and saving results to CSV files.
-   **Model Plugins (`*.py`) & Definitions (`*.md`)**: A two-file system for each model. A Markdown file defines the theory and parameters, while a Python file contains the computational implementation.

---

## 3. Workflow Overview

1.  **Initialization & Configuration:** `copernican.py` starts, creates an `output/` directory if one does not exist, and initializes logging.
2.  **Backend Selection:** The user is prompted to select the computational backend (Standard CPU or OpenCL GPU) if compatible OpenCL hardware is detected.
3.  **User Input:** The user is prompted to specify the alternative model plugin file, the SNe Ia data file, and the BAO data file.
4.  **SNe Ia Fitting:** For both LambdaCDM and the specified alternative model, `cosmo_engine.fit_sne_parameters` is called. The engine's dispatcher uses the user-selected backend to find the best-fit cosmological parameters.
5.  **BAO Analysis (Post-SNe Fit):** The SNe-fitted parameters for each model are used to calculate the model's BAO observable predictions and the corresponding chi-squared value.
6.  **Output Generation:** `output_manager.py` is called to generate and save all comparative plots (Hubble Diagram, BAO observables) and summary CSV files into the `output/` subdirectory.

---

## 4. Plugin Development Guide

One of the suite's primary goals is extensibility. Researchers can easily test new cosmological theories by creating a model definition file and a corresponding implementation.

### The Two-File System

A model is defined by a two-file system: a Markdown (`.md`) file for the theoretical and parametric definition, and a Python (`.py`) file for the implementation. This separates the human-readable theory from the computational code. Each new model requires two files with matching base names (e.g., `my_theory.md` and `my_theory.py`) located in the main project directory.

### Model Definition File (`.md`)

This file provides the human-readable theory and the machine-parsable parameter definitions, serving as the source of truth for the model's metadata.
-   **Format**: Markdown (`.md`).
-   **Structure**: Must contain a YAML front matter block and a "Quantitative Model Specification" section with a parameter table. See `usmf2.md` for a template.

### Model Implementation File (`.py`)

This file contains the functions that perform the cosmological calculations. Its metadata block should be generated from the `.md` file's table. It must contain the following components:

#### Required Metadata Block

A set of global variables at the top of the Python file that define the model's identity and parameters for the fitter.
-   `MODEL_NAME`
-   `PARAMETER_NAMES`
-   `INITIAL_GUESSES`
-   `PARAMETER_BOUNDS`
-   `MODEL_EQUATIONS_LATEX_SN` / `_BAO` (Optional)

#### Required Functions

A set of functions that calculate cosmological observables, accepting `z_array` and `*cosmo_params` as primary arguments.
-   `distance_modulus_model`
-   `get_luminosity_distance_Mpc`
-   `get_comoving_distance_Mpc`
-   `get_Hz_per_Mpc`
-   `get_sound_horizon_rs_Mpc`

#### Optional Performance Function

To accelerate fitting, you can provide an OpenCL version of the distance modulus function. The engine will automatically detect it and, if the user agrees, use it for fitting.
-   **`distance_modulus_model_opencl`**: This function must accept `z_array`, `*cosmo_params`, and the keyword arguments `cl_context` and `cl_queue`.

---

## 5. Future Expansion

-   **OpenCL Kernel Implementation**: The foundational support for OpenCL is now integrated. The next major architectural goal is to replace the placeholder functions in the model plugins with real, hardware-specific OpenCL kernels. This will require replacing SciPy's CPU-bound solvers (like `quad` and `brentq`) with custom, parallel-aware functions written for a GPU architecture.
-   **MCMC Integration**: The `fit_sne_parameters` function in `cosmo_engine.py` could be refactored to support other optimization backends, such as MCMC samplers (e.g., `emcee`, `dynesty`), for more robust parameter space exploration.