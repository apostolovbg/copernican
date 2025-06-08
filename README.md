# Copernican Suite - A Modular Cosmology Framework

**Version:** 1.1 (Stable)
**Last Updated:** 2025-06-08

> **Note:** This file serves as the complete and unified documentation for the Copernican Suite project. It contains all necessary information for users and developers, including architectural decisions and future development pathways.

---

## Table of Contents

1.  [**Project Overview**](#1-project-overview)
2.  [**Installation and Requirements**](#2-installation-and-requirements)
3.  [**Architecture**](#3-architecture)
4.  [**Workflow Overview**](#4-workflow-overview)
5.  [**Plugin Development Guide**](#5-plugin-development-guide)
    -   [The Two-File System](#the-two-file-system)
    -   [Finding the Templates](#finding-the-templates)
6.  [**High-Performance Computing with OpenCL**](#6-high-performance-computing-with-opencl)
    -   [Architectural Approaches: Direct vs. Hybrid](#architectural-approaches-direct-vs-hybrid)
    -   [Best Practice: The Hybrid CPU/GPU Model](#best-practice-the-hybrid-cpugpu-model)
    -   [A Note on Final Fit Results](#a-note-on-final-fit-results)
    -   [OpenCL Implementation Guide](#opencl-implementation-guide)

---

## 1. Project Overview

The **Copernican Suite** is a Python-based framework for the modular testing and comparison of cosmological models against observational data. It provides a platform for researchers to easily implement and evaluate new theories alongside the standard ΛCDM model.

The suite has now reached a stable release, capable of producing reliable, high-performance fits for complex, non-standard cosmological models. It features a user-selectable OpenCL backend for GPU acceleration and includes a dependency and system sanity checker to ensure a smooth user experience.

---

## 2. Installation and Requirements

The suite requires Python 3.x and several common scientific libraries. To simplify setup, an automatic dependency checker has been integrated directly into the main script.

When you first run `copernican.py`, it will:
1.  Verify that required Python libraries (`numpy`, `scipy`, `matplotlib`) are installed.
2.  Check for the optional `pyopencl` library, which is necessary for GPU acceleration.
3.  If `pyopencl` is found, it performs a sanity check to ensure it can communicate with system-level drivers and find a compute device.

If any of these checks fail, the script will print a detailed report of the missing components along with the correct commands to install them on your specific operating system (macOS, Linux, or Windows) before exiting.

---

## 3. Architecture

The suite is designed with a primary project directory containing all core scripts and model plugins. All outputs (logs, plots, CSVs) are saved into a dedicated `output` subdirectory.

-   **`copernican.py`**: The main orchestrator script.
-   **`data_loaders.py`**: Manages the loading and parsing of datasets.
-   **`cosmo_engine.py`**: Contains the core physics, statistics, and fitting logic, including the selectable compute backend (CPU/GPU) and the real-time OpenCL kernel compiler.
-   **`output_manager.py`**: Handles all forms of output (logging, plots, CSVs).
-   **Model Plugins (`*.py`) & Definitions (`*.md`)**: A two-file system for each model (e.g., `lcdm_model.py` and `lcdm_model.md`).

---

## 4. Workflow Overview

1.  **Dependency Check**: `copernican.py` first verifies that all required Python libraries and system drivers are available.
2.  **Initialization**: The script starts and creates the `./output/` directory for all results.
3.  **Backend Selection**: If OpenCL hardware is detected, the user is prompted to choose the compute backend (Standard CPU or OpenCL GPU).
4.  **Configuration**: The user specifies the file paths for the model and data files.
5.  **SNe Ia Fitting**: The `cosmo_engine` fits the parameters of both the ΛCDM model and the alternative model to the SNe Ia data.
6.  **BAO Analysis**: Using the best-fit parameters, the engine calculates BAO observables for each model.
7.  **Output Generation**: The `output_manager` saves all comparative plots and data summaries.

---

## 5. Plugin Development Guide

### The Two-File System

A model is defined by a two-file system:
1.  **Model Definition File (`.md`)**: A Markdown file containing the model's theory and a machine-parsable table of its parameters.
2.  **Model Implementation File (`.py`)**: A Python file containing the functions that perform the cosmological calculations, including the optional OpenCL kernel source.

Each new model requires two files with matching base names (e.g., `my_theory.md` and `my_theory.py`).

### Finding the Templates

To ensure consistency, templates for new model files are integrated directly into the base `lcdm_model` files.

-   **`.md` Template:** To create a new model definition file, copy `lcdm_model.md`, rename it, and edit the content. A generic template is also provided in a blockquote at the **end of the `lcdm_model.md` file**.
-   **`.py` Template:** To create a new model implementation file, use the heavily commented template located at the **end of the `lcdm_model.py` file**. This template demonstrates the recommended best practice for creating a reliable, high-performance OpenCL implementation.

---

## 6. High-Performance Computing with OpenCL

The framework's OpenCL implementation is crucial for fitting complex models. To ensure both speed and reliability, developers should understand the two available architectural approaches.

### Architectural Approaches: Direct vs. Hybrid

1.  **Direct GPU Approach (For Simple Models)**
    -   **Description:** The entire numerical calculation is performed within the OpenCL kernel on the GPU.
    -   **Use Case:** Ideal for models with mathematically stable equations, such as **ΛCDM**. The `lcdm_model.py` plugin uses this approach.
    -   **Advantage:** Maximum performance.

2.  **Hybrid CPU/GPU Approach (For Complex Models)**
    -   **Description:** The calculation is split. Numerically sensitive steps (e.g., root-finding) are performed on the **CPU** using reliable `SciPy` functions. The stable, pre-calculated results are then sent to the **GPU** for the final, parallelizable number-crunching.
    -   **Use Case:** Essential for complex models like **USMF_V2**, which involves a sensitive root-finding step.
    -   **Advantage:** Maximum reliability.

### Best Practice: The Hybrid CPU/GPU Model

For any new, non-trivial cosmological model, the **Hybrid CPU/GPU approach is the recommended best practice**. This architecture guarantees that the accelerated OpenCL mode will produce reliable and repeatable results consistent with the benchmark CPU implementation.

### A Note on Final Fit Results

With the reliable hybrid architecture, users should note that results from the CPU-only and hybrid OpenCL backends may not be bit-for-bit identical. The optimization process explores a high-dimensional parameter space, and the subtle differences in the underlying numerical calculations can cause the optimizer to follow slightly different paths.

This can occasionally lead to the OpenCL backend finding a different, and potentially even better, local minimum in the chi-squared landscape. This is not a sign of instability, but rather a feature of robustly exploring complex models. For instance, in test runs, the hybrid OpenCL fit for the `USMF_V2` model yielded a better $\chi^2$ value for Supernovae Ia data than both its CPU-only counterpart and the standard ΛCDM model, showcasing the suite's power in comparative cosmology.

### OpenCL Implementation Guide

To enable OpenCL acceleration for a new model, the developer should implement the hybrid model as demonstrated in the template at the end of `lcdm_model.py`. The key steps are:

1.  **Implement the Standard CPU Functions:** Create the full, reliable CPU-based implementation first.
2.  **Isolate Sensitive Calculations:** Create helper functions that use SciPy (e.g., `brentq`, `solve_ivp`) to handle any unstable parts of the model.
3.  **Write a Simple OpenCL Kernel:** The kernel, defined in the `OPENCL_KERNEL_SRC` string, should accept pre-calculated, stable inputs from the CPU.
4.  **Implement the Hybrid Entry Point:** The `distance_modulus_model_opencl` function orchestrates the process: it calls the CPU helpers, transfers the safe inputs to the OpenCL kernel, executes it, and retrieves the results for any final calculations.