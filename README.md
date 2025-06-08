# Copernican Suite - A Modular Cosmology Framework

**Version:** 1.1b (Development)
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
    -   [Architectural Approach: Real-Time Kernel Generation](#architectural-approach-real-time-kernel-generation)
    -   [Required Model File Modifications](#required-model-file-modifications)
    -   [Example Implementation: ΛCDM Model](#example-implementation-λcdm-model)

---

## 1. Project Overview

The **Copernican Suite** is a Python-based framework for the modular testing and comparison of cosmological models against observational data. It provides a platform for researchers to easily implement and evaluate new theories alongside the standard ΛCDM model.

This version introduces a user-selectable OpenCL backend for GPU acceleration, allowing for high-performance fitting of complex models. To ensure a smooth user experience, the suite now includes a dependency and system sanity checker that runs on startup, providing platform-specific installation instructions for any missing components.

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

-   **`copernican.py`**: The main orchestrator script. It now includes a dependency checker that runs on startup to validate the environment.
-   **`data_loaders.py`**: Manages the loading and parsing of datasets.
-   **`cosmo_engine.py`**: Contains the core physics, statistics, and fitting logic, including the selectable compute backend (CPU/GPU) and the real-time OpenCL kernel compiler.
-   **`output_manager.py`**: Handles all forms of output (logging, plots, CSVs).
-   **Model Plugins (`*.py`) & Definitions (`*.md`)**: A two-file system for each model (e.g., `lcdm_model.py` and `lcdm_model.md`).

---

## 4. Workflow Overview

1.  **Dependency Check**: `copernican.py` first verifies that all required Python libraries and system drivers are available. It will exit with instructions if the environment is not set up correctly.
2.  **Initialization**: The script starts and creates the `./output/` directory for all results.
3.  **Backend Selection**: If OpenCL hardware is detected, the user is prompted to choose the compute backend (Standard CPU or OpenCL GPU).
4.  **Configuration**: The user specifies the file paths for the model and data files.
5.  **SNe Ia Fitting**: The `cosmo_engine` fits the parameters of both the ΛCDM model and the alternative model to the SNe Ia data. If the OpenCL backend is chosen, the engine compiles the kernel from the model plugin in real-time and executes the fitting on the GPU.
6.  **BAO Analysis**: Using the best-fit parameters, the engine calculates BAO observables for each model using CPU-based functions.
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
-   **`.py` Template:** To create a new model implementation file, you can use the commented-out template located at the **end of the `lcdm_model.py` file**. This template includes the structure for both standard CPU and high-performance OpenCL implementations.

---

## 6. High-Performance Computing with OpenCL

The framework includes a functional OpenCL implementation for hardware-accelerated calculations.

> **Numerical Precision of OpenCL Kernels**
>
> As of the latest update, the OpenCL kernels provided in the example model plugins (`lcdm_model.py`, `usmf2.py`) have been significantly upgraded. They now use high-precision numerical methods (specifically, **40-point Gauss-Legendre quadrature** for integration) designed to produce results that are consistent with the high-accuracy `SciPy` (CPU) backend. The previous issue of numerical divergence due to simplified algorithms has been resolved.

### Architectural Approach: Real-Time Kernel Generation

To maintain the self-contained, two-file structure for each model, OpenCL kernels are **not** stored in separate `.cl` files. Instead, they are defined as Python f-strings within each model's `.py` plugin, typically in a variable named `OPENCL_KERNEL_SRC`.

The `cosmo_engine.py` script automatically handles the compilation of this kernel string at runtime. This allows for dynamic kernel generation and keeps all logic for a given model within its dedicated files.

### Required Model File Modifications

To enable OpenCL acceleration for a new model, the developer must make two primary additions to the model's `.py` plugin file.

1.  **Define the Kernel Source String (`OPENCL_KERNEL_SRC`):** Create a multi-line Python string containing the OpenCL C99 kernel code. For best results, use a high-order fixed quadrature method like the 40-point Gauss-Legendre rule for any integrations.
2.  **Implement the `distance_modulus_model_opencl` Function:** This function serves as the entry point for the OpenCL calculation. It will receive the pre-compiled `cl_program` object from the `cosmo_engine`. Its responsibilities are to manage memory buffers, execute the kernel, retrieve the results, and return the final distance modulus `mu`.

### Example Implementation: ΛCDM Model

The following is the actual implementation from `lcdm_model.py`, demonstrating how a standard model can be accelerated with OpenCL.

#### 1. The OpenCL Kernel String

This f-string, defined in `lcdm_model.py`, contains a kernel that performs numerical integration of `c/H(z)` using a 40-point Gauss-Legendre quadrature to find the luminosity distance.

```python
# Defined in lcdm_model.py
OPENCL_KERNEL_SRC = f\"\"\"
// --- Gauss-Legendre Quadrature Constants (40-point) ---
__constant double GL_NODES_40[40] = {{ ... }};
__constant double GL_WEIGHTS_40[40] = {{ ... }};

// Helper function for the integrand c/H(z)
inline double integrand_func(...) {{ ... }}

__kernel void lcdm_dl_integrator(
    __global const double *z_values,
    __global double *dl_out,
    // ... Cosmological Parameters ...
) {{
    int gid = get_global_id(0);
    double z_upper = z_values[gid];
    
    // ... C99 code for 40-point Gauss-Legendre integration ...
    // to calculate comoving distance (dc_integral)
    
    // Final luminosity distance calculation
    dl_out[gid] = dc_integral * (1.0 + z_upper);
}}
\"\"\"