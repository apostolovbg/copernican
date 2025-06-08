# Copernican Suite - A Modular Cosmology Framework

**Version:** 1.1b (Development)
**Last Updated:** 2025-06-08

> **Note:** This file serves as the complete and unified documentation for the Copernican Suite project. It contains all necessary information for users and developers, including architectural decisions and future development pathways.

---

## Table of Contents

1.  [**Project Overview**](#1-project-overview)
2.  [**Architecture**](#2-architecture)
3.  [**Workflow Overview**](#3-workflow-overview)
4.  [**Plugin Development Guide**](#4-plugin-development-guide)
    -   [The Two-File System](#the-two-file-system)
    -   [Finding the Templates](#finding-the-templates)
5.  [**Future Development: Implementing OpenCL Kernels**](#5-future-development-implementing-opencl-kernels)
    -   [Architectural Approach: Real-Time Kernel Generation](#architectural-approach-real-time-kernel-generation)
    -   [Required File Modifications](#required-file-modifications)
    -   [Example Implementation Sketch](#example-implementation-sketch)

---

## 1. Project Overview

The **Copernican Suite** is a Python-based framework for the modular testing and comparison of cosmological models against observational data. It provides a platform for researchers to easily implement and evaluate new theories alongside the standard ΛCDM model.

This development version introduces a user-selectable OpenCL backend for GPU acceleration, allowing for high-performance fitting of complex models. The core dispatch logic is in place, while the GPU-specific model calculations are currently placeholders, paving the way for full implementation.

---

## 2. Architecture

The suite is designed with a primary project directory containing all core scripts and model plugins. All outputs (logs, plots, CSVs) are saved into a dedicated `output` subdirectory.

-   **`copernican.py`**: The main orchestrator script.
-   **`data_loaders.py`**: Manages the loading and parsing of datasets.
-   **`cosmo_engine.py`**: Contains the core physics, statistics, and fitting logic, including the selectable compute backend (CPU/GPU).
-   **`output_manager.py`**: Handles all forms of output (logging, plots, CSVs).
-   **Model Plugins (`*.py`) & Definitions (`*.md`)**: A two-file system for each model (e.g., `lcdm_model.py` and `lcdm_model.md`).

---

## 3. Workflow Overview

1.  **Initialization:** `copernican.py` starts and creates the `./output/` directory.
2.  **Backend Selection:** If OpenCL hardware is detected, the user is prompted to choose the compute backend (Standard CPU or OpenCL GPU).
3.  **Configuration:** The user specifies the file paths for the model and data files.
4.  **SNe Ia Fitting:** The `cosmo_engine` fits the parameters of both the ΛCDM model and the alternative model to the SNe Ia data using the chosen computational backend.
5.  **BAO Analysis:** Using the best-fit parameters, the engine calculates BAO observables for each model.
6.  **Output Generation:** The `output_manager` saves all comparative plots and data summaries.

---

## 4. Plugin Development Guide

### The Two-File System

A model is defined by a two-file system:
1.  **Model Definition File (`.md`)**: A Markdown file containing the model's theory and a machine-parsable table of its parameters.
2.  **Model Implementation File (`.py`)**: A Python file containing the functions that perform the cosmological calculations.

Each new model requires two files with matching base names (e.g., `my_theory.md` and `my_theory.py`).

### Finding the Templates

To ensure consistency, templates for new model files are integrated directly into the base `lcdm_model` files.

-   **`.md` Template:** To create a new model definition file, copy `lcdm_model.md`, rename it, and edit the content. A generic template is also provided in a blockquote at the **end of the `lcdm_model.md` file**.
-   **`.py` Template:** To create a new model implementation file, you can use the commented-out template located at the **end of the `lcdm_model.py` file**. Copy this template into a new file and fill in your model's specific logic.

---

## 5. Future Development: Implementing OpenCL Kernels

The current framework uses placeholder functions for OpenCL calculations. This section outlines the intended path for implementing real, hardware-accelerated kernels.

### Architectural Approach: Real-Time Kernel Generation

To maintain the self-contained, two-file structure for each model, OpenCL kernels will **not** be stored in separate `.cl` files. Instead, they will be defined as Python f-strings directly within the `distance_modulus_model_opencl` function of each model's `.py` plugin. This allows for dynamic kernel generation and compilation at runtime and keeps all logic for a given model within its dedicated files.

### Required File Modifications

#### 1. Model Plugin Files (e.g., `lcdm_model.py`, `usmf2.py`)

The primary work will be done here by replacing the placeholder content in the `distance_modulus_model_opencl` function. A full implementation will require the following steps within this function:

1.  **Define the Kernel String:** Create a multi-line Python f-string containing the OpenCL C99 kernel code. The cosmological parameters (`*cosmo_params`) can be formatted into this string, making the kernel specific to the parameters being tested in a given evaluation.
2.  **Compile the Kernel:** Use the passed `cl_context` object to compile the kernel string in real-time: `program = cl.Program(cl_context, kernel_string).build()`.
3.  **Manage Memory Buffers:**
    -   Create an input buffer for `z_array` and transfer the data from the host (CPU) to the compute device (GPU).
    -   Create an output buffer on the device to store the results.
    -   Use `cl.Buffer(cl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=z_data)` for inputs and `cl.Buffer(cl_context, cl.mem_flags.WRITE_ONLY, size=output_size_bytes)` for outputs.
4.  **Execute the Kernel:** Call the compiled kernel function, passing the `cl_queue`, global work size (typically `z_array.shape`), local work size, and the memory buffers as arguments.
5.  **Retrieve Results:** Copy the data from the output buffer on the device back to the host CPU using `cl.enqueue_copy(cl_queue, host_array, device_buffer)`.

> **Important Note on Complex Models:** For models like `usmf2.py` that rely on SciPy's `quad` (integration) or `brentq` (root-finding), these functions cannot be used in a GPU kernel. The developer will need to implement custom, parallel-aware numerical algorithms (e.g., parallel reduction for integration) directly within the OpenCL kernel code.

#### 2. `cosmo_engine.py`

This file should require minimal changes. The dispatch logic is already in place. Future modifications may include adding more robust error handling to catch and log GPU-specific issues, such as kernel compilation failures or memory errors.

### Example Implementation Sketch

The following is a conceptual sketch of what a real `distance_modulus_model_opencl` function might look like inside `lcdm_model.py`.

```python
# This is a conceptual example for documentation purposes.

def distance_modulus_model_opencl(z_array, H0, Omega_m0, Omega_b0, cl_context=None, cl_queue=None):
    # 1. Define Kernel as an f-string, inserting current parameters
    # This example implements a simple numerical integration (trapezoidal rule)
    # A real implementation would use a more accurate method (e.g., Simpson's rule).
    
    h, Omega_r0, Omega_L0, _, _ = _get_derived_densities(H0, Omega_m0, Omega_b0)
    C_LIGHT_KM_S = FIXED_PARAMS["C_LIGHT_KM_S"]
    
    # Check for invalid parameters before attempting to build the kernel
    if any(np.isnan([h, Omega_r0, Omega_L0])):
        return np.full_like(z_array, np.nan)

    kernel_string = f"""
    __kernel void integrate_dl(__global const double *z_values, __global double *dl_out) {{
        int gid = get_global_id(0);
        double z = z_values[gid];
        
        if (z < 1e-9) {{
            dl_out[gid] = 0.0;
            return;
        }}

        // Simple integration loop (Trapezoidal Rule)
        int N_STEPS = 1000; // Number of integration steps
        double step_size = z / N_STEPS;
        double dc_integral = 0.0;

        for (int i = 0; i < N_STEPS; i++) {{
            double z_i = i * step_size;
            double z_i1 = (i + 1) * step_size;
            
            // H(z) calculation at step i and i+1
            double Ez_sq_i = {Omega_r0}*pow(1+z_i, 4) + {Omega_m0}*pow(1+z_i, 3) + {Omega_L0};
            double hz_i = {H0} * sqrt(Ez_sq_i);
            
            double Ez_sq_i1 = {Omega_r0}*pow(1+z_i1, 4) + {Omega_m0}*pow(1+z_i1, 3) + {Omega_L0};
            double hz_i1 = {H0} * sqrt(Ez_sq_i1);

            // Integrand c/H(z)
            double integrand_i = {C_LIGHT_KM_S} / hz_i;
            double integrand_i1 = {C_LIGHT_KM_S} / hz_i1;
            
            dc_integral += 0.5 * (integrand_i + integrand_i1) * step_size;
        }}

        // Final luminosity distance calculation
        dl_out[gid] = dc_integral * (1.0 + z);
    }}
    """
    
    # 2. Compile Kernel
    try:
        program = cl.Program(cl_context, kernel_string).build()
    except cl.RuntimeError as e:
        logger = logging.getLogger()
        logger.error(f"OpenCL kernel compilation failed: {e}")
        return np.full_like(z_array, np.nan)

    # 3. Manage Buffers
    z_buffer = cl.Buffer(cl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=z_array.astype(np.float64))
    dl_buffer = cl.Buffer(cl_context, cl.mem_flags.WRITE_ONLY, size=z_array.nbytes)
    
    # 4. Execute Kernel
    program.integrate_dl(cl_queue, z_array.shape, None, z_buffer, dl_buffer)

    # 5. Retrieve Results
    dl_mpc = np.empty_like(z_array, dtype=np.float64)
    cl.enqueue_copy(cl_queue, dl_mpc, dl_buffer).wait()
    
    # Convert to distance modulus
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = 5.0 * np.log10(dl_mpc) + 25.0
    mu[dl_mpc <= 0] = np.nan
    
    return mu