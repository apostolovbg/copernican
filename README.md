# Copernican Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The Copernican Suite is a Python-based framework designed for the modular testing and comparison of cosmological models against multiple observational datasets like Supernovae Ia (SNe Ia) and Baryon Acoustic Oscillations (BAO).

It allows cosmologists and researchers to quickly plug in new theories, fit them to data, and directly compare their performance against the standard LambdaCDM model using consistent statistical measures.

## Key Features

-   **Modular Model Plugins:** Easily add new cosmological models by creating a single Python file that adheres to a simple, standard interface.
-   **Consistent Workflow:** A core design principle is to use cosmological parameters derived from one observational probe (e.g., SNe Ia fits) when testing the same model against others (e.g., BAO).
-   **Versatile Data Loading:** Supports various data formats through a modular data loading system with interactive command-line menus.
-   **High-Quality Outputs:** Automatically generates detailed logs, summary CSV files, and publication-ready plots for both Hubble diagrams and BAO observables. The plots include binned residual averages to help visualize systematic trends.
-   **Clean Workspace:** All generated outputs are saved to a dedicated `output/` subdirectory to keep the main project folder clean.

## Architecture Overview

The suite is composed of several core modules:
-   `copernican.py`: The main orchestrator that handles user interaction and manages the workflow.
-   `cosmo_engine.py`: Contains the core physics and statistical logic, including the $\chi^2$ calculators and the parameter fitting engine.
-   `data_loaders.py`: Manages the loading and parsing of all observational datasets.
-   `output_manager.py`: Handles all outputs, including logging, plotting, and CSV generation.
-   `*_model.py`: Individual model plugins (e.g., `lcdm_model.py`, `usmf2.py`) that define specific cosmological models.

## Installation

The Copernican Suite is designed to be run from a local folder and requires Python 3 and a few common scientific libraries.

1.  **Prerequisites:**
    -   Python 3.x

2.  **Download:**
    -   Clone this repository or download the files as a ZIP and extract them to a folder on your computer.

3.  **Install Libraries:**
    -   Open your terminal or command prompt and install the required libraries using pip:
        ```sh
        pip install numpy scipy matplotlib
        ```

## How to Run

The suite includes easy-to-use launchers for different operating systems. Place these in the same folder as `copernican.py`.

#### For macOS / Linux

1.  **Make the script executable (one-time setup):**
    Open a terminal, navigate to the project folder, and run:
    ```sh
    chmod +x start.command
    ```
2.  **Run:** Double-click the `start.command` file. A new terminal window will open and run the script.

#### For Windows

1.  **Run:** Simply double-click the `run.bat` file. A Command Prompt window will open and run the script, pausing at the end to show the output.

#### Direct Execution (All Platforms)

You can also run the suite directly from an open terminal:
```sh
python3 copernican.py
```

## Adding a New Cosmological Model

One of the suite's primary features is the ability to easily test new models. To add your own theory:

1.  **Create a Model File:** Create a new Python file in the main project directory (e.g., `my_theory.py`).
2.  **Implement the Interface:** Your model file must contain a standard set of global variables and functions that the `cosmo_engine` expects.
    -   **Required Variables:** `MODEL_NAME`, `PARAMETER_NAMES`, `INITIAL_GUESSES`, `PARAMETER_BOUNDS`.
    -   **Required Functions:** `distance_modulus_model()`, `get_Hz_per_Mpc()`, `get_sound_horizon_rs_Mpc()`, and other distance functions as specified in `model.json`.
3.  **Run the Suite:** When you run `copernican.py`, simply enter the filename of your new model (`my_theory.py`) when prompted. The suite will load it dynamically and test it against ΛCDM.

## Future Development

The current version is a stable, CPU-based framework. The next major architectural goal is to develop a fork of the suite that leverages GPU acceleration.

-   **Goal:** To significantly speed up the computationally intensive fitting process for complex models like USMF.
-   **Technology:** This will involve a major redesign to use GPU-compatible frameworks like PyTorch, JAX, or Numba, with a focus on OpenCL/ROCm for cross-vendor (AMD/NVIDIA) support.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.