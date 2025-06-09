# Copernican Suite - A Modular Cosmology Framework

**Version:** 1.2b
**Last Updated:** 2025-06-09

> **Note:** This file serves as the complete and unified documentation for the Copernican Suite project. It contains all necessary information for users and developers.

---

## Table of Contents

1.  [**Project Overview**](#1-project-overview)
2.  [**Installation and Requirements**](#2-installation-and-requirements)
3.  [**Architecture**](#3-architecture)
4.  [**Workflow Overview**](#4-workflow-overview)
5.  [**Plugin Development Guide**](#5-plugin-development-guide)
    -   [The Two-File System](#the-two-file-system)
    -   [Finding the Templates](#finding-the-templates)

---

## 1. Project Overview

The **Copernican Suite** is a Python-based framework for the modular testing and comparison of cosmological models against observational data. It provides a platform for researchers to easily implement and evaluate new theories alongside the standard ΛCDM model.

The suite has now reached a stable release, capable of producing reliable fits for complex, non-standard cosmological models using its SciPy-based computational engine. It includes a dependency checker to ensure a smooth user experience.

---

## 2. Installation and Requirements

The suite requires Python 3.x and several common scientific libraries. To simplify setup, an automatic dependency checker has been integrated directly into the main script.

When you first run `copernican.py`, it will:
1.  Verify that required Python libraries (`numpy`, `scipy`, `matplotlib`) are installed.
2.  If any of these checks fail, the script will print a detailed report of the missing components along with the correct `pip install` commands before exiting.

---

## 3. Architecture

The suite is designed with a primary project directory containing all core scripts and model plugins. All outputs (logs, plots, CSVs) are saved into a dedicated `output` subdirectory.

-   **`copernican.py`**: The main orchestrator script.
-   **`data_loaders.py`**: Manages the loading and parsing of datasets.
-   **`cosmo_engine.py`**: Contains the core physics, statistics, and fitting logic using the SciPy library.
-   **`output_manager.py`**: Handles all forms of output (logging, plots, CSVs).
-   **Model Plugins (`*.py`) & Definitions (`*.md`)**: A two-file system for each model (e.g., `lcdm_model.py` and `lcdm_model.md`).

---

## 4. Workflow Overview

1.  **Dependency Check**: `copernican.py` first verifies that all required Python libraries are available.
2.  **Initialization**: The script starts and creates the `./output/` directory for all results.
3.  **Configuration**: The user specifies the file paths for the model and data files.
4.  **SNe Ia Fitting**: The `cosmo_engine` fits the parameters of both the ΛCDM model and the alternative model to the SNe Ia data.
5.  **BAO Analysis**: Using the best-fit parameters, the engine calculates BAO observables for each model.
6.  **Output Generation**: The `output_manager` saves all comparative plots and data summaries.

---

## 5. Plugin Development Guide

### The Two-File System

A model is defined by a two-file system:
1.  **Model Definition File (`.md`)**: A Markdown file containing the model's theory and a machine-parsable table of its parameters.
2.  **Model Implementation File (`.py`)**: A Python file containing the functions that perform the cosmological calculations using standard libraries like `numpy` and `scipy`.

Each new model requires two files with matching base names (e.g., `my_theory.md` and `my_theory.py`).

### Finding the Templates

To ensure consistency, templates for new model files are integrated directly into the base `lcdm_model` files.

-   **`.md` Template:** To create a new model definition file, copy `lcdm_model.md`, rename it, and edit the content. A generic template is also provided in a blockquote at the **end of the `lcdm_model.md` file**.
-   **`.py` Template:** To create a new model implementation file, use the heavily commented template located at the **end of the `lcdm_model.py` file**. This template demonstrates the best practice for creating a reliable, CPU-based model implementation.