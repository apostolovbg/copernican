# Copernican Suite - A Modular Cosmology Framework

**Version:** 1.3 (Stable)
**Last Updated:** 2025-06-11

> **Note:** This file serves as the complete and unified documentation for the Copernican Suite project. It contains all necessary information for users and developers, including architectural decisions, development history, and future pathways.

---

## Table of Contents

1.  [**Project Overview**](#1-project-overview)
2.  [**Installation and Requirements**](#2-installation-and-requirements)
3.  [**Current Architecture (v1.3)**](#3-current-architecture-v13)
4.  [**Workflow Overview**](#4-workflow-overview)
5.  [**Development History & Roadmap**](#5-development-history--roadmap)
    -   [Version 1.3 Updates (Stable Release)](#version-13-updates-stable-release)
    -   [Version 1.2 Updates (Major Refactor)](#version-12-updates-major-refactor)
    -   [The Future Vision: A Universal Engine](#the-future-vision-a-universal-engine)
6.  [**A Note on AI-Driven Development**](#6-a-note-on-ai-driven-development)

---

## 1. Project Overview

The **Copernican Suite** is an evolving Python-based framework for the modular testing and comparison of cosmological models against observational data. It provides a platform for researchers to easily implement and evaluate new theories alongside the standard ΛCDM model.

This version (1.3) resolves a critical plotting bug, enhances the data output capabilities for more detailed analysis, and streamlines the user experience.

---

## 2. Installation and Requirements

The suite requires Python 3.x and several scientific libraries. To simplify setup, an automatic dependency checker runs when the program starts.

When you first run `copernican.py`, it will verify that the following required Python libraries are installed:
-   `numpy`
-   `scipy`
-   `matplotlib`
-   `psutil` (for detecting CPU cores)

If any of these are missing, the script will print the necessary `pip install` commands before exiting.

---

## 3. Current Architecture (v1.3)

The suite is designed with a primary project directory containing all core scripts and model plugins. All outputs (logs, plots, CSVs) are saved into a dedicated `output` subdirectory.

-   **`copernican.py`**: The main orchestrator script, which now includes a "run again" loop and cache cleanup.
-   **`data_loaders.py`**: Manages the loading and parsing of datasets in a fully modular way.
-   **`cosmo_engine.py`**: Contains the core physics, statistics, and fitting logic using the SciPy library.
-   **`output_manager.py`**: Handles all forms of output, now providing streamlined, detailed CSV files for both SNe Ia and BAO analysis.
-   **Model Plugins (`*.py`) & Definitions (`*.md`)**: In the current version, a model is defined by a two-file system. The `.md` file holds the theory and parameters, while the `.py` file contains the Python functions for the calculations.

---

## 4. Workflow Overview

1.  **Dependency Check**: `copernican.py` first verifies that all required Python libraries are available.
2.  **Initialization**: The script starts and creates the `./output/` directory for all results.
3.  **Configuration**: The user specifies the file paths for the model and data files.
    -   **Test Mode**: A user can now enter `test` at the alternative model prompt to run ΛCDM against itself, providing a quick way to test the full analysis pipeline.
4.  **SNe Ia Fitting**: The `cosmo_engine` fits the parameters of both the ΛCDM model and the alternative model to the SNe Ia data.
5.  **BAO Analysis**: Using the best-fit parameters, the engine calculates BAO observables for each model.
6.  **Output Generation**: The `output_manager` saves all comparative plots and detailed, point-by-point data summaries for both SNe and BAO analyses.
7.  **Loop or Exit**: The user is prompted to either run another evaluation or exit the program. On each loop or on exit, temporary cache files are automatically cleaned up.

---

## 5. Development History & Roadmap

### Version 1.3 Updates (Stable Release)

This version addresses the primary issues identified in v1.2 and adds significant enhancements to the data output and plot clarity.

-   **CRITICAL BUG FIX - BAO Plotting Restored:** A bug that caused the smooth model lines on BAO plots to fail to render has been fixed. The issue was traced to a faulty multiprocessing implementation in `lcdm_model.py`, which was producing `NaN` arrays during the calculation of smooth curves. The function has been repaired, and BAO plots now render correctly for all models.
-   **Enhanced Plot Clarity:** The BAO plot now renders the alternative model's lines with 25% opacity. This prevents the alternative model from completely obscuring the underlying ΛCDM model in cases of a close fit, allowing for better visual comparison.
-   **Streamlined Data Outputs:** The CSV output has been simplified to produce only detailed, point-by-point data files for both SNe and BAO analyses, removing redundant summary files.

### Version 1.2 Updates (Major Refactor)

Version 1.2 marked a major step towards stability and a more focused architecture. Key changes from v1.1 include:
-   **CPU-Centric Engine:** All OpenCL GPU-acceleration code has been removed to eliminate instability and complex dependencies, favoring a more predictable and robust pure-Python/SciPy engine.
-   **Intelligent Multiprocessing:** A robust, batch-based multiprocessing system has been implemented. It uses the `psutil` library to detect the number of **physical** CPU cores and distributes workloads efficiently for a significant performance increase.
-   **Bug Fixes & Stability:**
    -   Resolved critical `PicklingError` and `NameError` bugs that prevented the multiprocessing and main script from running.
    -   Systematically fixed all `SyntaxWarning` issues related to LaTeX string formatting for clean, error-free execution.
-   **Workflow Enhancements:**
    -   **Test Sequence:** Added the `test` mode to allow for rapid verification of the program's fitting and plotting pipeline.
    -   **Run Again & Cleanup:** The main script now supports running multiple evaluations in a single session and automatically cleans up temporary cache files upon completion.
    -   **Harmonized Filenames:** All output files (plots, CSVs, logs) now follow a clean and consistent naming convention.

### The Future Vision: A Universal Engine

The next major architectural evolution will focus on making the suite even more accessible and powerful by removing the need for Python coding to define a new model.

-   **Deprecation of `.py` Plugins:** The long-term goal is to eliminate the need for custom `model_plugin.py` files altogether.
-   **Universal Math Engine:** A new, powerful engine will be developed within `cosmo_engine.py`. This engine will be capable of directly parsing, interpreting, and solving the mathematical equations provided in the `.md` definition files.
-   **Declarative Model Definition:** Researchers will only need to create a single, well-defined `.md` file. By specifying their model's key equations and parameters in the standard format, the universal engine will handle the rest, from fitting to plotting.
-   **Internalization of ΛCDM:** The logic for the standard ΛCDM model will be integrated directly into the core program. The `lcdm_model.md` file will remain as the primary template for new models and will serve as the "control sample" for the engine to run comparisons against.

This future direction will transform the Copernican Suite into a truly modular and user-friendly platform, allowing cosmologists to test new theories with unprecedented ease and speed.

---

## 6. A Note on AI-Driven Development

> **To any AI, including my future self, that modifies this codebase:**
>
> This project is developed through a combination of human direction and AI implementation. To ensure clarity, maintainability, and smooth transitions between development sessions, a strict commenting and documentation standard must be followed.
>
> **When modifying any file, you are required to:**
> 1.  **Add a `DEV NOTE` at the top of the file.** This note should summarize the changes made in the current version and provide guidance for future modifications.
> 2.  **Comment the code extensively.** Explain the purpose of new functions, the logic behind complex algorithms or bug fixes, and the flow of data. Your comments should be clear enough for another AI or a human developer to understand the "why" behind your code, not just the "what".
> 3.  **Update this README file.** The README is the single source of truth for the project's state. It must always reflect the latest changes, architectural decisions, and future plans.
>
> Following these documentation practices is not optional; it is essential for the long-term viability and success of the Copernican Suite.