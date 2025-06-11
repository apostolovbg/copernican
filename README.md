# Copernican Suite - A Modular Cosmology Framework

**Version:** 1.4b (Unstable)
**Last Updated:** 2025-06-12

> **Note:** This file serves as the complete and unified documentation for the Copernican Suite project. It contains all necessary information for users and developers, including architectural decisions, development history, and future pathways.

---

## Table of Contents

1.  [**Project Overview**](#1-project-overview)
2.  [**Installation and Requirements**](#2-installation-and-requirements)
3.  [**Current Architecture (v1.4b)**](#3-current-architecture-v14b)
4.  [**Workflow Overview**](#4-workflow-overview)
5.  [**Development History & Roadmap**](#5-development-history--roadmap)
    -   [Version 1.4b Updates (Plotting Restoration - IN PROGRESS)](#version-14b-updates-plotting-restoration---in-progress)
    -   [Version 1.4a Updates (The Great Refactor)](#version-14a-updates-the-great-refactor)
    -   [Version 1.3 Updates (Stable Release)](#version-13-updates-stable-release)
    -   [The Future Vision: A Universal Engine](#the-future-vision-a-universal-engine)
6.  [**A Note on AI-Driven Development**](#6-a-note-on-ai-driven-development)

---

## 1. Project Overview

The **Copernican Suite** is an evolving Python-based framework for the modular testing and comparison of cosmological models against observational data. It provides a platform for researchers to easily implement and evaluate new theories alongside the standard ΛCDM model.

This version (`1.4b`) is an unstable development release focused on restoring the high-quality plotting functionality from v1.3 into the new, more robust engine-based architecture introduced in v1.4a.

---

## 2. Installation and Requirements

The suite requires Python 3.x and several scientific libraries. To simplify setup, an automatic dependency checker runs when the program starts.

When you first run `copernican.py`, it will verify that the following required Python libraries are installed:
-   `numpy`
-   `scipy`
-   `matplotlib`
-   `pandas`
-   `psutil` (for detecting CPU cores, though no longer used by default plugins)

If any of these are missing, the script will print the necessary `pip install` commands before exiting.

---

## 3. Current Architecture (v1.4b)

The suite is now built on a **fully modular, decoupled output pipeline**. The `v1.4a` refactor isolated the computational engine; the `v1.4b` refactor isolates each component of the output stage.

-   **`copernican.py`**: The main orchestrator script. Manages user interaction and high-level workflow.
-   **`input_aggregator.py`**: The "Assembler." Gathers all model and data info and serializes it into the "Job JSON" for the engine.
-   **`cosmo_engine_*.py`**: A "Black Box" computational engine that accepts a "Job JSON" and returns a "Results JSON".
-   **`output_manager.py`**: The "Dispatcher." This module's *sole responsibility* is to accept the final "Results JSON" and delegate output tasks to the appropriate specialist modules. It contains no plotting or file-writing logic itself.
-   **`plotter.py`**: The "Plotting Specialist." A new module that contains all `matplotlib` logic for generating and saving plots in the required v1.3 style. It is called by the `output_manager`.
-   **`csv_writer.py`**: The "CSV Specialist." A new module containing all logic for writing detailed data files. It is called by the `output_manager`.
-   **`data_loaders.py`**: A modular system for parsing different SNe Ia and BAO data formats.
-   **Model Plugins (`*.py`) & Definitions (`*.md`)**: The two-file system for defining new cosmological models.

---

## 4. Workflow Overview

1.  **Dependency Check**: `copernican.py` first verifies all required Python libraries.
2.  **Engine Selection & Configuration**: The user selects an engine and provides paths for the model and data files.
3.  **Job Aggregation**: `input_aggregator.py` builds the complete "Job JSON".
4.  **Engine Execution**: The chosen `cosmo_engine` runs in isolation and returns a "Results JSON".
5.  **Output Dispatch**: `copernican.py` passes the "Results JSON" to `output_manager.py`.
6.  **Specialist Output Generation**: The `output_manager` calls `plotter.py` to create the plots and `csv_writer.py` to create the data files.
7.  **Loop or Exit**: The user is prompted to run another evaluation or exit.

---

## 5. Development History & Roadmap

### Version 1.4b Updates (Plotting Restoration - IN PROGRESS)

This version represents a concerted effort to restore the high-quality plotting of v1.3 within the new modular architecture, and to formalize the output pipeline for future stability.

-   **Goal**: Replicate the exact style and informational content of the v1.3 SNe Ia and BAO plots.
-   **Architectural Change**: The monolithic `output_manager.py` from v1.4a was refactored into three separate, single-responsibility modules:
    -   `output_manager.py`: Now a pure **Dispatcher** that coordinates output tasks.
    -   `plotter.py`: A new **Plotting Specialist** to handle all `matplotlib` code.
    -   `csv_writer.py`: A new **CSV Specialist** to handle all `.csv` file writing.
-   **Current Status (As of 2025-06-12):**
    -   ✅ **WORKING**: The SNe Ia Hubble Diagram is generated correctly, matching the v1.3 style with a main plot, residuals, binned averages, and detailed information boxes.
    -   ✅ **WORKING**: Detailed, point-by-point comparative CSV files for both SNe Ia and BAO analyses are generated correctly.
    -   ❌ **CRITICAL BUG**: There is a persistent bug preventing the theoretical model lines (for both ΛCDM and the alternative model) from being rendered on the BAO observables plot. The data points and info boxes are drawn correctly, but the smooth curves are missing. This is the primary blocker for a stable release.

### Version 1.4a Updates (The Great Refactor)

This version represented the most significant architectural evolution of the project, achieving the long-term goal of a truly modular system.
-   **Decoupled Engine Architecture:** The core logic was refactored into a "black box" `cosmo_engine`.
-   **Swappable Engines:** The main script now discovers and allows the user to select any available `cosmo_engine_*.py` file.

### Version 1.3 Updates (Stable Release)

This version addressed critical bugs and added significant enhancements to the data output, plot clarity, and developer framework.
-   **CRITICAL BUG FIX - BAO Plotting Restored:** Fixed a bug that caused smooth model lines on BAO plots to fail to render. (Note: This bug, while similar, had a different cause than the current v1.4b bug).

### The Future Vision: A Universal Engine

The `v1.4` architecture is the final stepping stone to this vision. With a fully decoupled pipeline, the focus can now shift to developing the declarative engine.

-   **Deprecation of `.py` Plugins:** The long-term goal is to eliminate the need for custom `model_plugin.py` files.
-   **Universal Math Engine (v1.5+):** A new engine (`cosmo_engine_v1.5_declarative.py`) will be developed that can directly parse, interpret, and solve the mathematical equations provided in the `.md` definition files using symbolic math libraries like `sympy`.

---

## 6. A Note on AI-Driven Development

> **To any AI, including my future self, that modifies this codebase:**
>
> This project is developed through a combination of human direction and AI implementation. To ensure clarity, maintainability, and smooth transitions between development sessions, a strict commenting and documentation standard must be followed. The `doc.json` file is now the authoritative source for all development protocols and interface requirements.
>
> **When modifying any file, you are required to:**
> 1.  **Add a `DEV NOTE` at the top of the file.** This note should summarize the changes made in the current version.
> 2.  **Comment the code extensively.** Explain the "why" behind your code, not just the "what".
> 3.  **Update this README file and `doc.json`**. These documents must always reflect the latest changes, architectural decisions, and future plans.
>
> Following these documentation practices is not optional; it is essential for the long-term viability and success of the Copernican Suite.