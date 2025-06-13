# Copernican Suite - A Modular Cosmology Framework

**Version:** 1.4b
**Last Updated:** 2025-06-12

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

This version (1.3) resolves a critical plotting bug, enhances data outputs, and introduces a formal specification to streamline future development.

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

-   **`copernican.py`**: The main orchestrator script.
-   **`data_loaders.py`**: Manages the loading and parsing of datasets.
-   **`cosmo_engine.py`**: Contains the core physics, statistics, and fitting logic.
-   **`output_manager.py`**: Handles all forms of output (logging, plots, detailed CSVs).
-   **`doc.json`**: An AI/developer-focused specification file that defines the project architecture and the model plugin interface.
-   **Model Plugins (`*.py`) & Definitions (`*.md`)**: A two-file system for defining new cosmological models.

---

## 4. Workflow Overview

1.  **Dependency Check**: `copernican.py` first verifies all required Python libraries are available.
2.  **Initialization**: The script starts and creates the `./output/` directory for all results.
3.  **Configuration**: The user specifies the file paths for the model and data files.
    -   **Test Mode**: A user can enter `test` to run ΛCDM against itself, providing a quick way to test the full analysis pipeline.
4.  **SNe Ia Fitting**: The `cosmo_engine` fits the parameters of both the ΛCDM model and the alternative model to the SNe Ia data.
5.  **BAO Analysis**: Using the best-fit parameters, the engine calculates BAO observables for each model.
6.  **Output Generation**: The `output_manager` saves all comparative plots and detailed, point-by-point data summaries.
7.  **Loop or Exit**: The user is prompted to run another evaluation or exit.

---

## 5. Development History & Roadmap

See `CHANGELOG.md` for complete version history.

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