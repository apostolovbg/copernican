# Copernican Suite - A Modular Cosmology Framework

**Version:** 1.4a (Stable)
**Last Updated:** 2025-06-11

> **Note:** This file serves as the complete and unified documentation for the Copernican Suite project. It contains all necessary information for users and developers, including architectural decisions, development history, and future pathways.

---

## Table of Contents

1.  [**Project Overview**](#1-project-overview)
2.  [**Installation and Requirements**](#2-installation-and-requirements)
3.  [**Current Architecture (v1.4a)**](#3-current-architecture-v14a)
4.  [**Workflow Overview**](#4-workflow-overview)
5.  [**Development History & Roadmap**](#5-development-history--roadmap)
    -   [Version 1.4a Updates (The Great Refactor)](#version-14a-updates-the-great-refactor)
    -   [Version 1.3 Updates (Stable Release)](#version-13-updates-stable-release)
    -   [The Future Vision: A Universal Engine](#the-future-vision-a-universal-engine)
6.  [**A Note on AI-Driven Development**](#6-a-note-on-ai-driven-development)

---

## 1. Project Overview

The **Copernican Suite** is an evolving Python-based framework for the modular testing and comparison of cosmological models against observational data. It provides a platform for researchers to easily implement and evaluate new theories alongside the standard ΛCDM model.

This version (1.4a) marks a major architectural refactor, transforming the suite into a fully decoupled, engine-based system. This design enhances stability, simplifies development, and paves the way for future high-performance and declarative backends.

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

## 3. Current Architecture (v1.4a)

The suite is designed around a **linear, decoupled pipeline** where each major component is a "black box" that communicates via a standardized JSON-based data contract (our "DSL").

-   **`copernican.py`**: The main orchestrator script. Its role is now focused purely on user interaction (getting file paths) and high-level workflow management. It discovers and allows the user to select a computational engine for the analysis.

-   **`input_aggregator.py`**: The "Assembler." This new module gathers all model and data information and serializes it into a single "Job JSON" string. This is the standardized "work order" for the engine.

-   **`cosmo_engine_*.py`**: A "Black Box" computational engine. Any file matching this pattern is a self-contained, swappable unit that:
    1.  Accepts a "Job JSON" string.
    2.  Performs all cosmological calculations and statistical fitting.
    3.  Returns a single "Results JSON" string with all findings.

-   **`output_manager.py`**: The "Renderer." This module accepts the final "Results JSON" and generates all user-facing outputs (plots, CSVs) from this data, without needing to know how the calculations were performed.

-   **`data_loaders.py`**: A modular system for parsing different SNe Ia and BAO data formats. Unchanged in v1.4a.

-   **`doc.json`**: The authoritative technical specification for the entire suite, including the formal schema for the Job/Results JSON interface.

-   **Model Plugins (`*.py`) & Definitions (`*.md`)**: The two-file system for defining new cosmological models. As of v1.4a, it is highly recommended that plugins be single-core to ensure stability.

---

## 4. Workflow Overview

1.  **Dependency Check**: `copernican.py` first verifies all required Python libraries are available.
2.  **Engine Selection**: The user is prompted to select which version of the `cosmo_engine` they wish to use for the analysis.
3.  **Configuration**: The user specifies the file paths for the alternative model, SNe data, and (optionally) BAO data.
4.  **Job Aggregation**: `copernican.py` passes these paths to `input_aggregator.py`, which builds the complete "Job JSON".
5.  **Engine Execution**: `copernican.py` dynamically loads the chosen engine and passes it the "Job JSON". The engine runs in isolation, performing all fits and calculations.
6.  **Output Generation**: The engine returns a "Results JSON" to `copernican.py`, which passes it to `output_manager.py` to generate all plots and data files.
7.  **Loop or Exit**: The user is prompted to run another evaluation or exit.

---

## 5. Development History & Roadmap

### Version 1.4a Updates (The Great Refactor)

This version represents the most significant architectural evolution of the project, achieving the long-term goal of a truly modular system.

-   **Decoupled Engine Architecture:** The core logic has been refactored into a "black box" `cosmo_engine` that communicates with the rest of the suite via a JSON-based DSL.
-   **Swappable Engines:** The main `copernican.py` script now discovers and allows the user to select any available `cosmo_engine_*.py` file, enabling easy testing of different computational backends or statistical methods in the future (e.g., `cosmo_engine_v1.5_numba.py`).
-   **Enhanced Stability:** By isolating components, bugs are easier to trace and fix. The root cause of the `multiprocessing` conflicts was identified and resolved by simplifying model plugins to be single-core, with the engine itself being the true unit of parallelization in future versions.
-   **Clear Path to Declarative Models:** The new JSON interface is the critical foundation for a future engine that can parse model equations directly from `.md` files, eliminating the need for custom `.py` plugins.

### Version 1.3 Updates (Stable Release)

This version addressed critical bugs and added significant enhancements to the data output, plot clarity, and developer framework.
-   **CRITICAL BUG FIX - BAO Plotting Restored:** Fixed a bug that caused smooth model lines on BAO plots to fail to render.
-   **New Developer Specification (`doc.json`):** Added a comprehensive `doc.json` file to act as a machine-readable "dictionary" for the suite's architecture.

### The Future Vision: A Universal Engine

The v1.4a refactor has turned this vision into a concrete, achievable goal. The next major architectural evolution will build directly on this foundation.

-   **Deprecation of `.py` Plugins:** The long-term goal is to eliminate the need for custom `model_plugin.py` files.
-   **Universal Math Engine (v1.5+):** A new, powerful engine (`cosmo_engine_v1.5_declarative.py`) will be developed that can directly parse, interpret, and solve the mathematical equations provided in the `.md` definition files using symbolic math libraries like `sympy`. It will also integrate JIT compilers like `numba` for high performance.
-   **Declarative Model Definition:** Researchers will only need to create a single, well-defined `.md` file. By specifying their model's key equations and parameters, the universal engine will handle the rest.

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