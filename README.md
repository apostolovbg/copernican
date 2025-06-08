# Copernican Suite - A Modular Cosmology Framework

**Version:** 1.2 (Stable)
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

The suite is designed for:
* **Rapid Prototyping:** Quickly implement new cosmological models.
* **Robust Analysis:** Fit models to SNe Ia and BAO data.
* **Comparative Cosmology:** Directly compare model performance against ΛCDM.
* **High Performance:** Leverage GPU acceleration via OpenCL for computationally intensive fits.

---

## 2. Installation and Requirements

* **Python:** 3.8+
* **Core Libraries:** `numpy`, `scipy`, `matplotlib`, `pandas`, `pyyaml`
* **GPU Acceleration (Optional):** `pyopencl` and a compatible OpenCL driver for your hardware (NVIDIA, AMD, Intel).

```bash
pip install numpy scipy matplotlib pandas pyyaml pyopencl