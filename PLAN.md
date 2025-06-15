# DEV NOTE (v1.4.1)
Added an initial roadmap for replacing model `.py` files with a JSON-based DSL and pluggable engines. This file will track progress as phases are completed.

# Development Roadmap
This document outlines the steps required to refactor the Copernican Suite so that cosmological models are defined entirely in a domain specific language (DSL). Engines become black-box components that consume compiled models. After completing each phase or step, update the corresponding section with an explanation of how it was achieved.

## Phase 1 – Define the Model DSL
1. **Design the JSON schema**
   - Specify required fields: `model_name`, `version`, `date`, parameter list, and LaTeX or SymPy equations for SNe Ia and BAO.
   - Include optional fields for constants, fixed parameters, and metadata about future data types.
2. **Draft example JSON models**
   - Convert an existing Markdown+Python model into JSON as a template.
   - Document the schema in `README.md` so contributors can easily create new models.

## Phase 2 – Implement a DSL Parser/Compiler
1. **Create `model_compiler.py`**
   - Load and validate JSON files against the schema.
   - Parse equations with SymPy and generate Python callables that match the current engine interface.
2. **Error handling and robustness**
   - Provide clear messages for missing fields or malformed equations.
   - Guard against division by zero and other numerical pitfalls when compiling wild theories.

## Phase 3 – Engine Abstraction Layer
1. **Refactor existing engine**
   - Modify `cosmo_engine_1_4b.py` to accept compiled model objects instead of importing Python plugins directly.
2. **Pluggable engines**
   - Standardize an interface so additional engines (Numba, OpenCL, etc.) can drop in without altering model definitions.

## Phase 4 – Incremental Migration of Models
1. **Convert Markdown models**
   - Translate each `cosmo_model_*.md` and its plugin into a single JSON file using the DSL.
2. **Validate conversions**
   - Confirm that compiled JSON models reproduce the outputs of the original plugins.
3. **Retire `.py` plugins**
   - Remove outdated plugin files once all engines operate on the DSL.

## Phase 5 – Expand Back-End Support
1. **Implement alternative engines**
   - Add optional back ends using Numba for JIT acceleration and OpenCL/CUDA for GPU support.
2. **Maintain engine modularity**
   - Ensure each engine conforms to the standardized interface so new technologies can be integrated easily.

## Phase 6 – Future Data Types and Extensibility
1. **Extend schema for new observations**
   - Prepare fields for CMB, gravitational waves, standard sirens, and other data.
2. **Update parsers and documentation**
   - Add parser modules and examples when new data types are supported.

---
**Progress Tracking**
After completing any phase or step, add a short paragraph under that bullet explaining how it was accomplished. This keeps the roadmap up to date as development proceeds.
