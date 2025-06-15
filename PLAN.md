# DEV NOTE (v1.5b)
Updated for Phase 0 and Phase 1 completion. Added pipeline skeleton modules and
documented the new JSON DSL with an example model file.

# Development Roadmap
This document outlines the steps required to refactor the Copernican Suite so
that cosmological models are defined entirely in a domain specific language
(DSL). Engines will become black-box components that consume compiled models.
Model JSON files will contain **only declarative information** describing the
theory. Generated Python code will live in the cache and never inside the model
files themselves. After completing each phase or step, update the corresponding
section with an explanation of how it was achieved.

## Phase 0 – Pipeline Overview
Before diving into individual phases, the suite will be reorganized around a
clear processing pipeline. `copernican.py` acts as the manager that coordinates
the following helper modules:

1. **`scripts/model_parser.py`** – validates `cosmo_model_*.json` files against
   the DSL template and writes validated information to `models/cache/cache_*`.
2. **`scripts/model_compiler.py`** – converts the parsed DSL into Python
   callables stored in the cache. No executable code is kept inside model JSON
   files.
3. **`scripts/engine_interface.py`** – loads the compiled callables from the
   cache, verifies them, and passes them to the selected engine implementation.
4. **`scripts/error_handler.py`** – monitors the workflow and produces user and
   developer friendly diagnostics if validation or computation fails.
5. **`scripts/logger.py`**, **`scripts/plotter.py`**, and **`scripts/csv_writer.py`**
   – modular utilities for logging, plotting, and CSV output respectively.

Cache files will be retained until the end of a run, at which point the user may
choose to delete them. This pipeline keeps each component focused and makes the
engines truly pluggable.

**Progress:** Implemented in version 1.5a by creating the `scripts/` package
with placeholder modules for parsing, conversion, engine interfacing, logging,
plotting, CSV writing and error handling. A cache directory was also added
under `models/`.

## Phase 1 – Define the Model DSL
1. **Design the JSON schema**
   - Specify required fields: `model_name`, `version`, `date`, parameter list, and LaTeX or SymPy equations for SNe Ia and BAO.
   - Include optional fields for constants, fixed parameters, and metadata about future data types.
   - Keep the syntax simple so that any scientist can copy an existing
    `cosmo_model_*.json` as a template and fill in their own theory without
    writing Python.
2. **Draft example JSON models**
   - Convert an existing Markdown+Python model into JSON as a template.
   - Document the schema in `README.md` so contributors can easily create new models.
   - Note a future tool (`model_json_maker.py`) may automate this process once
     the DSL stabilizes.

**Progress:** The JSON schema was outlined in `README.md` and an example file
`models/cosmo_model_lcdm.json` demonstrates the structure. These additions
complete Phase 1 for version 1.5a.

## Phase 2 – Implement a DSL Parser/Compiler
1. **Create `model_parser.py` and `model_compiler.py`**
   - `model_parser.py` validates the DSL files and writes sanitized content to the cache.
   - `model_compiler.py` reads that cache entry, parses equations with SymPy, and
     generates Python callables that match the current engine interface.
2. **Error handling and robustness**
   - Provide clear messages for missing fields or malformed equations.
   - Guard against division by zero and other numerical pitfalls when compiling
     wild theories.

**Progress:** Phase 2 implemented in version 1.5b. Parser validates JSON and compiler generates safe callables.
## Phase 3 – Engine Abstraction Layer
1. **Create `engine_interface.py`**
   - Serves as a manager between compiled models and the chosen engine.
   - Validates the generated callables before sending them to the backend.
2. **Refactor existing engine**
   - Modify `cosmo_engine_1_4b.py` to receive callables from
     `engine_interface.py` rather than importing plugins directly.
3. **Pluggable engines**
   - Standardize an interface so additional engines (Numba, OpenCL, etc.) can
     drop in without altering model definitions.

## Phase 4 – Incremental Migration of Models
1. **Convert Markdown models**
   - Translate each `cosmo_model_*.md` and its plugin into a single JSON file using the DSL.
2. **Validate conversions**
   - Confirm that compiled JSON models reproduce the outputs of the original plugins.
3. **Retire `.py` plugins**
   - Remove outdated plugin files once all engines operate on the DSL.
4. **Cache management**
    - Store compiled code in `models/cache/cache_*.json` during runs and prompt
      the user to delete or keep the cache afterward.

## Phase 5 – Expand Back-End Support
1. **Implement alternative engines**
   - Add optional back ends using Numba for JIT acceleration and OpenCL/CUDA for GPU support.
2. **Maintain engine modularity**
   - Ensure each engine conforms to the standardized interface so new technologies can be integrated easily.
3. **Modular utilities**
   - Split plotting and CSV generation into `plotter.py` and `csv_writer.py`.
   - Move logging into `logger.py` so each component can log through a common interface.

## Phase 6 – Future Data Types and Extensibility
1. **Extend schema for new observations**
   - Prepare fields for CMB, gravitational waves, standard sirens, and other data.
2. **Update parsers and documentation**
   - Add parser modules and examples when new data types are supported.

---
**Progress Tracking**
After completing any phase or step, add a short paragraph under that bullet explaining how it was accomplished. This keeps the roadmap up to date as development proceeds.
