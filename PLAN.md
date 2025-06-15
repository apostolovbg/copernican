# DEV NOTE (v1.5b)
Replaced the previous roadmap with a more detailed plan covering the new JSON-based model system, pipeline, and staged migration.

# Copernican Suite Refactoring Plan
This document explains how the project will evolve from the current Markdown + Python plugin system to a cleaner architecture where cosmological models are described solely in JSON. All engines will load code generated on the fly.

After finishing each phase or major step, append a short paragraph here describing what was done and when. This living document keeps everyone in sync.

## Phase 0 – Big Picture Pipeline
1. **New model arrives** – A `cosmo_model_*.json` file is placed in `models/`.
2. **`copernican.py` acts as manager** – It notices the new model and instructs helper modules to process it, much like a supervisor handing work to staff.
3. **`scripts/model_parser.py`** – Checks that the JSON follows the `cosmo_model_template.json` format. If valid, it writes a `models/cache/cache_*.json` file containing the cleaned data. If invalid, details are handed to `scripts/error_handler.py` so the user knows what to fix.
4. **`scripts/model_coder.py`** – Reads the cache, converts the model equations and plan into executable Python code, and stores that generated code back in the same cache file. There is no permanent Python model file.
5. **`scripts/engine_interface.py`** – Loads the generated code from the cache and passes it to whichever engine (`cosmo_engine_*.py`, Numba, OpenCL, etc.) the user chooses. Engines themselves remain black boxes.
6. **Results & cleanup** – `output_manager.py` orchestrates plotting via `plotter.py`, writes CSV files with `csv_writer.py`, and `logger.py` records every step. Afterward, the user is asked whether to delete the cache file.
   - *Done 2025-06-15 – Initial pipeline implemented with stub modules and JSON detection in `copernican.py`.*

This pipeline ensures that models stay purely declarative while engines receive ready-to-run Python functions.

## Phase 1 – Define the Model DSL
1. **Design a clear JSON schema**
   - Required fields: model name, version, list of parameters with guesses and bounds, and LaTeX or SymPy equations for each observable.
   - Optional fields for constants and notes about future data types.
   - Keep the syntax approachable so scientists can copy any existing `cosmo_model_*.json` and fill in their own theory without writing code.
2. **Create example models**
   - Translate one Markdown + plugin pair into JSON to serve as the official template.
   - Document the schema in `README.md` so non-programmers can create models correctly.
   - *Done 2025-06-15 – Schema documented and example `cosmo_model_lcdm.json` added.*

## Phase 2 – Build the Parser and Coder
1. **Implement `model_parser.py` and `model_coder.py`**
   - The parser validates JSON and writes sanitized content to the cache.
   - The coder reads equations from the cache, uses SymPy or similar tools to generate Python callables matching the engine interface, and stores them back in the cache.
2. **Robust error handling**
   - Detect division by zero, undefined variables, or other issues in "wild" models. Report them through `error_handler.py` so the user understands what went wrong.
   - *Done 2025-06-16 – Parser writes cache files and coder checks generated functions before use.*

## Phase 3 – Engine Abstraction Layer
1. **Introduce `engine_interface.py`**
   - Acts as the go-between for generated code and the chosen engine.
   - Ensures the callables conform to the expected signatures before they reach the engine.
2. **Adapt existing engines**
   - Update `cosmo_engine_1_4b.py` and future engines to obtain functions from the interface rather than importing plugins directly.
   - Maintain a consistent API so new engines (Numba, OpenCL, etc.) can be added with minimal effort.

## Phase 4 – Migrate Existing Models
1. **Convert `.md` definitions**
   - For each model, read its Markdown file and Python plugin. Recreate the same information in a single `.json` file using the new schema.
   - During this period, the original `.py` files can serve as edge-case examples of what the engine might need to compute.
2. **Validate outputs**
   - Run the converted models through the suite and compare results with the original plugins to ensure correctness.
3. **Deprecate and delete plugins**
   - Once all models work through the JSON pipeline, remove the old `.py` plugin files from the repository.
   - Going forward, all Python code for models will be produced on the fly and kept only in the cache for each run.

## Phase 5 – Expand Back-End Support
1. **Add alternative engines**
   - Implement faster back ends using Numba or GPU acceleration. Each engine simply consumes the callables provided by `engine_interface.py`.
2. **Refine modular utilities**
   - Keep `logger.py`, `plotter.py`, and `csv_writer.py` separate so they can be reused across engines and future data types.

## Phase 6 – Future Data Types & Extensibility
1. **Prepare the schema for new observations**
   - Add optional fields in the JSON for CMB, gravitational waves, standard sirens, and any other data types we may add later.
2. **Update parsers and documentation**
   - When new data become available, introduce matching parser modules and update examples so users can easily extend the suite.

---
### Progress Tracking
Whenever a phase or bullet point is completed, insert a short note below it summarizing what changed and the date. This running commentary keeps the plan relevant for both developers and non-programmers.

- **2025-06-15** – Phase 0 implemented. `copernican.py` now detects JSON model files and processes them through the new `scripts/` pipeline.
- **2025-06-15** – Phase 1 completed. Created `model_parser.py`, `model_coder.py`, and `engine_interface.py`; added an example JSON model and documented the schema in `README.md`.
- **2025-06-16** – Phase 2 completed. Parser now writes sanitized models to `models/cache/`; coder loads from cache, generates functions with sanity checks, and updates the cache.

