# Copernican Suite Change Log
<!-- DEV NOTE (v1.5.0): Adopted semantic versioning. -->
<!-- DEV NOTE (v1.5g): Data source reorganization and version bump. -->
<!-- DEV NOTE (v1.5.1): Removed theory block and auto-generated parameter guesses. -->

## How to Log Changes
Add one line for each substantive commit or pull request directly under the latest version header. AI assistant warning: please, always check what is the current date when you are logging last changes, and datestamp them with a current date! Don't put dates that are in the future or in the past! Use this template:

`- YYYY-MM-DD: short summary (author)`

Example:
`- 2025-07-15: Improved BAO solver stability (Alice Doe)`
## Version 1.7.3-beta (Development Release)
- 2025-07-05: Fixed Planck covariance reader for ASCII data and ensured CMB parameters use SNe best-fit values (AI assistant)
- 2025-07-05: Corrected Planck covariance parsing for binary Fortran record (AI assistant)
- 2025-07-05: Re-added integral expression support using numerical quadrature (AI assistant)
- 2025-07-05: Added `_wrap_math` helper and updated parameter label rendering (AI assistant)
- 2025-07-05: Updated LICENSE.md with new definitions and effective date (AI assistant)
- 2025-07-05: Restored 1.6.4 and 1.6.5 changelog entries (AI assistant)
## Version 1.7.2-beta (Development Release)
- 2025-07-05: Fixed Planck covariance parser using np.loadtxt (AI assistant)
- 2025-07-05: Added default CAMB parameter mapping from SNe fits (AI assistant)
- 2025-07-05: Handled binary Planck covariance matrix fallback (AI assistant)
## Version 1.7.1-beta (Development Release)
- 2025-07-05: Updated version references to 1.7.1-beta (AI assistant)
- 2025-07-05: Implemented Planck 2018 lite CMB parser (AI assistant)
- 2025-07-05: Added `valid_for_cmb` flag and updated plugin validation (AI assistant)
- 2025-07-05: Added CAMB-based CMB analysis and chi-squared routines (AI assistant)
- 2025-07-05: Added cmb.param_map metadata to models and documentation (AI assistant)
- 2025-07-05: Stored CAMB parameter order in Planck 2018 parser (AI assistant)
- 2025-07-05: Added automatic CMB wrapper and parameter mapping helper (AI assistant)
- 2025-07-05: run_cmb_analysis now converts fitted parameters with get_camb_params (AI assistant)

## Version 1.7.0-beta (Development Release)
- 2025-07-05: Skip CMB evaluation when model sets valid_for_cmb=false (AI assistant)
- 2025-07-05: Implemented CMB spectrum plotting (AI assistant)
- 2025-07-05: Added CMB residual CSV export (AI assistant)
- 2025-07-05: Documented cmb.param_map usage and parser param_names attribute (AI assistant)
- 2025-07-05: Bumped version to 1.7.0 and reorganized changelog (AI assistant)
- 2025-07-05: Removed obsolete CMB placeholder parser and dataset (AI assistant)
- 2025-07-05: Added CAMB dependency to pyproject and updated docs (AI assistant)
- 2025-07-05: Corrected CMB spectrum units and Planck parser to use D_l (AI assistant)
- 2025-07-05: Removed DEV NOTE headers from pyproject.toml (AI assistant)

## Version 1.6.5 (Patch Release)
- 2025-06-23: Fixed plot info boxes to display equations from the selected alternative theory and ensured Greek letters render correctly (AI assistant)
- 2025-06-23: Updated README and AGENTS documentation for corrected JSON schema and version bump (AI assistant)

## Version 1.6.4 (Patch Release)
- 2025-06-23: Added numerical quadrature support for Integral expressions (AI assistant)

## Version 1.6.3 (Patch Release)
- 2025-06-22: Restored `pyproject.toml` and silenced Pandas whitespace warning (AI assistant)
- 2025-06-22: Declared Python 3.13.1+ requirement in pyproject and README (AI assistant)
## Version 1.6.2 (Patch Release)
- 2025-06-22: Added LCDM equations and sound horizon formula (AI assistant)
## Version 1.6.1 (Patch Release)
- Restored model equations in plot info boxes.
- 2025-06-22: Fixed plot crashes when model equations used display-mode dollar signs (AI assistant)
- Added standardized plot footer with run metadata.
- start.command cleaned up.
- 2025-06-21: Documented stable plotting style and algorithms (AI assistant)
- 2025-06-21: Clarified when MINOR vs PATCH increments occur in README (AI assistant)
## Version 1.6 (Stable Release)
- 2025-06-21: Fixed trailing text in start.command and ensured newline (AI assistant)
- 2025-06-21: First stable release with reliable SNe Ia and BAO calculations (AI assistant)
- 2025-06-21: Legacy DEV NOTE headers removed from source files and notes migrated to `CHANGELOG.md` (AI assistant)
- 2025-06-21: Plugin now exposes model equations and filename (AI assistant)
- 2025-06-21: Plugin filename stored during JSON loading (AI assistant)
- 2025-06-21: Plots now include a timestamped footer with comparison details (AI assistant)
## Version 1.5.1 (Development Release)
- 2025-06-20: Added CHANGELOG template and updated docs to reference it (AI assistant)
- Removed ``initial_guess`` from JSON models; parameter guesses now computed
  automatically from bounds.
- Consolidated model metadata: ``theory`` block removed and equations moved
  under ``equations``.
- Documentation updated to reflect declarative model design.
- Development protocol revised: DEV NOTE markers removed in favor of documenting changes in `CHANGELOG.md` or `AGENTS.md`.
- Schema documentation updated: `abstract` and `description` are now mandatory and all contributors summarize updates in `CHANGELOG.md`.
- 2025-06-20: Added explicit `rs_expression` to `cosmo_model_lcdm.json` and migrated legacy documentation notes to `CHANGELOG.md` (AI assistant)

## Version 1.5.0 (Development Release)
- Data files and parsers reorganized under ``data/<type>/<source>/``.
- Parser selection now based on data source only.
- Removed deprecated `parsers/` directory and UniStra h2 parser.
- Updated documentation for version 1.5.0.
- Hotfix: Prompts list friendly dataset names with a clear title for every selection.

## Version 1.5f (Development Release)
- Completed Phase 6: JSON schema extended with optional fields for CMB,
  gravitational waves and standard sirens. Added placeholder parser modules
  and loader functions for these data types.
- Updated documentation for version 1.5f.
- Hotfix 5: Removed automatic dependency installer. Users are now instructed to
  run a printed `pip install` command when packages are missing.
- Hotfix 7: `Hz_expression` added to JSON models and compiled automatically for
  distance predictions.
- Hotfix 8: Sound horizon `r_s` is now computed automatically when possible using
  a fallback integral if `rs_expression` is missing.
- Hotfix 9: Parser auto-discovery now searches the project's top-level `parsers`
  directory instead of a nonexistent `scripts/parsers` folder.
- Hotfix 10: Fixed BAO smooth curve generation by allowing `_dm` to accept array
  redshift values.

## Version 1.5e (Development Release)
- Added Numba-based engine and modular utility wrappers.
- Updated documentation for version 1.5e.

## Version 1.5d (Development Release)
- Completed Phase 4: all models converted to JSON and legacy plugins removed.
- Updated documentation and headers for version 1.5d.
- Automatic dependency installer added and invoked by `copernican.py` when
  packages are missing.

## Version 1.5c (Development Release)
- Completed Phase 3: engine_interface now validates plugins and engines use the new abstraction layer.
- Updated documentation and headers for version 1.5c.

## Version 1.5b (Development Release)
- Completed Phase 2: parser caches validated JSON and coder generates callables with sanity checks.
- Updated documentation and headers for version 1.5b.

## Version 1.5a (Development Release)
- Introduced JSON-based model pipeline and new `scripts/` modules.
- Added example JSON model and updated documentation for version 1.5a.

## Version 1.4.1 (Maintenance Release)
- LCDM model separated into lcdm.py plugin.
- Added splash screen and improved logging with per-run timestamps.


## Version 1.4 (Stable Release)
- Refactored into a fully pluggable architecture with discoverable engines,
  parsers and models.
- Migrated specification into `AGENTS.md` and cleaned documentation.
- Added modular data and model directories.
- Finalized engine and model interfaces for long-term stability.

## Version 1.3 (Stable Release)
- CRITICAL BUG FIX - BAO plotting restored (fixed multiprocessing issue).
- Added developer specification `doc.json`.
- BAO plot clarity improved with transparency.
- Streamlined CSV outputs to detailed files only.

## Version 1.2 (Major Refactor)
- Removed GPU code for stability.
- Implemented robust multiprocessing using `psutil`.
- Added test mode and cache cleanup loop.

