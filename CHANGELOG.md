## How to Log Changes
Add one line for each substantive commit or pull request directly under the latest version header. AI assistant warning: please, always check the current date when logging new changes, and datestamp them with a current date! Don't put dates that are in the future or in the past! Follow this template: 
```
## Version 1.1.0
- 2025-05-27: Added plotting and CSV (Apostol Apostolov) 

## Version 1.0.0
- 2025-05-26: Debugged copernican.py script (AI assistant)
- 2025-05-26: Created copernican.py (Apostol Apostolov)

```
## Log changes below and keep the line after this one empty:

## Version 1.14.9
- 2025-07-26: Tweaked plot margins and info box spacing; improved footer layout and tightened CMB subplot padding (AI assistant)

## Version 1.14.8
- 2025-07-26: Improved footer spacing, unified CMB legends and added verbose dataset summaries (AI assistant)

## Version 1.14.7
- 2025-07-26: Combined JLA systematic and statistical covariances and updated parser logic (AI assistant)

## Version 1.14.6
- 2025-07-26: Unified info box spacing with margins, adjusted footer placement and fixed CMB title overlap (AI assistant)

## Version 1.14.5
- 2025-07-26: Documented JLA covariance fallback and tightened info box layout (AI assistant)

## Version 1.14.4
- 2025-07-27: Handled near-singular JLA covariance by falling back to diagonal errors (AI assistant)

## Version 1.14.3
- 2025-07-27: Removed deprecated UniStra SNe data and fixed JLA covariance handling (AI assistant)
- 2025-07-27: Improved fit report outputs and enlarged plot dimensions (AI assistant)

## Version 1.14.2
- 2025-07-26: Lightened grid lines, widened plot margins and fixed BAO info box equation parsing (AI assistant)

## Version 1.14.1
- 2025-07-26: Human intervention in CHANGELOG.md due to messed up order, dates and lack of template (Apostol Apostolov)
- 2025-07-26: Unified plot style and improved info boxes across all data types (AI assistant)

## Version 1.14.0
- 2025-07-25: Added JLA 2014 dataset with full covariance matrix and new metadata field `authors_all` (AI assistant)
- 2025-07-25: Fixed version string handling and updated documentation (AI assistant)

## Version 1.13.1
- 2025-07-25: Renamed test BAO dataset and updated documentation (AI assistant)

## Version 1.13.0
- 2025-07-24: Enforced automatic SemVer bumps and updated version references (AI assistant)

## Version 1.12.9
- 2025-07-19: Expanded and clarified documentation; explained `.egg-info` folder and added CONTRIBUTING guide (AI assistant)

## Version 1.12.8
- 2025-07-19: Updated logger to avoid duplicate console output and capture user input (AI assistant)
- 2025-07-19: Footer lines now rendered with smaller font to prevent overlap (AI assistant)

## Version 1.12.7
- 2025-07-16: Log now records console output verbatim and strips absolute paths (AI assistant)

## Version 1.12.6
- 2025-07-16: Improved footer wrapping, plot legends and info boxes with combined chi2; tweaked BAO residuals (AI assistant)

## Version 1.12.5
- 2025-07-16: Ignored virtual env directories when scanning imports for dependency check (AI assistant)
- 2025-07-16: Removed automatic dependency installation and virtual environment logic (AI assistant)
- 2025-07-16: Implemented BAO residual plots with smoothed averages (AI assistant)
- 2025-07-16: Added smoothed residual averages to all plots and extended footer wrapping (AI assistant)
- 2025-07-16: Dependency check now prints install command with only missing packages (AI assistant)
- 2025-07-16: Dependency checker parses imports via AST and prints OS-aware install instructions (AI assistant)
- 2025-07-16: Fixed logger crash and missing AST import in dependency check (AI assistant)

## Version 1.12.4
- 2025-07-15: Fixed CMB spectrum scaling bug and added Dl verification test (AI assistant)
- 2025-07-15: Updated documentation and developer guide with raw string rule (AI assistant)
- 2025-07-15: Converted math docstrings to raw strings to silence escape warnings (AI assistant)
- 2025-07-15: Fixed dependency check for Python 3.13 `find_spec` ValueError (AI assistant)

## Version 1.12.3
- 2025-07-13: Unified timestamp handling and console output format updated (AI assistant)

## Version 1.12.2
- 2025-07-10: Unified dataset metadata files and expanded plot footers (AI assistant)
- 2025-07-10: Fixed file name sanitization for Planck dataset (AI assistant)

## Version 1.12.1
- 2025-07-10: Dynamic BAO metadata parsing and verbose fit summaries (AI assistant)

## Version 1.11.9
- 2025-07-10: Automatic virtual environment setup and start scripts for Windows, macOS and Linux. Cancelling a run now removes its log file (AI assistant)

## Version 1.11.8
- 2025-07-09: Added official JLA and Pantheon+ dataset names and short identifiers (AI assistant)
- 2025-07-09: Simplified plot footers and updated documentation (AI assistant)

## Version 1.11.7
- 2025-07-09: Renamed Pantheon+ files and made parser auto-detect dataset names (AI assistant)
- 2025-07-09: Moved chi-squared helpers back into the engine and removed chi2_helper module (AI assistant)

## Version 1.11.6
- 2025-07-09: Removed deprecated 1.4b and numba engines and set combined engine as default (AI assistant)

## Version 1.11.5
- 2025-07-09: Documented SNe refinement step in workflow section of README (AI assistant)
- 2025-07-08: Added SNe pre-fit step to combined engine to improve convergence and updated documentation (AI assistant)
- 2025-07-08: Updated minimum Python version to 3.12 and synced README (AI assistant)
- 2025-07-08: Added runtime check for Python version and documented exit behavior (AI assistant)

## Version 1.11.4
- 2025-07-08: Expressions in all cosmo_model JSON files converted to LaTeX and parser updated (AI assistant)

## Version 1.11.3
- 2025-07-07: Fixed missing extra CMB parameters in run_cmb_analysis and bumped version (AI assistant)

## Version 1.11.2
- 2025-07-07: Moved chi-squared helpers to chi2_helper module and updated docs (AI assistant)

`- 2025-07-05: short summary (author)`
## Version 1.11.1
- 2025-07-07: Unified SNe data processing and chi-squared helpers (AI assistant)


## Version 1.10.1-beta (Development Release)
- 2025-07-07: Unified CMB handling with SNe and BAO, removed engine interface fallbacks, updated docs (AI assistant)

## Version 1.9.3-beta (Development Release)
- 2025-07-07: Fixed parameter list mutation in combined engine and bumped version (AI assistant)
- 2025-07-07: Removed deprecated L-BFGS-B solver options to silence SciPy warnings (AI assistant)
- 2025-07-07: Increased CMB cache precision to six significant digits (AI assistant)

## Version 1.9.2-beta (Development Release)
- 2025-07-07: Bumped version to 1.9.2-beta and expanded code comments (AI assistant)

## Version 1.9.1-beta (Development Release)
- 2025-07-07: Renamed scripts package to copernican_lib and updated documentation (AI assistant)

Example:
`- 2025-07-15: Improved BAO solver stability (Alice Doe)`

## Version 1.9.0-beta (Development Release)
- 2025-07-07: Centralized optimization wrappers and updated documentation (AI assistant)

## Version 1.8.5-beta (Development Release)
- 2025-07-07: Enforced spawn start method and restricted JSON validation to main process (AI assistant)

## Version 1.8.4-beta (Development Release)
- 2025-07-07: Restored compatibility of chi_squared_cmb with plugin interface (AI assistant)
- 2025-07-07: Bumped development version and updated documentation (AI assistant)
- 2025-07-07: Documented engine-plugin architecture and updated JSON example (AI assistant)
- 2025-07-07: Revised AGENTS overview and expanded README with developer guide (AI assistant)
- 2025-07-07: Fixed test discovery and matplotlib cleanup in run-tests mode (AI assistant)

## Version 1.8.3-beta (Development Release)
- 2025-07-06: Rewrote combined engine for true joint optimisation (AI assistant)
- 2025-07-06: Fixed CMB chi-squared interface and allowed fitting of CAMB
  parameters (AI assistant)

## Version 1.8.2-beta (Development Release)
- 2025-07-06: Optimized CMB evaluation with cached CAMB calls (AI assistant)
- 2025-07-06: Enabled true joint fitting with optional SALT2 parameters (AI assistant)

## Version 1.8.1-beta (Development Release)
- 2025-07-06: Made combined-fit engine verbose and fixed docstring escape warning (AI assistant)

## Version 1.8.0-beta (Development Release)
- 2025-07-06: Added combined-fit engine and optional test execution (AI assistant)
- 2025-07-06: Bumped version to 1.8.0-beta (AI assistant)
- 2025-07-06: Integrated combined-fit workflow and updated documentation (AI assistant)

## Version 1.7.10-beta (Development Release)
- 2025-07-06: Corrected CAMB spectrum scaling and updated docs (AI assistant)
- 2025-07-06: Bumped version to 1.7.10-beta (AI assistant)

## Version 1.7.11-beta (Development Release)
- 2025-07-06: Fixed Planck 2018 lite parser and trimmed covariance to TT block (AI assistant)
- 2025-07-06: Bumped version to 1.7.11-beta (AI assistant)

## Version 1.7.12-beta (Development Release)
- 2025-07-06: Added TE/EE spectrum handling and improved cosmic variance plotting (AI assistant)
- 2025-07-06: Bumped version to 1.7.12-beta (AI assistant)

## Version 1.7.9-beta (Development Release)
- 2025-07-06: Fixed Planck lite scaling and covariance endianness (AI assistant)
- 2025-07-06: Enhanced default CMB wrapper and engine spectra output (AI assistant)
- 2025-07-06: Updated documentation and version bump to 1.7.9-beta (AI assistant)

## Version 1.7.8-beta (Development Release)
- 2025-07-06: Added dedicated CMB analysis stage with verbose logging (AI assistant)
- 2025-07-06: Updated documentation and version bump to 1.7.8-beta (AI assistant)

## Version 1.7.7-beta (Development Release)
- 2025-07-06: Overhauled Planck parser with µK² conversion and TE/EE support (AI assistant)
- 2025-07-06: Redesigned CMB plot with log scaling and variance shading (AI assistant)
- 2025-07-06: Documentation updates and version bump to 1.7.7-beta (AI assistant)

## Version 1.7.6-beta (Development Release)
- 2025-07-05: Bumped COPERNICAN_VERSION and docs to 1.7.6-beta. (AI assistant)
- 2025-07-06: Added TE/EE spectrum handling in parser, engine and plotter. (AI assistant)
- 2025-07-06: Improved Planck lite parser covariance checks with fallback warnings. (AI assistant)
- 2025-07-06: Fixed chi-squared label formatting warnings in plotter. (AI assistant)

## Version 1.7.5-beta (Development Release)
- 2025-07-05: Removal of user-selectable test mode. (AI assistant)
- 2025-07-05: Automatic functional tests run at startup. (AI assistant)
- 2025-07-05: Updated documentation and model guide. (AI assistant)
- 2025-07-05: Clarified CMB requirements in cosmo_model_guide and bumped guide version. (AI assistant)
- 2025-07-05: Documented automatic startup test suite in README. (AI assistant)

## Version 1.7.4-beta (Development Release)
- 2025-07-05: Fixed unit conversion (K\u00b2 \u2192 \u03bcK\u00b2) by applying a 1e12 scale factor (AI assistant)
- 2025-07-05: Added neutrino density mapping (`omnuh2`) to the \u039bCDM parameter map (AI assistant)

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

