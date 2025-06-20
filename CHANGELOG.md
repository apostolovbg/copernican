# Copernican Suite Change Log
<!-- DEV NOTE (v1.5.0): Adopted semantic versioning. -->
<!-- DEV NOTE (v1.5g): Data source reorganization and version bump. -->
## Version 1.5.0 (Development Release)
- Data files and parsers reorganized under ``data/<type>/<source>/``.
- Parser selection now based on data source only.
- Removed deprecated `parsers/` directory and UniStra h2 parser.
- Updated documentation for version 1.5.0.
- Hotfix: Prompts list friendly dataset names with a clear title for every selection.

## Version 1.5f (Development Release)
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
