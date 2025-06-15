# Copernican Suite Change Log
# DEV NOTE (v1.5c): Documented Phase 3 engine abstraction and version bump.
## Version 1.5c (Development)
- Implemented engine abstraction layer via `engine_interface`.
- Refactored `cosmo_engine_1_4b.py` to accept model dictionaries.
- Updated documentation and version metadata for the 1.5c cycle.

## Version 1.5b (Development)
- Implemented DSL parser validation and new `model_compiler.py` replacing
  `model_compiler.py`.
- Improved error handling for malformed JSON and numerical issues.
- Updated documentation and version metadata for the 1.5b development cycle.

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
