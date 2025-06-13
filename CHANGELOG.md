# Copernican Suite Change Log

## Version 1.4rc
- Refactored into a fully pluggable architecture with discoverable engines,
  parsers and models.
- Migrated specification into `AGENTS.md` and cleaned documentation.
- Added modular data and model directories.

## Version 1.3 (Stable Release)
- CRITICAL BUG FIX - BAO plotting restored (fixed multiprocessing issue).
- Added developer specification `doc.json`.
- BAO plot clarity improved with transparency.
- Streamlined CSV outputs to detailed files only.

## Version 1.2 (Major Refactor)
- Removed GPU code for stability.
- Implemented robust multiprocessing using `psutil`.
- Added test mode and cache cleanup loop.
