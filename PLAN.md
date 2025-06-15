<!-- DEV NOTE (v1.5a): Created plan document and completed Phase 1 tasks. -->
# Development Plan

## Phase 1: Version 1.5a Preparation
This phase bumps the development version to **1.5a** and establishes new
contributor rules.

### Work Performed
- Updated version references in documentation and key modules.
- Added a rule requiring human approval for any future version changes.
- Added a rule forbidding Git conflict markers in all files.

### Implementation Notes
The updates were applied directly in `README.md`, `AGENTS.md` and several
package `__init__` files. The main program `copernican.py` and
`output_manager.py` now report version `1.5a` via the `COPERNICAN_VERSION`
constant and header comments. The `CHANGELOG.md` was extended to include this
development entry.

Future phases will build upon this foundation.
