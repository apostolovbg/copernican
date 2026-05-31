# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-05-31
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

Use this plan to track the Copernican package-structure, runtime-entrypoint,
dependency-surface, license-surface, and governance migration.

Keep slices dependency-ordered, concrete, current, and runtime-focused.

## Table of Contents

1. [Problem Preamble](#problem-preamble)
2. [Overview](#overview)
3. [How Slices Are Executed](#how-slices-are-executed)
4. [Execution Slices](#execution-slices)
5. [Validation Routine](#validation-routine)

## Problem Preamble

Copernican did not begin as a standard Python package. It began as a
single runtime script, `copernican.py`, with supporting folders growing
around it over time. The current repository still reflects that history.

The old shape was understandable when the project was one script:

* `copernican.py` was the main runtime/orchestrator.
* runtime-adjacent folders lived beside it at the repository root;
* output behavior grew around the script;
* model files lived as local YAML assets;
* dependency management treated the repository as one dependency surface;
* DevCovenant itself lived inside this repository as ordinary code.

That historical shape became fragile after DevCovenant matured.

The current DevCovenant model has separate dependency and license surfaces:

* the DevCovenant runtime surface;
* the Copernican package runtime surface;
* the root workspace surface;
* package-level license output;
* root-level license output.

Copernican has not fully migrated to that model. Instead, it carries a
hybrid layout:

* `copernican.py` remains at the repository root;
* `copernican_lib/` acts like a package surface;
* `copernican_lib/` is not the real project package identity;
* `engines/` is treated as importable runtime code from the root;
* `models/` is a repository-level model library;
* `models/` is not package code;
* `data/` still holds curated datasets and parser modules today;
* the installed package should own those curated datasets and parsers;
* dependency and license surfaces are manually pointed at `copernican_lib`;
* package metadata still has to explain the mixed shape to setuptools;
* generated dependency and license refreshes are vulnerable to mismatch.
* the old start scripts and parity policy still encode launcher history;

This is why the Astropy hash problem should not be treated as the root
issue.

Current baseline on 2026-05-31:

* `copernican.py` still owns the root orchestration path;
* `copernican_lib/gui/app.py` still owns the GUI launch path;
* `start.command` still bootstraps the managed `.venv` and legacy launcher
  flow;
* `models/` stays repository-level YAML data;
* `data/` still carries curated datasets and parsers.

The missing macOS Astropy wheel hash exposed stale generated dependency
metadata, not just a bad wheel.
`devcovenant/custom/profiles/userproject/userproject.yaml` owns the
repo-specific hash-target matrix.
`devcovenant/config.yaml` and the registry are derived outputs.
Patching the generated config by hand caused license-report collapse
because that layer was not yet rebuilt from the profile source of truth.

The fix is not to keep patching `requirements.lock`.

The fix is to make Copernican structurally normal:

* the import package should be `copernican/`;
* the distribution name should be `copernican`;
* the CLI command should be `copernican`;
* many current modules should move into `copernican/lib/`;
* some runtime modules should remain at package root;
* curated datasets and parsers should ship inside the package;
* external dataset discovery should not be supported;
* `cosmo_` should be dropped only from model and parser names;
* the old start scripts, parity policy, and linked tests should go;
* orchestration logic should live inside the package;
* `models/` should stay at the repository root;
* `models/` should remain YAML/YML examples and configurations;
* both `.yml` and `.yaml` model extensions should be accepted;
* the existing run-result shape should be preserved;
* only output base-directory selection should be improved;
* DevCovenant surfaces should be simplified after the package is real.

The desired end state is not a cosmetic rename. It is a real migration from
a script-centered repository to a package-centered repository.

After the refactor, Copernican should have a coherent identity:

* repository: `copernican`;
* distribution: `copernican`;
* import package: `copernican`;
* CLI command: `copernican`;
* GUI entrypoint: package-owned, not root-owned;
* package runtime lock: `copernican/runtime-requirements.lock`;
* package licenses: `copernican/licenses/`;
* root workspace lock: `requirements.lock`;
* root license report: `licenses/THIRD_PARTY_LICENSES.md`.

The package should own runtime Python code, curated datasets, and parser
modules. The repository root should own model examples, governance files,
dependency locks, and output defaults.

The migration must preserve existing scientific behavior. This plan is not a
model-behavior redesign, not an output-format redesign, and not a GUI rewrite
for its own sake. It is a structural migration so that the existing runtime,
GUI, CLI, model loading, output production, dependency locking, and license
reporting can operate from a clean project shape.

Token conservation is a first-class requirement. Prefer commands that
rename, move, or copy files when they can preserve meaning. Rewrite code or
whole files only when no path-preserving command can carry the change.

## Overview

* Copernican is a Python toolkit for evaluating cosmological models against
  SNe Ia, BAO, and CMB observations.
* The current repository is stable in intent but structurally legacy in
  layout.
* `devcovenant/custom/profiles/userproject/userproject.yaml` should own the
  repo-specific dependency-matrix source of truth.
* The target is a standard package layout with `copernican/` as the real
  package.
* `copernican_lib/` should be retired as the package surface.
* `copernican_lib/` should be replaced by `copernican/` with a
  `copernican/lib/` subtree.
* `copernican.py` should stop being a root-level runtime module.
* Runtime orchestration should move into package modules.
* CLI behavior should be exposed through `copernican.cli`.
* GUI behavior currently represented by
  `copernican_lib/gui/app.py` should move into the package.
* Curated datasets and parsers should ship in the package.
* External dataset discovery is not part of the support model.
* `cosmo_` should be removed only from model and parser names.
* `cosmo_model_*` should become `model_*`.
* `cosmo_parser_*` should become `parser_*`.
* `models/` should remain at repository root because models are YAML/YML
  configuration assets, not importable Python code.
* Copernican must load model files from arbitrary filesystem paths.
* Copernican must not require model files to live under repo `models/`.
* Copernican must accept both `.yml` and `.yaml` model extensions.
* The existing run-output shape must be preserved.
* Output changes should be limited to selecting a safe writable base
  directory.
* The root `output/` directory may remain the default local output target.
* Root `output/` should only be used when the current working directory is
  writable.
* Headless, installed, or supercomputer usage must not assume a desktop.
* Headless, installed, or supercomputer usage must not assume package-adjacent
  output.
* The start scripts, their parity policy, and their linked tests should be
  removed.
* DevCovenant dependency and license surfaces should be repaired after the
  package layout is real.
* The migration should leave Codex and human maintainers with a reviewable
  sequence of small, complete slices.

## How Slices Are Executed

* Each slice means a complete implementation pass, not a note.
* Each slice must leave code, tests, docs, and changelog evidence where
  behavior changed.
* Do not mark a slice done unless the relevant checks support it.
* Do not treat contract-only behavior as runtime completion.
* Do not redesign scientific output content.
* Do not redesign result file shape.
* Do not move `models/` into the package.
* Do not make model loading depend on repository-relative paths.
* Do not make output depend on a desktop path.
* Do not write runtime output into the installed package directory.
* Do not manually patch generated lock hashes as the durable fix.
* Do not create a copied custom `python` profile to compensate for package
  layout problems.
* Prefer copy, rename, or move commands when they can preserve the change.
* Avoid rewriting whole files when a path-preserving command can do the job.
* Treat token conservation as a first-class requirement.
* Use DevCovenant's normal package, root workspace, and DevCovenant runtime
  surfaces after the package structure is corrected.
* Keep generated artifacts generated.
* Keep repository-specific DevCovenant profile overrides narrow and factual.
* Preserve existing run-output naming and file layout.
* Use `CHANGELOG.md` to record slice outcomes when behavior, documentation,
  or governance changes.
* Use the configured local governance workflow around each completed slice.
* Keep every slice small enough to review, but complete enough to run.
* Remove obsolete start scripts, parity policy, and legacy tests rather than
  preserving them.

## Execution Slices

1. [open] Slice 1 - Rebaseline the current Copernican structure.

   Depends on:

   * current repository state

   Status:

   * Open. The baseline is recorded below; the slice stays open until the
     package migration work finishes.

   Surfaces:

   * `pyproject.toml`
   * `copernican.py`
   * `copernican_lib/gui/app.py`
   * `copernican_lib/`
   * `engines/`
   * `models/`
   * `data/`
   * `output/`
   * start scripts
   * `devcovenant/custom/profiles/userproject/userproject.yaml`
   * `requirements.in`
   * `requirements.lock`
   * `copernican_lib/runtime-requirements.lock`
   * `licenses/`
   * `copernican_lib/licenses/`
   * `README.md`
   * `AGENTS.md`
   * `CONTRIBUTING.md`
   * `CHANGELOG.md`
   * tests

   Scope:

   * Inspect the current runtime entrypoints.
   * Identify CLI runtime, GUI runtime, shared workflow logic, package code,
     model data, curated dataset data, parser code, generated output, and
     governance artifacts.
   * Document the current start-script behavior.
   * Document how `copernican_lib/gui/app.py` is launched and what runtime
     code it imports.
   * Document how `copernican.py` is launched and what runtime code it
     imports.
   * Document how model files are currently discovered and loaded.
   * Document how curated datasets and parsers are currently discovered.
   * Document which output files are currently produced.
   * Preserve the current output structure as the migration baseline.
   * Confirm that `models/` is repository-level YAML/YML model data.
   * Confirm that `models/` is not importable package code.
   * Confirm whether `engines/` is importable runtime code.
   * Confirm which files in `copernican_lib/` are runtime code.
   * Confirm which files in `copernican_lib/` are metadata or generated
     artifacts.
   * Confirm the current DevCovenant dependency and license surface mismatch.
   * Do not change package layout in this slice.

   Findings:

   * `copernican.py` still owns the root orchestration path and imports
     `copernican_lib` helpers directly.
   * `copernican_lib/gui/app.py` still owns the GUI launch path.
   * `engines/` remains importable runtime code from the repository root.
   * `models/` remains repository-level YAML/YML data, not package code.
   * `data/` still carries curated datasets and parser modules.
   * `start.command` and sibling launchers still bootstrap the legacy
     managed `.venv` flow.
   * DevCovenant dependency and license surfaces still point at the legacy
     `copernican_lib` layout.

   Done when:

   * current runtime entrypoints are mapped;
   * current GUI entrypoint behavior is mapped;
   * current CLI/script behavior is mapped;
   * current model-loading behavior is mapped;
   * current output shape is documented as preserved behavior;
   * package-code and non-package-data boundaries are explicit;
   * the next package-layout slice can proceed without guessing.

2. [open] Slice 2 - Create the real `copernican` package.

   Depends on:

   * Slice 1

   Surfaces:

   * `copernican/`
   * `copernican/lib/`
   * `copernican/datasets/`
   * `copernican_lib/`
   * `copernican.py`
   * `engines/`
   * `pyproject.toml`
   * imports
   * tests

   Scope:

   * Rename or replace `copernican_lib/` with `copernican/`.
   * Preserve `copernican/VERSION`.
   * Move many shared runtime helpers into `copernican/lib/`.
   * Keep some runtime modules at the package root.
   * Preserve package runtime dependency artifact locations under the new
     package path.
   * Preserve package license artifact locations under the new package path.
   * Move shared runtime/orchestration logic out of root `copernican.py`.
   * Move shared runtime/orchestration logic into package modules.
   * Move curated datasets and parser modules into `copernican/datasets/`.
   * Keep bundled datasets and parsers loadable as package resources.
   * Update the vendored-data exception in the same slice so the new
     package data path stays protected and the parser/metadata carve-out
     survives the move.
   * Do not support external dataset discovery.
   * Do not create `copernican/copernican.py` as the long-term orchestrator
     name.
   * Use a clearer package module name for orchestration, such as
     `workflow.py`, unless code review identifies a better existing name.
   * Move importable runtime engine code into `copernican/` if Slice 1
     confirms that `engines/` is package code.
   * Keep `models/` at repository root.
   * Keep `data/` as source-tree migration input until the bundled package
     data is in place.
   * Keep `output/` out of the package.
   * Add or preserve `copernican/__init__.py`.
   * Add `copernican/__main__.py` for `python -m copernican`.
   * Keep imports explicit and package-relative where appropriate.
   * Remove root-level package ambiguity from the runtime path.

   Done when:

   * `copernican/` is the real import package;
   * runtime code imports through `copernican` and `copernican.lib`;
   * root `copernican.py` is no longer required as an import module;
   * curated datasets and parsers ship in the wheel;
   * `models/` remains root-level non-package data;
   * `data/` remains a source-tree asset only;
   * the vendored-data exception protects the new package data path;
   * parser and metadata carve-outs still work after the move;
   * package imports are clean;
   * basic import tests pass.

3. [open] Slice 3 - Split and preserve CLI and GUI entrypoints.

   Depends on:

   * Slice 2

   Surfaces:

   * `copernican/cli.py`
   * `copernican/__main__.py`
   * `copernican/lib/gui/app.py`
   * `start.bat`
   * `start.sh`
   * `start.command`
   * `tests/test_start_scripts.py`
   * `tests/devcovenant/custom/policies/start_script_parity/`
   * `tests/devcovenant/custom/policies/start_script_guardrails/`
   * `pyproject.toml`
   * tests
   * docs

   Scope:

   * Create or finalize `copernican.cli:main` as the command-line entrypoint.
   * Preserve existing CLI behavior.
   * Preserve existing GUI behavior currently represented by
     `copernican_lib/gui/app.py`.
   * Move GUI implementation into a package-owned module.
   * Ensure CLI and GUI share runtime workflow code.
   * Ensure CLI and GUI do not duplicate execution logic.
   * Ensure `python -m copernican` invokes the CLI path.
   * Ensure the installed console script invokes the CLI path.
   * Remove the start scripts, their parity policy, and their linked tests.
   * Avoid hardcoded absolute checkout paths.
   * Avoid desktop assumptions.
   * Avoid rewriting whole files when a move, rename, or copy command can
     preserve the contents.

   Done when:

   * CLI starts from `copernican.cli:main`;
   * `python -m copernican` works;
   * GUI behavior still starts through a maintained entrypoint;
   * `start.bat`, `start.sh`, and `start.command` are deleted;
   * the start-script parity policy and tests are deleted;
   * root `copernican.py` is removed or reduced to a compatibility shim;
   * tests or smoke checks cover CLI and GUI startup boundaries where local
     dependencies allow.

4. [open] Slice 4 - Preserve model loading as filesystem data behavior.

   Depends on:

   * Slice 3

   Surfaces:

   * model loader code
   * `models/`
   * `pyproject.toml`
   * CLI arguments
   * GUI model-selection behavior
   * tests
   * docs

   Scope:

   * Keep `models/` at repository root.
   * Treat `models/` as example/reference model configuration data.
   * Do not package `models/` as importable Python code.
   * Ensure model paths may be absolute filesystem paths.
   * Ensure model paths may be relative filesystem paths.
   * Ensure model paths do not need to live under the repository root.
   * Accept `.yml` model extensions.
   * Accept `.yaml` model extensions.
   * Preserve existing model semantics.
   * Preserve existing model metadata behavior.
   * Ensure CLI model loading and GUI model loading use the same validation
     rules.
   * Ensure errors distinguish missing file, unsupported extension, unreadable
     file, and invalid model content.
   * Keep generated parser/metadata exceptions aligned with read-only directory
     rules where applicable.

   Done when:

   * `.yml` model files load;
   * `.yaml` model files load;
   * model files outside the repository load;
   * model files in root `models/` still load;
   * CLI and GUI use consistent model-loading rules;
   * invalid model path and invalid model content errors are clear;
   * tests cover both extension variants.

5. [open] Slice 5 - Preserve output shape and repair output base selection.

   Depends on:

   * Slice 4

   Surfaces:

   * output manager/runtime output code
   * existing run-output naming
   * `output/`
   * CLI options
   * GUI save/export behavior
   * tests
   * docs

   Scope:

   * Preserve the existing run directory shape.
   * Preserve existing run filename conventions.
   * Preserve existing manifest outputs.
   * Preserve existing parameter summary outputs.
   * Preserve existing posterior outputs.
   * Preserve existing plot outputs.
   * Preserve existing CSV outputs.
   * Preserve existing text outputs.
   * Do not introduce a new nested result layout.
   * Separate output shape from output base-directory selection.
   * Support explicit output directory selection from CLI.
   * Support GUI output directory selection where GUI behavior expects it.
   * Use root `output/` as the default local output base only when the current
     working directory is writable and appropriate.
   * Do not assume a desktop directory exists.
   * Do not write output into the installed package directory.
   * For headless or non-interactive usage, choose a deterministic writable
     output base instead of prompting.
   * For interactive usage, allow asking or confirming an output location only
     when a terminal or UI is available.
   * Use a persistent user-writable fallback before using a temporary
     directory.
   * Use temporary output only as a last resort.
   * Report any temporary output location clearly.
   * Ensure output exists on disk during the run.
   * Do not make final save/export the first moment real output exists.
   * Preserve enough output on disk that a completed run is not lost merely
     because a later export was not performed.

   Done when:

   * existing output structure is unchanged;
   * CLI can write output without desktop assumptions;
   * GUI can still save or direct output according to existing behavior;
   * output base directory is reported clearly;
   * non-interactive runs do not block on prompts;
   * installed/package-directory runs do not write into package code;
   * tests or smoke checks verify default writable output behavior;
   * tests or smoke checks verify explicit output directory behavior.

6. [open] Slice 6 - Rewrite package metadata for the standard shape.

   Depends on:

   * Slice 5

   Surfaces:

   * `pyproject.toml`
   * package metadata
   * package discovery
   * package data
   * console scripts
   * build tests

   Scope:

   * Set the project distribution name to `copernican`.
   * Remove top-level `py-modules = ["copernican"]`.
   * Use package discovery for `copernican` and `copernican.*`.
   * Exclude repository data and governance folders from package discovery.
   * Set console script entrypoint to `copernican.cli:main`.
   * Preserve package version behavior.
   * Preserve package data for `VERSION`.
   * Do not package root `models/` as code.
   * Do not package root `data/` as code.
   * Ensure package builds without multiple top-level package discovery errors.
   * Ensure editable/development installs and normal package builds agree on
     import behavior.

   Done when:

   * package metadata identifies the distribution as `copernican`;
   * package discovery includes only intended package code;
   * console script points at `copernican.cli:main`;
   * build succeeds;
   * import smoke tests pass;
   * no legacy `py-modules` root package ambiguity remains.

7. [open] Slice 7 - Realign DevCovenant dependency and license surfaces.

   Depends on:

   * Slice 6

   Surfaces:

* `devcovenant/custom/profiles/userproject/userproject.yaml`
* `devcovenant/config.yaml`
* `devcovenant/registry/registry.yaml`
* `requirements.in`
* `requirements.lock`
* `copernican/runtime-requirements.lock`
   * `copernican/licenses/`
   * `licenses/`
   * `devcovenant/runtime-requirements.lock`
   * `devcovenant/licenses/`
   * DevCovenant refresh outputs

   Scope:

   * Replace `copernican_lib` package-surface paths with `copernican`.
   * Keep DevCovenant inherited behavior inherited where possible.
   * Keep `userproject` overrides limited to repo-specific facts.
   * Keep `userproject` as the repo-specific source of truth for
     dependency hash targets.
   * Ensure the package runtime surface uses `pyproject.toml`.
   * Ensure the package runtime surface writes
     `copernican/runtime-requirements.lock`.
   * Ensure the package license surface writes
     `copernican/licenses/THIRD_PARTY_LICENSES.md`.
   * Ensure root workspace composes `requirements.in`,
     `devcovenant/runtime-requirements.lock`, and
     `copernican/runtime-requirements.lock`.
   * Ensure root license surface writes `licenses/THIRD_PARTY_LICENSES.md`.
   * Ensure DevCovenant runtime dependency and license surfaces stay separate.
   * Regenerate `devcovenant/config.yaml` and
     `devcovenant/registry/registry.yaml` from `userproject` before any
     dependency refresh.
   * Ensure generated config mirrors the profile-owned hash-target matrix
     before `refresh-force` is accepted.
   * Do not create a custom `python` profile to patch the package shape.
   * Do not manually edit generated lock hashes as the durable fix.
   * Run DevCovenant refresh after source paths are corrected.
   * Run dependency-management force refresh after generated metadata is
     correct.
   * Inspect generated config before accepting refreshed locks and licenses.

   Done when:

   * generated config points package runtime at
     `copernican/runtime-requirements.lock`;
   * generated config points package licenses at `copernican/licenses`;
   * generated config and registry mirror the userproject hash-target
     matrix;
   * root workspace includes the package runtime lock;
   * root workspace includes the DevCovenant runtime lock;
   * package dependency lock refreshes through the normal force-refresh path
     after generated config is rebuilt from `userproject`;
   * root dependency lock refreshes through the normal force-refresh path;
   * package license report is populated;
   * root license report is populated;
   * no license surface collapses to only DevCovenant dependencies;
   * no manual hash patch is needed as the durable fix.

8. [open] Slice 8 - Rebuild tests, docs, and changelog around migration.

   Depends on:

   * Slice 7

   Surfaces:

   * tests
   * `README.md`
   * `AGENTS.md`
   * `CONTRIBUTING.md`
   * `CHANGELOG.md`
   * `PLAN.md`
   * model-loading documentation
   * GUI documentation
   * output documentation
   * legacy test cleanup

   Scope:

   * Update imports in tests.
   * Add or repair CLI smoke tests.
   * Add or repair GUI import/startup smoke tests where local dependencies
     allow.
   * Add model-loading tests for `.yml`.
   * Add model-loading tests for `.yaml`.
   * Add model-loading tests for model paths outside repo `models/`.
   * Add output-base-directory tests without changing output shape.
   * Remove remnant tests such as `test_version_*.py`, start-script parity
     tests, and other old-shape coverage.
   * Update docs to describe `copernican/` as the package.
   * Update docs to describe packaged datasets and parsers.
   * Update docs to describe root `models/` as model examples/config data.
   * Update docs to describe CLI entrypoints.
   * Update docs to describe GUI entrypoints.
   * Update docs to describe safe output base-directory behavior.
   * Record the migration in `CHANGELOG.md`.
   * Keep public docs free of private implementation history beyond necessary
     migration facts.

   Done when:

   * tests import the migrated package;
   * CLI smoke checks pass;
   * GUI boundary checks pass where local dependencies allow;
   * model-loading checks pass for `.yml`;
   * model-loading checks pass for `.yaml`;
   * model-loading checks pass for root models;
   * legacy tests tied to the old shape are removed or rewritten;
   * output behavior checks pass without output-shape redesign;
   * docs match the migrated structure;
   * changelog records the completed migration.

9. [open] Slice 9 - Validate local, installed, GUI, and headless operation.

   Depends on:

   * Slice 8

   Surfaces:

   * local checkout runtime
   * installed package runtime
   * CLI
   * GUI
   * model loading
   * packaged datasets and parsers
   * output writing
   * DevCovenant gates
   * dependency locks
   * license reports

   Scope:

   * Validate local checkout execution.
   * Validate installed package execution.
   * Validate `python -m copernican`.
   * Validate console script execution.
   * Validate GUI startup/import path.
   * Validate packaged dataset and parser loading from resources.
   * Validate model loading from root `models/`.
   * Validate model loading from an arbitrary external path.
   * Validate `.yml`.
   * Validate `.yaml`.
   * Validate output creation in an explicit output directory.
   * Validate output creation from a writable checkout.
   * Validate non-desktop/headless output behavior.
* Validate DevCovenant refresh.
* Validate dependency-management force refresh.
* Verify generated config and registry mirror the userproject hash-target
  matrix.
* Validate lock installation in a clean environment where local platform
  constraints allow.
   * Document any platform-specific limitations honestly.

   Done when:

   * local CLI run works;
   * installed CLI run works;
   * GUI entrypoint is preserved;
   * packaged datasets and parsers load from package resources;
   * model loading works from expected filesystem locations;
   * output is written to a real, reported, writable location;
   * dependency locks are refreshed through DevCovenant;
   * license reports are complete;
   * remaining limitations are documented.

## Validation Routine

Run the validation routine after each completed slice when relevant.

Minimum validation:

* inspect `git status --short`;
* run targeted tests for touched code;
* run import smoke checks for `copernican`;
* run CLI startup smoke checks;
* run GUI import/startup smoke checks where local dependencies allow;
* check that packaged datasets and parsers are reachable from the package;
* check that root `models/` remains outside the package;
* check that `.yml` and `.yaml` model paths are accepted;
* check that existing output shape is preserved;
* check that generated output goes to a real writable path;
* check that no dependency artifact was accidentally collapsed;
* check that no license artifact was accidentally collapsed;
* update `CHANGELOG.md` when behavior, structure, docs, or governance changes.

Package validation after package-layout slices:

* build the package;
* verify package discovery includes `copernican` and `copernican.*`;
* verify packaged datasets and parsers are included;
* verify package discovery does not include root `models/` as code;
* verify package discovery does not include root `data/` as code;
* verify `python -m copernican` works;
* verify the console script entrypoint works.

DevCovenant validation after dependency-surface slices:

* run DevCovenant refresh;
* inspect generated dependency-management surfaces;
* verify package runtime lock path is `copernican/runtime-requirements.lock`;
* verify package license path is
  `copernican/licenses/THIRD_PARTY_LICENSES.md`;
* verify root workspace includes `requirements.in`;
* verify root workspace includes `devcovenant/runtime-requirements.lock`;
* verify root workspace includes `copernican/runtime-requirements.lock`;
* run dependency-management force refresh;
* verify root and package license reports are populated;
* verify lock installation behavior in a clean environment where local platform
  constraints allow.

Completion validation:

* all migrated imports resolve;
* CLI and GUI entrypoints are preserved;
* packaged datasets and parsers resolve from package resources;
* model loading is filesystem-based and extension-correct;
* output shape is unchanged;
* output base-directory behavior is safe for local and headless use;
* legacy start scripts and their parity policy are gone;
* DevCovenant dependency and license surfaces are aligned with the package;
* docs and changelog match the migrated repository structure.
