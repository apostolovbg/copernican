# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-06-03
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

Copernican started as a single script and has now moved into a
package-centered layout. The old root `data/` tree has been deleted;
bundled datasets and parsers now live under `copernican/datasets/`;
`copernican/lib/` is the legacy surface being retired; and
`copernican.py`, the old start scripts, and the script-era policy shims
are migration residue. DevCovenant dependency and license surfaces must
now follow the package model, not the old script model.

The migration goal is a standard package layout with one coherent runtime,
one coherent dataset home, stable output behavior, and policy surfaces
that match the package tree. The work should be grouped into a few large,
reviewable slices rather than many small fragments.

The current target state is simple:

* the import package should be `copernican/`;
* the distribution name should be `copernican`;
* the CLI command should be `copernican`;
* runtime helpers should live inside the package, with `copernican/lib/`
  holding shared code;
* engine code should move inside `copernican/`;
* curated datasets and parsers should ship inside the package;
* bundled model assets should move into `copernican/models/`;
* root `models/` should stop being treated as a live model home once the
  package move is complete;
* validation helpers should move into `copernican/validation/`;
* the high-resolution docs logo should live in `docs/logo.png` and
  `copernican/docs/logo.png`, while the GUI-only small logo should live
  in `copernican/img/logo_small.png`;
* bundled RNG mini-games should move into `copernican/rng_minigames/`
  and keep loading from the GUI after the move;
* the root README, package README, root docs, and `copernican/docs/`
  should all stay identical for now, but the content should grow into
  the same longform DevCovenant-style documentation standard with
  explicit navigation, meaningful section depth, practical rules,
  cross-links, and non-laconic explanations;
* output shape should stay stable while the root `output/` tree is
  removed and the default writable output home moves to
  `~/copernican_output`;
* DevCovenant surfaces should be realigned after the package shape is
  stable.

The old `data/` tree is no longer the live dataset home. The package tree
is. The plan should therefore describe the remaining work in coarse,
coherent slices rather than in many narrowly separated ones.

## Overview

* Copernican is a Python toolkit for evaluating cosmological models
  against SNe Ia, BAO, and CMB observations.
* The repository is now mid-migration from a script-centered layout to a
  package-centered layout.
* `copernican/` is the target import package.
* `copernican/lib/` is legacy surface area that should be retired.
* `copernican/engines/` is the canonical packaged home for runtime
  engines.
* `copernican/datasets/` is the canonical bundled dataset home.
* `copernican/models/` is the canonical bundled model home.
* root `models/` is migration residue, not a live long-term home.
* `copernican/rng_minigames/` is the canonical packaged home for bundled
  RNG mini-games.
* `copernican/validation/` is the canonical packaged home for validation
  helpers.
* `docs/logo.png` and `copernican/docs/logo.png` should stay identical,
  and the GUI should load the small logo from `copernican/img/logo_small.png`.
* The README/docs set should not stay overview-only; it should mirror the
  sibling DevCovenant documentation standard with TOCs or equivalent
  navigation maps, substantial sectioned explanations, practical rules,
  and explicit cross-links, while keeping the root and package copies
  identical for now.
* Existing output shape should be preserved.
* the default writable output home should be `~/copernican_output`.
* DevCovenant dependency and license surfaces should be aligned with the
  package layout once the package shape is stable.
* The remaining work should be grouped into a small number of large,
  reviewable slices rather than many thin fragments.

## How Slices Are Executed

* Each slice means a complete implementation pass, not a note.
* Each slice must leave code, tests, docs, and changelog evidence where
  behavior changed.
* Do not mark a slice done unless the relevant checks support it.
* Do not treat contract-only behavior as runtime completion.
* Do not redesign scientific output content.
* Do not redesign result file shape.
* Move `engines/`, `models/`, and `validation/` into the package.
* Keep model loading accepting both `.yml` and `.yaml`, and keep CLI and
  GUI on the same loading rules for external model paths.
* Do not make model loading depend on repository-relative paths.
* Do not make output depend on a desktop path.
* Keep output selection optional and default to `~/copernican_output`.
* Do not write runtime output into the installed package directory.
* Do not manually patch generated lock hashes as the durable fix.
* Do not create a copied custom `python` profile to compensate for package
  layout problems.
* Prefer copy, rename, or move commands when they can preserve the change.
* Avoid rewriting whole files when a path-preserving command can do the
  job.
* Treat token conservation as a first-class requirement.
* Keep related runtime, docs, changelog, and governance changes together
  when they belong to the same migration step.
* Keep generated artifacts generated.
* Keep repository-specific DevCovenant profile overrides narrow and
  factual.
* Preserve existing run-output naming and file layout.
* Use `CHANGELOG.md` to record slice outcomes when behavior, documentation,
  or governance changes.
* Use the configured local governance workflow around each completed slice.
* Keep every slice small enough to review, but complete enough to run.
* Remove obsolete start scripts, parity policy, and legacy tests rather
  than preserving them.

## Execution Slices

1. [closed] Slice 1 - Rebaseline the migrated Copernican tree.

   Depends on:

   * current repository state

   Status:

   * Closed. The current migrated baseline has already been established,
     and this slice now serves as the recorded inventory of the starting
     point for the remaining work.

   Surfaces:

   * `copernican/`
   * `copernican/lib/`
   * `models/`
   * `validation/`
   * `output/`
   * root docs
   * package docs
   * DevCovenant generated surfaces

   Scope:

   * Confirm the current package and runtime boundaries.
   * Confirm that `models/` and `validation/` are part of the baseline
     migration residue that later slices must move into the package.
   * Confirm the current output shape that must be preserved.
   * Confirm that `copernican/datasets/` is the live dataset home.
   * Confirm the current docs mirror and governance state.
   * Record the baseline facts that later slices depend on.

   Done when:

   * the migrated starting point is documented;
   * the package/data/runtime boundaries are explicit;
   * later slices can proceed without re-discovering the baseline.

2. [closed] Slice 2 - Finish the package/runtime and bundled-asset migration.

   Depends on:

   * Slice 1

   Surfaces:

   * `copernican/`
   * `copernican/lib/`
   * `copernican/datasets/`
   * `copernican/rng_minigames/`
   * `copernican/__main__.py`
   * `copernican/img/logo_small.png`
   * `docs/logo.png`
   * `copernican/docs/logo.png`
   * imports
   * tests

   Scope:

   * Finalize the `copernican` package as the real import package.
   * Move shared runtime helpers into `copernican/lib/`.
   * Keep any package-root runtime modules that still belong at the top of
     `copernican/`.
   * Move curated datasets and parser modules into `copernican/datasets/`.
   * Keep bundled datasets and parsers loadable as package resources.
   * Keep the docs logo in `docs/logo.png` and `copernican/docs/logo.png`
     while the GUI loads its small asset from `copernican/img/logo_small.png`.
   * Move bundled RNG mini-games into `copernican/rng_minigames/` and
     update the GUI code that loads them so they still render after the
     refactor.
   * Move the mirrored RNG mini-game tests alongside the packaged assets
     so the test tree follows the new package layout.
   * Keep the parser/metadata carve-out working on the package dataset tree.
   * Keep the existing output shape stable.
   * Keep external dataset discovery unsupported.
   * Remove root-launcher dependence from the runtime path.
   * Keep package imports explicit and package-relative where appropriate.
   * Rework the root README and mirrored package README into a full
     DevCovenant-style front door with a docs map or table of contents,
     substantial sectioned explanations, practical rules, and explicit
     cross-links.
   * Rework the root docs and `copernican/docs/` as identical mirrored
     manuals for now, using the same documentation standard as
     DevCovenant: TOCs or equivalent navigation maps, longform sections,
     ownership maps, recovery notes, and practical rules instead of
     overview-only summaries.
   * Remove any stale documentation-policy page or other dead doc surface
     that no longer serves the mirrored manual if it remains in the tree.

   Done when:

   * runtime code imports through `copernican` and `copernican.lib`;
   * curated datasets and parsers ship from the package tree;
   * the GUI logo still loads from its new package path;
   * bundled RNG mini-games still load from their new package path;
   * mirrored RNG mini-game tests track the packaged asset layout;
   * the README/docs set is mirrored and matches the DevCovenant-style
     manual standard while remaining identical between root and package
     copies;
   * package imports are clean;
   * basic import tests pass.

3. [closed] Slice 3 - Consolidate packaging, DevCovenant, docs,
   tests, and validation.

   Depends on:

   * Slice 2

   Surfaces:

   * `pyproject.toml`
   * `devcovenant/custom/profiles/userproject/userproject.yaml`
   * `devcovenant/config.yaml`
   * `devcovenant/registry/registry.yaml`
   * `requirements.in`
   * `requirements.lock`
   * `copernican/runtime-requirements.lock`
   * `engines/`
   * `models/`
   * `validation/`
   * `licenses/`
   * `copernican/licenses/`
   * docs
   * `ABOUT.md`
   * `SECURITY.md`
   * `SUPPORT.md`
   * `CITATION.cff`
   * `copernican/ABOUT.md`
   * `copernican/SECURITY.md`
   * `copernican/SUPPORT.md`
   * `copernican/CITATION.cff`
   * tests
   * `CHANGELOG.md`
   * final validation

   Scope:

   * Rewrite package metadata for the standard package shape.
   * Remove top-level `py-modules = ["copernican"]`.
   * Use package discovery for `copernican` and `copernican.*`.
   * Keep `VERSION` as package data.
   * Keep editable installs aligned with normal builds.
   * Move bundled engines into `copernican/`.
   * Move bundled models into `copernican/models/` and retire the root
     `models/` surface.
   * Move validation helpers into `copernican/validation/`.
   * Realign the dependency and license surfaces with the package layout.
   * Remove legacy start scripts and parity-policy remnants.
   * Rebuild the tests around the migrated package surface.
   * Update the docs and mirrored docs to match the package layout.
   * Extend the package-doc-sync mirror surface to keep
     `ABOUT.md`, `SECURITY.md`, `SUPPORT.md`, and `CITATION.cff`
     identical between the root and `copernican/`.
   * Broaden the user-visible and doc-quality surfaces so the new package
     root docs are version-synced and checked.
   * Route runtime, security, support, and citation changes to the
     appropriate top-level doc targets through doc routes.
   * Record the completed migration in `CHANGELOG.md`.
   * Run the final verification and runtime checks.
   * Do not split docs, tests, changelog, and final validation into separate
     mini-slices once the package shape is stable.

   Done when:

   * package metadata identifies the distribution as `copernican`;
   * DevCovenant surfaces mirror the package layout;
   * docs and mirrored docs match the migrated structure;
   * the root package-doc mirror includes `ABOUT.md`, `SECURITY.md`,
     `SUPPORT.md`, and `CITATION.cff` under `copernican/`;
   * doc-quality, user-visible, and last-updated coverage includes the
     added package-root docs;
   * doc routes keep the README, ABOUT, SECURITY, SUPPORT, and citation
     surfaces tied to the correct code changes;
   * tests cover the migrated package surface;
   * bundled engines, models, and validation helpers ship from the
     package tree;
   * changelog records the completed migration;
   * final verification and runtime checks pass.

4. [open] Slice 4 - Finish runtime behavior, model/output policy, and
   legacy cleanup.

   Depends on:

   * Slice 3

   Surfaces:

   * `models/`
   * `output/`
   * CLI
   * GUI
   * imports
   * tests
   * docs
   * `devcovenant/custom/profiles/userproject/userproject.yaml`

   Scope:

   * Accept bundled models from the package tree and arbitrary external
     filesystem paths.
   * Accept both `.yml` and `.yaml` model files through the same CLI and
     GUI loading rules.
   * Make explicit output selection optional and default to the per-user
     `~/copernican_output` home on every supported OS.
   * Remove the root `output/` tree as a live target.
   * Preserve the existing run-output shape and named artifacts.
   * Move bundled models fully in-package and retire remaining root-model
     assumptions.
   * Clean up the legacy `cosmo_` naming in model and parser code and in
     the matching DevCovenant profile references.
   * Remove stale legacy tests that still reflect the old shape, including
     `test_version_*.py`, start-script parity residue, and similar old
     coverage if it remains.
   * Finish any remaining runtime docs for model loading, output policy,
     CLI, and GUI behavior.

   Done when:

   * external model paths work;
   * `.yml` and `.yaml` model paths work through CLI and GUI;
   * output lands in `~/copernican_output` by default;
   * root `output/` is no longer a live target;
   * bundled models are in-package;
   * `cosmo_` runtime naming residue is gone where required;
   * stale legacy tests are removed or updated;
   * runtime docs match the runtime behavior.

5. [open] Slice 5 - Final validation and regression confirmation.

   Depends on:

   * Slice 4

   Surfaces:

   * validation
   * tests
   * CLI
   * GUI
   * docs

   Scope:

   * Validate local checkout execution.
   * Validate installed-package execution.
   * Validate `python -m copernican`.
   * Validate console-script execution.
   * Validate GUI startup and import-path behavior.
   * Validate the final model-loading and output-policy behavior.
   * Validate the final docs, changelog, and governance state.

   Done when:

   * the full runtime matrix passes;
   * the model and output behavior is confirmed on the final package
     shape;
   * docs and changelog are consistent with the completed migration.

## Validation Routine

Run the validation routine after each completed slice when relevant.

Minimum validation:

* inspect `git status --short`;
* run targeted tests for touched code;
* run import smoke checks for `copernican`;
* run CLI startup smoke checks;
* run GUI import/startup smoke checks where local dependencies allow;
* check that packaged datasets, models, validation helpers, and parsers are
  reachable from the package;
* check that root `models/` and `validation/` are no longer live code/data
  roots;
* check that `.yml` and `.yaml` model paths are accepted;
* check that external model paths are accepted;
* check that existing output shape is preserved;
* check that generated output goes to a real writable path;
* check that generated output defaults to the per-user
  `~/copernican_output` location;
* check that no dependency artifact was accidentally collapsed;
* check that no license artifact was accidentally collapsed;
* update `CHANGELOG.md` when behavior, structure, docs, or governance changes.

Package validation after package-layout slices:

* build the package;
* verify package discovery includes `copernican` and `copernican.*`;
* verify packaged datasets and parsers are included;
* verify package discovery exposes packaged models and validation helpers;
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
* packaged datasets, models, and validation helpers resolve from package
  resources;
* model loading is package-based, filesystem-based, and extension-correct;
* model loading accepts external paths and both model file extensions;
* output shape is unchanged;
* output base-directory behavior is safe for local and headless use and
  defaults to `~/copernican_output`;
* legacy start scripts and their parity policy are gone;
* DevCovenant dependency and license surfaces are aligned with the package;
* docs and changelog match the migrated repository structure.
