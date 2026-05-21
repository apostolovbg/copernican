# Builtin Profiles
**Last Updated:** 2026-04-27
**Project Version:** 1.0.1b5

## Table of Contents
- [Overview](#overview)
- [Profile Responsibilities](#profile-responsibilities)
- [Manifest Schema](#manifest-schema)
- [Translator Declarations](#translator-declarations)
- [Asset Materialization Rules](#asset-materialization-rules)
- [Workflow](#workflow)

## Overview
Builtin profiles are shipped under `devcovenant/builtin/profiles/<name>/`.

Profiles are metadata and asset providers. They do not activate policies.
Policy activation authority is `config.policy_state`.
This README documents folder contracts and ownership boundaries, not the
current profile inventory. For the active builtin/custom profile catalog, use
`PROFILE_MAP.md`.

Builtin profiles can also carry a sibling `test_blueprints.yaml` descriptor
when the shipped profile tests should remain package data instead of a live
discoverable `tests/**` tree.
Those blueprints are the packaged form of the shipped tests and are
materialized into `tests/devcovenant/custom/profiles/<name>/` when a
repository shadows the builtin profile with
`devcovenant custom --profile <profile-name> --do`.

## Profile Responsibilities
Profiles may provide:
- metadata overlays
- selector metadata
- workflow runs
- asset templates
- pre-commit hook fragments
- CI fragments
- translator declarations (language profiles only)

DevCovenant also ships one copy-ready bootstrap template profile:
- `userproject`

Copy `devcovenant/builtin/profiles/userproject/` to
`devcovenant/custom/profiles/userproject/` when a repository needs its first
custom profile.
It is not meant to be activated directly.

Any active profile category may contribute metadata overlays, and language or
stack profiles may declare workflow runs through `workflow_runs`.

## Manifest Schema
Each profile directory contains a `<name>.yaml` manifest.
Common keys include:
- `profile`
- `category`
- `suffixes`
- `ignore_dirs`
- optional `gitignore_fragments`
- optional `gitignore_template` (global baseline template)
- optional `ci_and_test_template` (base workflow template on an opt-in CI
  owner profile such as `github`)
- optional `ci_and_test` (workflow fragment overlay)
- `policy_overlays`
- `assets`
- `pre_commit`
- optional `workflow_runs`
- optional `translators`

Custom profiles with the same profile name override builtin profiles.

## Blueprint Contract
When a builtin profile ships tests, the directory may also include
`test_blueprints.yaml`.
That file stores repo-relative paths and serialized file contents for the
checked-in builtin test tree so the package can ship a descriptor instead of
the live tree itself.
The checked-in builtin tests remain under
`tests/devcovenant/builtin/profiles/<name>/` for verification.
When the profile is copied into `devcovenant/custom/profiles/<name>/`,
`devcovenant custom --profile <name> --do` materializes the mirrored tests
under `tests/devcovenant/custom/profiles/<name>/`.
`--undo` removes the repo-owned copy and its mirrored tests again.

## Translator Declarations
Only language profiles declare translators.
Declaration fields include:
- `id`
- `extensions`
- `can_handle` strategy and entrypoint
- `translate` strategy and entrypoint

Translator entrypoint paths are validated as profile-contained paths.

## Asset Materialization Rules
During deploy, upgrade, or refresh:
- missing assets are created from templates
- existing non-one-line files are preserved
- managed blocks are refreshed where document contracts require it

Generated assets include `.pre-commit-config.yaml`, `.gitignore`, and managed
docs selected by active profile metadata.
`.gitignore` is generated from global template fragments plus per-profile
manifest fragments and config overlays; profiles do not ship `.gitignore`
asset files.
Global template source:
`devcovenant/builtin/profiles/global/assets/gitignore.yaml`.
`.github/workflows/ci.yml` is generated when an active profile owns
`ci_and_test_template` such as the builtin `github` profile. Active-profile
CI fragments and config overlays/overrides are then merged on top. The
builtin `github` base bootstraps DevCovenant from the shipped
`devcovenant/runtime-requirements.lock` so it does not assume the
repository's own dependency files belong to DevCovenant. The package also
ships `devcovenant/licenses/**`, which contains DevCovenant's packaged
license files.

## Workflow
1. Edit the profile manifest, assets, and `test_blueprints.yaml` together.
2. Keep the checked-in builtin test tree aligned with the blueprint.
3. Use `devcovenant custom --profile <name> --do` to create or refresh a
   repo-owned shadow copy and mirrored tests.
4. Use `--undo` to remove the copied profile tree and mirrored tests.
5. Run `devcovenant refresh`.
6. Run `devcovenant gate --open`.
7. Run `devcovenant gate --verify` until clean.
8. Verify with `devcovenant run`.
9. Finalize with `devcovenant gate --close`.
