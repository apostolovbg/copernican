# Custom Extensions
**Last Updated:** 2026-04-27
**Project Version:** 1.0.1b5

## Table of Contents
- [Overview](#overview)
- [Policy Extensions](#policy-extensions)
- [Profile Extensions](#profile-extensions)
- [Override Rules](#override-rules)
- [Workflow](#workflow)

## Overview
`devcovenant/custom/` is the repository-owned extension surface.

Use this directory to add or override behavior without editing shipped builtin
files. Everything here is project-specific and stays under repository control.
If you are choosing a first custom path, start with
[customization.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/customization.md).
When the repository is shadowing a builtin policy or profile, use
`devcovenant custom --policy <policy-id> --do` or
`devcovenant custom --profile <profile-name> --do` to create the repo-owned
copy. When the selected builtin ships blueprints, the command also
materializes mirrored tests under `tests/devcovenant/custom/**`; starter
templates like `userproject` simply copy and refresh without a mirror.
Use `--undo` to remove that copied tree and return to the builtin default.
The command follows the same managed-environment resolution as the other
lifecycle commands, so it works from whichever interpreter layout the
repository has declared.

The usual downstream shape keeps the always-active builtin `devcovuser`
baseline in place and then adds repo-owned custom profiles for repository-
specific behavior. `userproject` is the copy-ready bootstrap template for
that first repo-owned profile, and Python repositories often add a custom
`python` or `python_venv` profile for environment ownership. That split keeps
builtin layers generic and puts the real project law in repository-owned
metadata.

## Path Summary
Use the profile path when the repository needs reusable repo shape, assets,
workflow additions, environment declarations, or translator declarations.
Use the policy path when the repository needs new rule logic, different
enforcement, or a shadow of builtin rule behavior.

The materialization rules stay simple:
- `devcovenant custom --profile <name> --do` copies the builtin profile into
  the custom tree
- `devcovenant custom --policy <id> --do` copies the builtin policy into the
  custom tree
- mirrored tests appear only when the builtin source ships blueprints
- `userproject` is a starter profile, so it copies without a test mirror
- `--undo` removes the copied tree and any mirror

If you need the guided first custom path, start with
[customization.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/customization.md).

## Policy Extensions
Place custom policies under:
- `devcovenant/custom/policies/<policy-id>/`

Typical custom policy files:
- `<policy-id>.py` for check logic
- `<policy-id>.yaml` for descriptor text and metadata defaults
- `test_blueprints.yaml` when the policy was copied from a builtin source and
  needs a packaged test blueprint mirror
- optional `autofix/` modules
- optional `assets/` templates referenced by profiles

## Profile Extensions
Place custom profiles under:
- `devcovenant/custom/profiles/<profile-name>/`

Typical profile files:
- `<profile-name>.yaml` manifest
- `test_blueprints.yaml` when the profile was copied from a builtin source
  and needs a packaged test blueprint mirror
- optional `assets/` templates
- optional `<profile-name>_translator.py` when profile category is language

## Override Rules
- Same-id custom policy overrides same-id builtin policy.
- Same-name custom profile overrides same-name builtin profile.
- Policy activation still comes only from `config.policy_state`.
- Profiles contribute overlays, assets, selectors, hooks, and translators.

Custom changes should keep contract compatibility unless a contract change is
intentional and documented in `SPEC.md`.

## Workflow
1. Implement custom policy/profile changes in this directory, or use
   `devcovenant custom --policy <policy-id> --do` or
   `devcovenant custom --profile <profile-name> --do` to promote a builtin
   copy into this tree.
2. Keep mirrored tests under `tests/devcovenant/custom/...` aligned with the
   copied builtin blueprint when a builtin source is shadowed.
3. Update relevant docs and maps (`POLICY_MAP.md`, `PROFILE_MAP.md`).
4. Run `devcovenant refresh` after manifest or descriptor changes.
5. Run full gate sequence before handing off changes:
   `gate --open` -> `gate --verify` (rerun until clean) ->
   `run` -> `gate --close`.
