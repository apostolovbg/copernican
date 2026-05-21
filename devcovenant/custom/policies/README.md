# Custom Policies
**Last Updated:** 2026-04-27
**Project Version:** 1.0.1b5

## Table of Contents
- [Overview](#overview)
- [Directory Layout](#directory-layout)
- [Override Semantics](#override-semantics)
- [Metadata and Assets](#metadata-and-assets)
- [Workflow](#workflow)

## Overview
Custom policies live under `devcovenant/custom/policies/<policy-id>/`.
They are repository-owned extensions and can fully replace builtin policies
with matching IDs.
When a builtin policy is being shadowed and it ships `test_blueprints.yaml`,
`devcovenant custom --policy <policy-id> --do` copies the builtin tree into
this directory and materializes the mirror tests under
`tests/devcovenant/custom/policies/<policy-id>/`.
`--undo` removes that repo-owned copy and any mirror again.

## Directory Layout
Expected files mirror builtin policy structure:
- `<policy-id>.yaml` descriptor
- `<policy-id>.py` check script
- optional `test_blueprints.yaml` when the directory is a copied builtin
  policy shadow and ships packaged test descriptors
- optional `autofix/*.py`
- optional `assets/` templates

## Override Semantics
When a custom policy ID matches a builtin policy ID:
- custom script is loaded
- builtin script is suppressed
- builtin autofix helpers for that policy are suppressed

Activation remains config-driven via `policy_state`.

## Metadata and Assets
Descriptor metadata is merged with profile/config layers in the standard
precedence order.
The repo-local `builtin-test-blueprint-sync` policy reads
`mirror_roots`, `blueprint_directories`, and `blueprint_name` metadata from
its own descriptor, so the mirrored-test surface stays declarative instead
of embedding path families in code.

Policy-owned asset files should be declared via profile manifests so refresh
can materialize them consistently. Avoid hidden side-channel file writes from
policy scripts when declarative assets are sufficient.

## Workflow
1. Change descriptor, script, and `test_blueprints.yaml` together when the
   policy came from a builtin copy.
2. Update mirrored tests under `tests/devcovenant/custom/policies/...` when
   a builtin policy is shadowed.
3. Run `devcovenant custom --policy <policy-id> --do` to refresh the copied
   tree and its test mirror.
4. Use `--undo` to remove the copied tree and mirror.
5. Run `devcovenant refresh` after descriptor/profile updates.
6. Run full gate sequence:
   `gate --open` -> `gate --verify` (rerun until clean) ->
   `run` -> `gate --close`.
