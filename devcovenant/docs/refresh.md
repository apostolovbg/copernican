# Refresh Behavior
**Last Updated:** 2026-04-27
**Project Version:** 1.0.1b5

## Overview
`refresh` rebuilds the DevCovenant-owned files that belong to the repository's
saved setup.
It updates tracked outputs, but it does not invent fake live session state.

Use this page when you need to answer simple questions like these:
- which files does refresh rebuild?
- which file bodies are preserved?
- which template or descriptor owns a generated document?

## What Refresh Updates
Refresh can rebuild:
- tracked registry state
- generated config sections
- managed policy output in `AGENTS.md`
- generated workflow and tooling files
- managed docs selected through `doc_assets`
- generated `.gitignore` and pre-commit files

If profiles, descriptors, or managed templates changed, refresh is the point
where those changes become real in the repository.
Generated GitHub workflow output appears only when an active profile owns the
base workflow template, such as the builtin `github` profile, or when
`config.ci_and_test.overrides` takes full local ownership.
That same refresh pass also rebuilds tracked policy metadata from the
active resolved script path, so same-id custom policy overrides are
recorded as `custom: true` in generated policy state even while the
builtin implementation remains present on disk.
For dependency-maintenance surfaces, the refresh-owned runtime walks
supported `-r`/`--requirement` includes when it builds license
inventories and refreshes provider lock surfaces before surfaces that
compose those lockfiles.

## When Refresh Runs
A full refresh runs in:
- `devcovenant refresh`
- `devcovenant deploy`
- `devcovenant upgrade`
- gate-owned refresh or autofix paths during the normal workflow

`check` is read-only and does not run startup refresh.

## Managed Docs
Managed docs are descriptor-driven.
The managed-doc code owns:
- descriptor loading
- descriptor validation
- header rendering
- managed block rendering
- adoption of compatible seeded docs
- replacement of placeholder or seed docs that match the import rules

That keeps document behavior in one place instead of scattering it across many
commands.

## Preservation Rules
The practical preservation rules are:
- missing doc: may be created
- empty doc: may be replaced
- one-line doc: may be replaced
- otherwise: only managed headers and managed blocks should change

That is how DevCovenant can manage docs without treating ordinary human prose
as disposable.

## Descriptor Model
A managed-doc descriptor defines the target path, identity headers, managed
block content, and body template.
Some docs also opt into project-governance headers.

The descriptor owns the document structure.
The live file owns preserved authored content outside the managed areas.

Some descriptors intentionally keep the managed block empty.
That is the rule for `README.md` and `devcovenant/README.md` in this
repository: the `<!-- DEVCOV:BEGIN -->` / `<!-- DEVCOV:END -->` block stays
present but empty so DevCovenant does not inject runtime prose at the top of
user-facing README pages.

The same rendering machinery is reused by `devcovenant asset`.
That command does not own a second template engine.
It renders plain profile assets through the same shared asset renderer that
refresh uses, and it renders descriptor-backed docs through the same
managed-doc runtime, but writes the result as a Desktop copy instead of the
repository-managed target path.

## Custom Managed Docs
Profiles can add managed docs through their asset trees.
The active model is:
- the global profile contributes the base descriptor set
- active profiles can add new managed-doc targets
- active profiles can override a global descriptor by reusing the same target
  path
- later active profiles win over earlier ones for the same target path

That is how a shared custom profile can reuse the shared base while still
replacing
individual docs such as `SECURITY.md`, `PRIVACY.md`, or `SUPPORT.md`
without pushing repository-specific prose into package docs.

## Failure Modes
Refresh should fail clearly when a managed-doc descriptor is invalid.
It should not guess what a broken descriptor meant.

Common failures are:
- missing descriptor for an enabled managed doc
- invalid descriptor shape
- broken target or template mapping

## Practical Rule
When refresh behavior seems confusing, ask two questions first:
1. which descriptor owns this output?
2. is the file supposed to be preserved, regenerated, or adopted?

Most refresh confusion gets much simpler once those ownership questions are
answered.
