# Project Governance
**Last Updated:** 2026-04-27
**Project Version:** 1.0.1b5

## Overview
This document is the normative home for the `project-governance` contract.
Keep `devcovenant/docs/contracts.md` nearby when you need the stable document
map for the rest of the package surfaces.

`project-governance` is where a repository states what it is called, what
lifecycle stage it is in, how actively it is still changing, and what
compatibility promise it is making.
Open this page when you need to answer those public identity questions
deliberately instead of letting package metadata or README wording drift into
becoming the source of truth by accident.
It is not a packaging afterthought and it is not derived from `pyproject.toml`.
Other public surfaces render from this metadata.

## Core Fields
`project_name`: any non-empty string. Default seed value: `Project Name`.

`project_description`: any non-empty string. Default seed value:
`Describe the project: what it does, who it helps, and
what problem it solves.`

`stage`: one value from `allowed_stages`. The default allowed set is
`prototype`, `alpha`, `beta`, `stable`, `deprecated`, `archived`.

`maintenance_stance`: one value from `allowed_maintenance_stances`. The
default allowed set is `active`, `maintenance`, `frozen`, `sunset`.

`compatibility_policy`: closed enum. Allowed values are
`backward-compatible`, `breaking-allowed`, `forward-only`, and
`unspecified`.

- `backward-compatible`: preserve the current public contract. Add
  compatibility bridges only when they are intentional, documented, and
  tested.
- `breaking-allowed`: compatibility is optional. Do not imply support you do
  not intend to keep.
- `forward-only`: keep one active contract and fail explicitly on
  unsupported shapes instead of carrying compatibility branches.
- `unspecified`: no compatibility promise is implied yet. Make contract
  changes explicit before code or docs start depending on them.

`versioning_mode`: `versioned` or `unversioned`.

`codename`: optional free-form string.

`build_identity`: optional free-form string.

`unversioned_label`: any non-empty string used as the displayed project
version in unversioned mode. Default: `Unversioned`.

`unreleased_heading`: any non-empty string used as the required top visible
changelog heading in unversioned mode. Default: `## Unreleased`.

`changelog_file`: any non-empty repo-relative path string. Default:
`CHANGELOG.md`.

`allowed_stages`: non-empty list of allowed stage tokens. Repositories may
tighten or rename this list, but `stage` must always be one of its entries.

`allowed_maintenance_stances`: non-empty list of allowed stance tokens.
Repositories may tighten or rename this list, but `maintenance_stance` must
always be one of its entries.

## Names And Path Tokens
`project_name` is the canonical public/project identity string.
Keep the real project or distribution name there even when another tool needs
a normalized path token.

DevCovenant derives normalized path tokens such as `{{ PROJECT_NAME_PATH }}`
where package-safe paths are needed.
That means a repository can keep `webcam-micro` as `project_name` while still
using `webcam_micro` for Python package paths or other normalized filesystem
surfaces.
Do not force Python import-package spelling into `project_name` just to make
one path look convenient.

## Compatibility Policy Versus Product Notes
`compatibility_policy` is only about compatibility promises.
Use it to say whether the project preserves old contracts, allows breaking
changes, or stays explicitly forward-only.

Do not overload `compatibility_policy` with free-form product notes such as
cross-platform support, packaging targets, deployment models, or feature
priorities.
Those belong in `project_description`, `README.md`, `SPEC.md`, or another
product-facing doc.

## What It Controls
Project-governance metadata feeds several visible surfaces:

- managed README identity

- generated governance headers in managed docs that opt into those headers

- tracked registry state

- changelog and release-heading behavior

- package metadata surfaces that are synchronized from repository identity

- the generated compatibility-guidance line in `AGENTS.md`

## Working Rule
If the repository identity, maturity, or versioning stance changes, update
`project-governance` first and then let the governed outputs regenerate.
That keeps the repository's public identity sourced from one place instead
of being duplicated across unrelated files.
