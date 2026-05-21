# Contracts
**Last Updated:** 2026-04-27
**Project Version:** 1.0.1b5

## Overview
This page is the ownership map for the DevCovenant package docs.
Use it to answer two questions:

1. what kind of rule or definition am I looking at?
2. which document explains it?

The package docs work best when ownership stays explicit.
Each page should explain one area clearly and point to the next page when the
subject crosses a boundary.

## Main Kinds Of Rules
DevCovenant has five practical kinds of rules and definitions.
Keeping them separate is the easiest way to understand how the product works.

### 1. Project Governance
`project-governance` is the repository's public identity and lifecycle stance.
It answers questions such as:

- what is this project called?
- what stage is it in?
- what compatibility promise does it make?
- is it versioned or unversioned?

Read more in:

- `devcovenant/docs/project_governance.md`
- `devcovenant/docs/config.md`

### 2. Workflow Definition
The workflow definition is the fixed gate/run sequence plus the declared runs
that live between `verify` and `close`.
It covers:

- `gate --open`, `gate --verify`, `run`, `gate --close`
- reserved anchors `open`, `verify`, `close`
- run ordering via `after`, `before`, and `order`
- workflow freshness and evidence rules
- CI mapping for the generated workflow file

Read more in:

- `devcovenant/docs/workflow.md`
- `devcovenant/docs/config.md`
- `devcovenant/docs/registry.md`

### 3. Configurable Policies
Policies are the repository-facing enforcement units.
They are enabled or disabled through `policy_state` and then tuned through
profile settings and config overrides.
They cover things like changelog coverage, dependency management, test
structure, docs growth, and version synchronization.

Read more in:

- `devcovenant/docs/policies.md`
- `devcovenant/docs/config.md`

### 4. Profiles, Assets, And Translators
Profiles describe reusable repository shape.
They contribute:

- settings overlays
- workflow runs
- managed assets
- pre-commit fragments
- CI fragments
- translators

Read more in:

- `devcovenant/docs/profiles.md`
- `devcovenant/docs/config.md`

### 5. Generated And Local State
DevCovenant writes both tracked and runtime-local state.
That includes:

- managed docs
- tracked registry data
- runtime session ledgers
- run logs
- generated workflow and ignore files

Read more in:

- `devcovenant/docs/registry.md`
- `devcovenant/docs/refresh.md`
- `devcovenant/docs/workflow.md`

## Document Ownership Map
Use the package docs like this:

- `devcovenant/docs/installation.md`
  install, deploy, upgrade, uninstall, and first-review flow

- `devcovenant/docs/workflow.md`
  gate sequence, declared runs, evidence files, and CI mapping

- `devcovenant/docs/config.md`
  public `devcovenant/config.yaml` settings and ownership model

- `devcovenant/docs/policies.md`
  configurable policies, runtime actions, and policy commands

- `devcovenant/docs/profiles.md`
  profiles, assets, translators, workflow runs, and CI fragments

- `devcovenant/docs/project_governance.md`
  repository identity, maturity, compatibility, and versioning stance

- `devcovenant/docs/registry.md`
  tracked registry structure, runtime ledgers, and generated state

- `devcovenant/docs/refresh.md`
  managed-doc refresh and descriptor-driven materialization

- `devcovenant/docs/architecture.md`
  internal layer ownership and runtime composition

## Writing Rule
When behavior changes, update the owning page first.
Then update summaries, templates, maps, and supporting docs that point back
to it.

That keeps the package docs honest without turning every page into a dump of
every other page.
