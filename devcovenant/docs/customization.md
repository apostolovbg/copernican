# Customization
**Last Updated:** 2026-04-27

**Project Version:** 1.0.1b5

## Overview
This page is the guided first custom-governance path.
Use it after the front-door README when you want to move from "what is
DevCovenant?" to "how do I make it fit my repository?"

The builtin `devcovuser` layer is the always-active baseline. In practice,
most downstream repositories keep it active, then add a repo-owned
`userproject` starter profile carrying repository-specific values, and Python
repositories often add a custom `python` or `python_venv` profile too. That
split is deliberate: builtin layers stay generic while repository-owned
metadata holds the real project law.

There are two first custom paths:
- profile-first when the repository shape changes
- policy-first when the rule itself changes

Start with a profile when you need repo identity, assets, workflow additions,
dependency ownership, or environment metadata.
Start with a policy when you need a new rule, a different enforcement shape,
or a shadow of builtin rule logic.

## First Custom Profile Path
Use a custom profile when the repository needs reusable behavior of its own.
The concrete bootstrap example is the copy-ready `userproject` template.

Run:

```bash
devcovenant custom --profile userproject --do
```

That command copies the builtin `userproject` template into
`devcovenant/custom/profiles/userproject/`. Because the template does not
ship test blueprints, it does not materialize mirrored tests. The command
still refreshes the repository afterward.

Edit the copied manifest and assets there.
Put repository-specific values in the copied profile, not in the builtin
template.
Use that layer for:
- repo identity and managed-doc ownership
- workflow additions that belong to this repository family
- environment ownership such as `python_venv`
- dependency-surface defaults and other repo-shape metadata

Keep inherited values inherited.
Do not restate builtin values in the copied profile unless the repository
really needs to override them.
When a custom profile shares the same profile name as a builtin profile, the
custom profile fully shadows the builtin one.

Use `--undo` when the repository should return to the builtin baseline:

```bash
devcovenant custom --profile userproject --undo
```

## First Custom Policy Path
Use a custom policy when the repository needs a new rule or needs to shadow
an existing builtin rule.
The concrete shadowing example is `security-scanner`.

Run:

```bash
devcovenant custom --policy security-scanner --do
```

That command copies the builtin policy tree into
`devcovenant/custom/policies/security-scanner/` and materializes the mirrored
tests under `tests/devcovenant/custom/policies/security-scanner/`.

Edit the copied policy descriptor and runtime files there.
Put rule logic and policy-specific metadata in the policy layer, not in a
profile.
Use that layer for:
- severity and enforcement behavior
- selector scopes for the files the rule should watch
- audit or scanner metadata
- autofix helpers and explicit operator commands

If no builtin policy is close enough, create a new policy under
`devcovenant/custom/policies/<policy-id>/` instead of forcing a shadow.
Use `--undo` when the repository should return to the builtin policy:

```bash
devcovenant custom --policy security-scanner --undo
```

## Shadowing And Materialization
`devcovenant custom --profile ... --do` and `devcovenant custom --policy ...
--do` are the materialization commands.
They promote a builtin tree into repository-owned custom files and mirror the
shipped builtin tests into `tests/devcovenant/custom/**` when the builtin
ships blueprints.

The inverse `--undo` commands retract the repository-owned copy and any test
mirror.

The shadowing rules are strict:
- same-name custom profiles fully shadow builtin profiles
- same-id custom policies fully shadow builtin policies
- builtin entries are ignored once the matching custom entry exists

That is how DevCovenant keeps the extension surface explicit without hiding
where the active law came from.
The command still follows the repository's declared managed environment, so it
works from the interpreter layout the repository owns instead of assuming a
local `.venv`.

## Which Layer To Edit
Choose the layer that matches the change you want to make:
- profile when the repository shape changes
- policy when the rule changes
- config when the active stack or overlays change
- builtin source only when you are changing the shipped baseline itself

If the change feels like "the repo needs its own law", reach for a custom
profile or a custom policy instead of ad hoc overrides.

## Why It Matters
This layered model reduces drift.
It keeps repository law in owned metadata, makes the extension surface
repeatable, and gives consulting or integration work a clear starting point.

That is why the extension path is part of the product story, not just an
implementation detail.
It is the path that turns DevCovenant from a fixed rule bundle into a
repository-specific governance system.

## Next Steps
Read the deeper authoring pages once you know which layer you want to own:
- [policies.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/policies.md)
  for policy logic, selectors, and shadowing
- [profiles.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/profiles.md)
  for reusable repository shape and assets
- [config.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/config.md)
  for activation, overlays, and repository instance choices
- [custom/README.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/custom/README.md)
  for the extension-surface overview
