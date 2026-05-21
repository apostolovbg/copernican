# Profiles
**Last Updated:** 2026-04-27

**Project Version:** 1.0.1b5

## Overview
Profiles tell DevCovenant what kind of repository it is working in and which
reusable behavior should come with that setup.

A profile can contribute:
1. metadata overlays
2. workflow runs
3. managed assets
4. pre-commit fragments
5. suffix inventories
6. translator declarations
7. CI fragments through `ci_and_test`
8. ignore-directory hints that feed generated `.gitignore`, pre-commit
   excludes, and the `gate --open` snapshot walk

Profiles do not directly turn policies on or off.
Policy activation still lives in `policy_state`.

## Profile Types
The common profile categories are:
- `global` and `defaults` as the shared base
- `devcovuser` as the always-active core user baseline
- `userproject` as the copy-ready bootstrap template for the first
  repo-owned custom profile
- `github` as the optional but default-enabled GitHub Actions layer
- language profiles
- optional tooling profiles such as `python_venv`
- framework or tooling profiles
- custom profiles

The normal pattern is:
1. keep the base profiles active
2. keep `devcovuser` active for an ordinary repository using DevCovenant
3. keep `github` active when the repository wants the generic generated GitHub
   Actions workflow; remove it when the repository does not want that
   workflow
4. add the needed language or stack profiles
5. add a custom profile on top when the repository needs its own rules,
   assets, workflow additions, or dependency-surface ownership
6. add an optional GitHub-specific custom profile when the repository needs
   reusable GitHub-only CI fragments that should not affect local behavior

Use direct overlays when you only need a very small local tweak.
Use a custom profile when the repository has real repeatable behavior of its
own.

The built-in `userproject` profile is the copy-ready bootstrap template for
that first custom layer.
Copy `devcovenant/builtin/profiles/userproject/` to
`devcovenant/custom/profiles/userproject/`, then edit the copied manifest.
It is not meant to be activated directly.
It does not ship test blueprints, so copying it into the custom tree
refreshes the manifest without creating a mirror.
Keep inherited values inherited.
Do not restate builtin values in the copied profile.
Here, "inherited" means values from other active profiles.
When a custom and builtin profile share a profile name, the custom profile is
loaded, fully shadows the builtin profile, and the builtin profile with that
name is ignored.

## Same-Name Shadowing
Same-name custom profiles are the supported way to replace shipped profile
behavior without editing the builtin package files.

The rule is strict:
- if `devcovenant/custom/profiles/python/python.yaml` exists, that custom
  `python` profile is the active `python` profile
- the builtin `python` profile is ignored
- DevCovenant does not merge the builtin profile into the custom one

That is why copied same-name shadow profiles should stay thin.
Keep inherited values inherited from the other active profiles instead of
restating them in the copied profile, and copy only the owned assets,
translators, helpers, or tests that the shadow profile replaces.

## Builtin Profile Test Blueprints
Builtin profile packages can ship a sibling `test_blueprints.yaml` file when
the shipped tests should remain package data instead of a live discoverable
`tests/**` tree.
That blueprint stores repo-relative paths and serialized file contents for
the shipped tests.
The repository keeps the actual builtin test tree under
`tests/devcovenant/builtin/profiles/<name>/` for verification.
When a repository shadows a builtin profile that ships `test_blueprints.yaml`,
`devcovenant custom --profile <name> --do` copies the builtin tree into
`devcovenant/custom/profiles/<name>/` and materializes the mirrored tests
under `tests/devcovenant/custom/profiles/<name>/`. If the builtin profile
does not ship blueprints, the command copies the tree and refreshes without
creating a mirror. `--undo` removes the repo-owned copy and any mirror again.
The command follows the normal managed-environment resolution path, so it
works from the repository's declared interpreter layout instead of assuming a
repo-local `.venv`.
The repo-local `builtin-test-blueprint-sync` policy reads the same
`mirror_roots`, `blueprint_directories`, and `blueprint_name` metadata to
keep builtin and custom profile mirrors declarative rather than hardcoded.

## Custom Profiles As Governance Packs
A custom profile is the normal way to package governance for one repository
family so it travels as one coherent stack instead of a pile of one-off
config edits.

For a guided first custom profile path, start with
[customization.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/customization.md).
That page shows when the repo-owned `userproject` starter profile is the
right place to put repository-specific values and when to add a custom
`python` or `python_venv` layer.

A custom profile can contribute:
- metadata overlays for builtin or custom policies
- managed docs and other assets
- workflow runs
- CI fragments
- managed-environment declarations
- dependency roles and dependency-surface ownership
- translator declarations for a language or stack
- ignore-directory hints and other reusable repo-shape facts

The copied `userproject` template is deliberately minimal.
It exists so a repository does not have to start from a blank file just to
express its own identity and local behavior.
That keeps the shipped base profiles generic while still giving new
repositories a concrete starting point.

That is why custom policies and custom profiles work best together.
The policy owns rule logic.
The profile packages the reusable metadata, assets, environment model, and run
shape that make the rule meaningful for a given repository family.

## What Profiles Should Own
Profiles are the right place for reusable behavior.
That includes:
1. dependency file roles for a language ecosystem
2. managed-environment expectations for a stack
3. generated asset templates
4. translator declarations for a language
5. documentation routes for a reusable profile stack
6. extra CI jobs that should apply to similar repositories instead of every
   DevCovenant repository
7. declared workflow runs that should be required for repositories of the same
   shape

If the behavior should apply to more than one repository of the same shape, it
probably belongs in a profile instead of local config.

The built-in `defaults` profile stays environment-neutral.
The built-in `python_venv` profile is the shipped opt-in profile for a plain
Python `.venv` starting point:
- expected paths and interpreters
- required commands for the standard gate/runtime surface
- manual guidance that uses `{current_python}` and `{managed_python}`
- bootstrap-stage and open-stage bootstrap commands for lifecycle commands

The normal installed repository baseline keeps
`policy_state.managed-environment: false`.
When a repository wants the `.venv` contract, it should:
1. add `python_venv` to `profiles.active`
2. set `policy_state.managed-environment: true`
3. run `deploy`, `refresh`, `gate`, and `run` through any working launcher;
   the declared `python_venv` bootstrap commands can create the missing
   `.venv` before re-exec enters the managed interpreter

That is a starting point, not a promise that every repository should use
`.venv`.
Repositories may instead declare a system interpreter, bench-managed
environment, container-managed environment, or another execution layout
through their active profile stack or metadata overlays.
The important contract is that DevCovenant can run from that declared managed
context or resolve the interpreter path or environment root it should use.
Builtin profiles such as `defaults` do not try to guess hidden launcher hops.
If the managed environment keeps commands in extra PATH locations, declare
`command_search_paths` alongside the environment root so `required_commands`
resolve from the managed stack instead of the host shell PATH.

The built-in `devcovuser` profile is the always-active core user baseline.
It keeps DevCovenant's own shipped runtime files out of ordinary app-code
checks while still keeping `devcovenant/custom/**` in scope for
repository-owned extensions.
The global profile also owns the repository covenant text in `AGENTS.md`, so
its question handling and work-style rules can say that repository-authored
instructions come first, that read-only commands are acceptable when a human
prompt is a question, and that work should stay deliberate instead of chasing
speed.
That same narrowing applies to mirrored test expectations and assertion
coverage, so normal repositories keep DevCovenant internals out of scope
while still enforcing `devcovenant/custom/**` and
`tests/devcovenant/custom/**`.
In this repository, the repo-owned `userproject` starter profile widens the
same code-style policies, including `line-length-limit`, `name-clarity`,
`docstring-and-comment-coverage`, `security-scanner`, and `no-raw-errors`,
so they cover the full `devcovenant/**` tree and `tests/**` mirror instead of
stopping at the custom shadow layer. The generated registry, bundled license
dumps, and builtin test-blueprint verification tree stay out of that generic
scan so the profile only governs repo-owned source and custom mirrors.

Profiles may also contribute `ignore_dirs` for disposable local outputs that
should stay out of generated `.gitignore`, out of pre-commit's all-files
scan, and out of the startup snapshot walk used by `gate --open`.
Typical examples are temporary build directories, cache roots, declared
environment folders, or other data trees that should not count as
user-owned source files during the gate session snapshot.

A custom profile can then strengthen the standard stack for one project shape.
For example, it may add `managed_commands`, extra assets, CI steps, or
surface overrides such as a project-owned `root_workspace`.
A separate GitHub-oriented custom profile is useful when a repository wants
GitHub-only CI extensions without making those rules part of the local
runtime model.

## Assets And Managed Docs
Profiles can ship assets, including managed-document templates.

`devcovenant asset FILE.ext [OUTPUTNAME.ext]` is the command for those shipped
assets.
It can resolve both:
- manifest-declared profile assets
- descriptor-backed managed docs such as `SPEC.md`

Resolution works like this:
- exact target-path matches beat basename matches
- active profiles are considered first in active-profile order
- remaining discovered profiles are considered afterward in profile-name order
- if the winning profile still exposes multiple basename matches,
  DevCovenant stops and asks for an exact target path

The command writes Desktop copies only.
It uses the same rendering code that refresh and deploy use.
Use `devcovenant asset --help` for the operator-facing syntax; this page owns
the profile-selection and ownership rules behind that command.

Managed-doc descriptor ownership follows profile precedence by target path:
- the global profile provides the baseline descriptor set
- active profiles may add new target docs
- active profiles may override a global descriptor by shipping the same target
  path
- later active profiles win over earlier ones for the same target path

The global `LICENSE` descriptor is one special case worth calling out.
It keeps only the title line in sync as
`# {{ PROJECT_NAME }} {{ PROJECT_VERSION }}`.
The rest of the legal text stays user-owned, so repositories can change
their license body without fighting managed metadata lines.
The seeded legal body begins with `The MIT License (MIT)`, then uses
`{{ COPYRIGHT_NOTICE }}` from `project-governance`, and always places
`All rights reserved.` on the next line.

## Translators
Translators are owned by language profiles.
They let policies work with a normalized view of source files instead of making
every policy understand every language directly.

A translator declaration normally includes:
- a stable translator id
- handled file extensions
- a `can_handle` entrypoint
- a `translate` entrypoint

Practical resolution flow:
1. identify the file extension
2. collect candidate translators from the active language profiles
3. run `can_handle`
4. require one effective translator
5. return one normalized language unit

Framework and tooling profiles should not become alternate language owners.
Translator ownership belongs with language profiles.
The translator should also do the expensive parsing work once.
For example, the builtin Python translator collects identifier,
symbol-doc, and risk facts in one tree walk so translator-driven policies can
reuse the same language unit instead of reparsing the same module repeatedly.
When a policy only needs a narrow fact set, the language profile may also
expose a lighter translation path instead of forcing every caller through the
full normalized symbol model.

## Metadata Overlays
Profiles are the preferred place for reusable stack-specific settings.
Examples include:
1. dependency-management surfaces
2. version-sync file roles
3. documentation-growth routes
4. no-print sink metadata from language profiles
5. reusable workflow runs such as a stack's `tests` run
6. reusable `ci_and_test` fragments for stack-specific CI jobs
7. managed-environment roots that cleanup and other services need to respect

Version ownership belongs here too.
Profiles and overlays decide which file is the governed project version source
through `version-governance.version_file` and `version-sync.version_file`.
That lets one repository treat `VERSION` as canonical, another use a package
subpath, and both still keep the rest of their declared version-bearing
targets synchronized through the same policy contract.

The builtin `python` profile also demonstrates policy-owned scanner metadata.
It contributes the default Bandit backend to `security-scanner` and ships the
matching `bandit.yaml` asset.
That means the Python baseline gets the same scanner path in local gates,
local `check`, installed-repo runs, and generated CI without adding a
separate workflow command just for Bandit.

The CI boundary matters.
The builtin `github` workflow template should stay generic.
It should bootstrap DevCovenant from the shipped
`devcovenant/runtime-requirements.lock`, not from the project's
dependency files.
That generic base should avoid floating installer state.
If the shipped lock pins `pip`, the workflow should install from that lock
instead of upgrading `pip` live first.
If a repository needs extra project dependency setup or extra CI steps, that
extension belongs in the relevant profile instead of in the builtin base
workflow.
Repository-owned artifact-proof bootstrap snippets should update the real
`install.config_reviewed` field line in `devcovenant/config.yaml`, not a
comment occurrence earlier in the template text.
When those snippets activate `python_venv` in `profiles.active`, they should
reuse the existing list-item indentation so the commented config file stays
valid and preserves its review guidance.
If a Python repository turns on hash-locked `requirements.lock`, keep any
local-artifact install path split into two steps:
1. install the locked requirements
2. install the local wheel or sdist with `--no-deps`

The same split helps config stay readable.
The global config asset lists the full `project-governance` key set and the
allowed values.
Profiles and local config can then tighten or extend behavior without
making the shared base too specific to one repository.

That same model is what makes custom policies versatile.
A profile can feed structured YAML into a custom policy without changing the
policy code again.
Because mapping lists with stable `id` keys merge by `id`, a profile can
extend inventories such as dependency surfaces, route groups, evidence sets,
or ownership maps instead of replacing the whole list every time.

If a language or stack has a standard run, the profile should declare it
through `workflow_runs`.
That keeps shared run definitions in one place.
In the built-in Python profile, the standard `tests` run lives there and uses
`python3 -m unittest discover -v` as the default Python test command.

The built-in Python and GitHub profiles also demonstrate the default
dependency-surface split:
1. `root_workspace` for the repository's working environment
2. `package_runtime` for the repository's shipped Python package, when the
   package path exists
3. `devcovenant_runtime` for the shipped DevCovenant bootstrap surface used by
   the builtin `github` workflow

For most repositories, the first two surfaces are the ones they actually own.
`devcovenant_runtime` exists so the packaged DevCovenant bootstrap path can
ship its own runtime contract; it is not usually a surface that ordinary
adopters edit directly.

In the default Python stack, `root_workspace` starts from `requirements.in`.
That input inherits the shipped `devcovenant/runtime-requirements.lock`, and
`dependency-management` then writes the real `requirements.lock` during
`deploy`/`refresh`.
When a repository also enables `package_runtime`, the normal composed shape is
still three managed surfaces: `root_workspace`, `devcovenant_runtime`, and
`package_runtime`.
The root workspace stays flat, while dependency-management treats the
inherited runtime locks as already-resolved provider surfaces instead of
re-solving all three surfaces as one synthetic target closure.
The builtin Python surfaces resolve against the supported CPython 3.11 through
3.14 matrix on Linux, Windows, and macOS so workspace locks do not depend on
the machine that happened to run refresh.
Tracked dependency fingerprints for those surfaces must stay repo-relative and
checkout-stable so a refresh on one machine does not rewrite registry state on
another just because the checkout root changed.

The shipped defaults keep the Python dependency surfaces hash-locked:
`root_workspace`, `devcovenant_runtime`, and `package_runtime` when a
repository enables that optional package surface.
A custom profile can override those defaults per surface instead of inventing
a second special-case dependency model.
Those same builtin surfaces also seed surface-local vulnerability auditing, so
the Python and GitHub profiles do not need a separate repository CI command to
check locked dependencies on only one host platform.
When a repository needs repo-specific dependency-lock ownership, that usually
belongs in its copied `userproject` custom profile instead of in a same-name
shadow of the builtin language profile.

Profile ownership also includes the shipped translator maps.
The builtin language translator set currently covers `csharp`, `dart`, `go`,
`java`, `javascript`, `objective_c`, `opencl`, `php`, `python`, `ruby`,
`rust`, `sql`, `swift`, and `typescript`.
Those profiles continue to own `can_handle`, `translate`, and their asset
metadata through files such as `python.yaml`, profile assets such as
`PROFILE_MAP.yaml`, and repository-specific custom-profile overlays,
while the shared translator runtime lives in the flat core modules
`devcovenant/core/translator.py` and `devcovenant/core/run_events.py`.

## Workflow Runs
`workflow_runs` is the public profile authoring model for extra run steps.
Each run may declare:
- `id`
- ordering fields `after`, `before`, and `order`
- `runner`
- `success_contract`
- `freshness`
- `recording`

Ordering is real behavior, not decorative metadata.
- `after` and `before` may reference reserved anchors: `open`, `verify`,
  `close`
- `after` and `before` may also reference other declared run ids
- DevCovenant validates those references
- DevCovenant rejects cycles instead of silently keeping broken rules
- when multiple runs are eligible at the same time, `order`, then owner id,
  then run id break ties

Supported runner kinds are:
- `command_group`
- `runtime_action`
- `policy_command`
- `manual_attestation`

Supported success-check kinds are:
- `all_commands_exit_zero`
- `runtime_action_success`
- `policy_command_success`
- `manual_attested`
- `external_artifact_check`
