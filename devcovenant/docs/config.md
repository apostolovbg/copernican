# Configuration
**Last Updated:** 2026-04-27

**Project Version:** 1.0.1b5

## Overview
`devcovenant/config.yaml` is the main control file for a repository using
DevCovenant.
It is where you choose the profile stack, the managed docs, the workflow
settings, the policy toggles, and the general CLI behavior.

Refresh rewrites the generated parts of this file and keeps the human-owned
parts in place.
So the practical question is not "is this file generated?"
The practical question is "which parts do I review and own, and which parts are
there so DevCovenant can show resolved state?"

## How To Read This File
Most people should read it in this order:

1. `project-governance`
2. `developer_mode`
3. `profiles.active`
4. `doc_assets`
5. `workflow`
6. `policy_state`
7. `engine`
8. `paths`

If you only remember one rule, remember this one:
change the human-owned settings to tell DevCovenant how the repository should
behave, and treat the generated sections as status and reference output.

## Ownership Model
### Human-Owned Sections
These are the parts a human is expected to review and edit:
- `project-governance`
- `developer_mode`
- `profiles.active`
- `doc_assets`
- `paths.*`
- `workflow.*`
- `policy_state`
- `user_metadata_overlays`
- `user_metadata_overrides`
- `clean.overlays`
- `clean.overrides`
- `engine.*`
- `install.config_reviewed`
- generated-workflow override sections such as `pre_commit.*`, `gitignore.*`,
  and `ci_and_test.*`

### Refresh-Owned Sections
These sections are rebuilt by refresh and should not be treated as normal edit
targets:
- `profiles.generated`
- `autogen_metadata_overlays`
- `autogen_metadata_overrides`
- `install.import_managed_docs`

### Mixed Sections
These sections contain both human-owned and refresh-owned keys:
- `profiles`
- `doc_assets`
- `install`
- metadata layers split as `autogen_*` and `user_*`

## Key Sections
### profiles.active
This is the active profile stack.
Profiles describe the repository shape: language, framework, tooling, assets,
and workflow additions.

For most repositories, the normal pattern is:
- keep the shared base profiles
- keep `devcovuser` active
- keep `github` active when the repository wants the generic generated GitHub
  Actions workflow; remove it when the repository does not want that workflow
- add language or framework profiles as needed
- add a custom profile on top when the repository needs its own rules,
  assets, workflow additions, or dependency-surface ownership
- add an optional GitHub-specific custom profile when the repository needs
  reusable GitHub-only CI fragments beyond the builtin base

The copy-ready builtin bootstrap template lives at
`devcovenant/builtin/profiles/userproject/`.
Copy it to `devcovenant/custom/profiles/userproject/` when a repository needs
its first custom profile.
Keep inherited values inherited.
Do not restate builtin values in the copied profile.
Here, "inherited" means values from other active profiles.
When a custom and builtin profile share a profile name, the custom profile is
loaded, fully shadows the builtin profile, and the builtin profile with that
name is ignored.

If the repository needs the first custom profile or the first custom policy
path explained end to end, start with
[customization.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/customization.md).
That page describes the copy, shadow, and mirror rules in one place.

Use direct overlays for small one-off tweaks.
Use a custom profile when the repository has real repeatable behavior of its
own.
Before the first gate cycle, make sure the environment declared by that stack
actually exists.
If the active stack seeds a local `.venv`, `deploy`/`refresh` materializes the
workspace dependency artifacts and one manual realization is creating `.venv`
and installing `requirements.lock`.
That seeded `.venv` flow is only one starting point.
If the repository uses a system interpreter, bench-managed environment,
container-managed environment, or other custom environment, declare that
environment in the profile stack or metadata overlays instead of relying on
DevCovenant to guess an unknown layout or hidden launcher hop.
DevCovenant must either run from that declared managed context already or be
able to resolve the declared interpreter path or environment root.
If the environment needs extra executable locations, use
`managed-environment.command_search_paths` to declare those PATH entries
instead of depending on the host shell PATH.
If a profile intentionally wants `bootstrap`-mode operations to stay on the
current interpreter until the target environment exists, set
`allow_current_interpreter_fallback: true`. Otherwise the policy treats the
missing target environment as an explicit failure instead of silently masking
an incomplete profile.
Those profile-level `ignore_dirs` also feed the startup snapshot walk used by
`gate --open`, so disposable trees stay out of the recorded session state as
well as generated `.gitignore` and pre-commit excludes.
The generated config scaffold describes normal work as starting with
`gate --open`, not as opening the work itself.

The shipped user baseline keeps `github` active by default.
That makes the generated GitHub Actions workflow available out of the box for
GitHub-hosted repositories, but the profile is still optional: remove it when
the repository does not want generated GitHub Actions CI.

### doc_assets
This is the managed-doc selection.
Use it to choose which managed doc target paths are enabled for the repository.
The simple model is:
- global and active profiles contribute available managed-doc descriptors
- `doc_assets.autogen` names the target paths the current config enables
- `doc_assets.user` subtracts target paths after `autogen`
- when multiple active descriptor roots provide the same target path, the later
  active profile wins

### project-governance
This section describes the project itself.
It answers questions like:
- what is the project called?
- what copyright notice should seeded license docs use?
- what stage is it in?
- how actively is it maintained?
- what compatibility promise does it make?
- is it versioned or intentionally unversioned?

Managed docs and generated headers read from this section.
If those public-facing descriptions look wrong, start here.
That includes the global `LICENSE` template, which seeds:
- `Copyright (c) {{ COPYRIGHT_NOTICE }}`
- `All rights reserved.`

`project_name` is the canonical public/project identity string.
DevCovenant derives normalized path tokens such as `{{ PROJECT_NAME_PATH }}`
where package-safe paths need them, so `project_name` does not need to be
forced into Python import-package spelling.
For example, a project may keep `webcam-micro` as `project_name` while using
`webcam_micro` as a Python package path.

`compatibility_policy` is only about compatibility promises.
Do not overload it with free-form product notes such as cross-platform
support; those belong in `project_description`, `README.md`, or `SPEC.md`.

Version ownership splits across two config surfaces:
- `version-governance.version_file` names the canonical version file.
- `version-sync` fans that canonical value out to Project Version headers,
  DevCovenant Version headers, changelog headers, and manifest version
  fields.
- Managed docs render those headers from the same canonical value during
  refresh.

### paths
This section chooses where DevCovenant keeps important local files such as:
- `policy_definitions`
- `registry_file`
- `gate_status_file`
- `workflow_session_file`

These are path settings, not policy toggles.
Every key in this section is human-owned.

### workflow
This section controls workflow behavior such as:
- `pre_commit_command`
- `skipped_globs`

These settings define how the gate and run flow works.
They do not belong in `policy_state`.
When a managed Python environment is active, DevCovenant runs pre-commit
through that selected interpreter instead of depending on a host-side
console-script shim.

### policy_state
This is the on/off map for configurable policies.
Use it to decide which non-core policies are enabled.
Critical policies can still remain enforced even if a config toggle tries to
turn them off.
The shipped install baseline keeps `managed-environment: false` until the
repository enables a matching managed-environment profile such as
`python_venv`.
That separation keeps ordinary installed repositories environment-neutral by
default while still allowing explicit opt-in stacks to own their own
bootstrap commands and interpreter paths.

### user_metadata_overlays and user_metadata_overrides
These sections are where nested policy metadata lives.
They accept scalars, lists, and mappings directly.
There is not a separate "flat companion key" mode for structured metadata.
If one key is a mapping or a list of mappings, keep using subkeys under that
same key instead of inventing suffixed sibling keys for the same meaning.

This is also where most custom-policy tuning happens.
Any non-reserved descriptor key can be shaped here.
If a custom policy defines metadata such as:
- contract groups
- route inventories
- exception lists
- evidence requirements
- ownership maps
- environment commands
- stack-specific thresholds

those keys can be overlaid or overridden directly in YAML.

Selector-role metadata belongs here too.
If a policy declares or infers a role such as `api_contract`,
`release_docs`, or `seed_data`, config can supply:
- `<role>_globs`
- `<role>_files`
- `<role>_dirs`

That keeps scope shaping declarative instead of forcing policy code changes
for every repository.

The important rule is shape discipline:
- keep one metadata key in one shape
- do not replace a structured mapping with ad-hoc sibling flat keys
- do not overlay a scalar onto a mapping for the same key

Overlay and override have different jobs:
- overlays extend inherited metadata
- overrides replace inherited metadata

When a metadata key is a list of mappings with stable `id` values,
DevCovenant merges by `id` instead of treating the list as plain strings.
That is what lets repositories extend structured inventories such as
dependency surfaces or other policy-specific mapping lists without copying the
entire inherited payload.

For dependency management, that means using one structured
`dependency-management.surfaces` list instead of inventing separate
special-case root-versus-extra keys.
Surface overlays merge by `id`, so the normal pattern is:
1. inherit the default surface ids
2. override only the subkeys a repository needs to change
3. keep per-surface vulnerability audit settings such as `audit_service`
   and `audit_ignore_ids` on the owning surface instead of in separate
   CI-only commands

The same rule applies to
`dependency-management.license_source_overrides`.
Keep fallback license-source declarations in that structured list instead of
inventing repo-local scripts when a dependency omits bundled dist-info
license files.
Those override entries also merge by `id`, where the `id` is the normalized
package name.

### engine
This section controls general CLI behavior such as:
- failure threshold
- autofix enablement
- output mode
- test output mode
- run-log retention
- bytecode cache routing

These settings change how DevCovenant behaves.
They do not change what the repository claims about itself.
CLI flags such as `--quiet`, `--normal`, and `--verbose` override output mode
for one command only.

### ci_and_test
This section is for repository-local customization of the generated `CI`
workflow.
Activate the builtin `github` profile when the repository wants the standard
generated GitHub Actions workflow.
Use it for:
1. small repository-local overlays on the generated workflow
2. the rarer case where the repository deliberately takes full ownership of the
   generated workflow payload

Do not use it as the first place to add reusable behavior for a shared custom
profile.
If the added job should travel with a profile stack, put that behavior in a
profile `ci_and_test` fragment instead.
The builtin `github` base bootstraps DevCovenant from the shipped
`devcovenant/runtime-requirements.lock`. If a repository needs extra project
dependency setup, keep that in the relevant profile or explicit local
override instead of changing the builtin base.
If a repository needs to change dependency lock mode, do that by overriding the
relevant `dependency-management.surfaces` entry.
Those surface entries can turn `generate_hashes` on or off while still using
the same declared target matrix and refresh engine.
The same surface entries also own vulnerability auditing for their lockfiles.
If a repository wants that audit on or off, wants a reviewed exception, or
wants to inherit the shipped defaults, change the owning surface metadata
instead of adding a one-off `pip-audit` step to CI.
For example, a custom profile may own workspace dependency surfaces while an
optional GitHub-specific custom profile adds extra CI fragments.

### clean
Cleanup settings decide what DevCovenant may delete.
Use `clean.overlays` for extra cleanup targets.
Use `clean.overrides` only when the repository intentionally wants to replace
inherited cleanup lists.

### developer_mode
`developer_mode` answers a simple question:
is the project using DevCovenant as a tool, or is it being used to develop
DevCovenant itself?

Use `false` for a normal repository using DevCovenant.
Use `true` only when the repository is developing DevCovenant itself.

## Practical Review Order
For a new repository, this is the shortest useful config review:
1. set `project-governance`
2. confirm `developer_mode`
3. review `profiles.active`
4. keep `devcovuser` active for a normal repository
5. keep `github` if the repository wants the generated GitHub Actions workflow
   that ships in the default stack, or remove it if the repository does not
   want that workflow
6. copy `devcovenant/builtin/profiles/userproject/` when the repository needs
   a starting custom profile, then edit only the repo-specific facts there
7. review `doc_assets`
8. review `workflow` and `policy_state`
9. review `engine.*`
10. set `install.config_reviewed: true`
11. run `devcovenant deploy`

If generated docs or generated files do not match what you expect, the first
places to check are `project-governance`, `profiles.active`, and `doc_assets`.
