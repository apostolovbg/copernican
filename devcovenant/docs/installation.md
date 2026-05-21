# Installation and Lifecycle
**Last Updated:** 2026-04-27

**Project Version:** 1.0.1b5

## Overview
This page explains how to install DevCovenant, when to use `pipx` versus a
source checkout, and what each lifecycle command does inside a repository.

DevCovenant separates setup from activation:
- `install` copies DevCovenant into the repository and writes the starting
  config
- `deploy` applies that reviewed config and writes the managed outputs

The important human decision happens between those two commands.
Review `devcovenant/config.yaml`, decide how the repository should work, and
only then run `deploy`.
If you want a terse static reminder of that sequence, run
`devcovenant quickstart`.
If you want a disposable guided evaluation repo, run `devcovenant demo`.
That demo proves a repo-owned custom security policy shadow copy, a
changelog-safe legacy drift example, and the normal gate/run/close flow.

## Before You Start
You need:
- a way to run DevCovenant on your machine: `pipx` for normal use, or a full
  source checkout when you are developing DevCovenant itself
- a git repository
- Python and any tools required by the active profile stack
- permission to create or use the managed environment when that policy is
  enabled

## User Install Versus Source Development
Use these two paths deliberately:

1. `pipx` install for normal CLI use.
   This is the preferred machine install when you want to use DevCovenant in
   one or more repositories.

2. source checkout for DevCovenant development.
   Use a full source checkout only when you are developing DevCovenant itself
   or testing unreleased changes.

If the console script is unavailable from a source checkout, use
`python3 -m devcovenant ...`.
On Windows, `py -m devcovenant ...` is the usual equivalent form.
Every public command also accepts `--quiet`, `--normal`, or `--verbose`.

## Preferred Machine Install
For normal CLI use, install DevCovenant with `pipx`:

```bash
pipx install devcovenant
devcovenant --version
```

Use these companion commands for the installed CLI:

```bash
pipx upgrade devcovenant
pipx uninstall devcovenant
```

`devcovenant --version` reports the bundled DevCovenant package version from
the shipped `devcovenant/VERSION` file.
That is separate from the governed project version inside a repository.
The project version comes from the configured `version-sync.version_file`, and
the `version-sync` policy repairs declared version-bearing targets so they
follow that canonical project version during the normal workflow.
Managed docs render `DevCovenant Version` headers from the same canonical
version value during refresh, so package and repository-facing headers stay
in step without a separate bump path.
The repository manifest in `pyproject.toml` keeps package metadata and
repository URLs together. `version-sync` owns the version field and changelog
headers; repo-facing `project.urls` entries stay on the `main` branch.

That machine install is separate from repository activation.
Installing the CLI makes `devcovenant` available on the machine.
Running `devcovenant install` inside a repository adds DevCovenant to that
repository.
The shipped runtime also includes policy-owned tool dependencies such as
Bandit for the Python `security-scanner` backend, so installed repos and
generated CI do not need a separate Bandit bootstrap step outside the normal
DevCovenant runtime surface.

## Install Versus Deploy
The shortest accurate model is:

1. `install` is setup.
   It copies DevCovenant into the repository, writes config, and prepares the
   tracked files DevCovenant needs.

2. config review is the checkpoint.
   Start with `project-governance`, `developer_mode`, and `profiles.active`.
   The seeded `devcovenant/config.yaml` should keep explanatory comments for
   that review, so treat those comments as the first checklist instead of
   guessing from memory.
   For most repositories, keep `devcovuser` active as the always-on baseline
   and add a repo-owned `userproject` starter profile when the repository
   needs its own rules, assets, workflow additions, or dependency-surface
   overrides.
   A good starting point is copying
   `devcovenant/builtin/profiles/userproject/` to
   `devcovenant/custom/profiles/userproject/`, then editing only the
   repo-specific facts there.
   Keep inherited values inherited.
   Do not restate them in the copied profile.
   Here, "inherited" means values from other active profiles.
   When a custom and builtin profile share a profile name, the custom profile
   is loaded, fully shadows the builtin profile, and the builtin profile with
   that name is ignored.
   Keep `github` active when the repository wants the generic generated
   GitHub Actions workflow that ships in the default user stack; remove it
   when the repository does not want that workflow.
   Use an optional GitHub-specific custom profile when the
   repository needs reusable GitHub-only CI fragments beyond the builtin base.
   Use direct overlays only for small local exceptions.

3. `deploy` is activation.
   It runs the full refresh path and writes managed docs, generated files,
   registry outputs, and other DevCovenant-managed outputs.
   Repo-owned custom policies and copied custom profiles stay in place during
   that lifecycle work; the install/deploy/upgrade flow should not prune
   `devcovenant/custom/**` content that belongs to the governed repository.

`install.config_reviewed` exists only to make that checkpoint explicit.
It means a human reviewed the starting config and is ready to activate it.
It is not a hidden switch.

## Copy-Ready Custom Profile Template
DevCovenant ships a copy-ready bootstrap template for the first custom
profile:
- copy `devcovenant/builtin/profiles/userproject/`
  to `devcovenant/custom/profiles/userproject/`
- edit the copied manifest there
- keep inherited values inherited instead of copying builtin metadata into the
  custom layer
- when the copied profile keeps the same profile name, it fully shadows the
  builtin profile with that name

Use that copied profile for repo-specific identity, version paths, package
paths, extra hooks, local workflow runs, or managed-environment details.
If bootstrap-mode reuse of the current interpreter is intentional, set
`allow_current_interpreter_fallback: true` in the managed-environment
metadata; otherwise keep `managed_commands` populated so the profile does not
silently rely on an incomplete environment declaration.
If the repository later needs GitHub-only CI extensions, keep those in a
separate optional custom profile instead of the repo-identity profile.

## Common Starting Situations
### Empty Repository
`install` writes the starting runtime files and config.
`deploy` creates the first managed docs and generated governance files.
Set `project-governance` before that first `deploy` so the generated docs use
real project identity and release settings from the start.

### Repository Seeded With DevCovenant-Shaped Docs
If the repo already contains compatible `README.md`, `SPEC.md`, `PLAN.md`, or
similar docs, keep them in place before the first `deploy`.
Refresh can adopt compatible docs instead of overwriting their authored body.

### Existing Repository With Real Content
`install` leaves ordinary repository files alone.
`deploy` adds DevCovenant around them using the managed-doc preservation rules.
Before the first `deploy`, make sure `project-governance` matches the real
project so generated docs and headings describe the repository honestly.

## Lifecycle Commands
### install
Copies DevCovenant into the repository, writes config, and prepares tracked
state.
It does not activate managed docs or generated governance files.
If DevCovenant already exists, `install` stops and points you to `upgrade`.

### deploy
Requires `install.config_reviewed: true`.
Runs the full refresh path and writes the active managed outputs.

### refresh
Rebuilds tracked registry state, managed docs, generated config sections,
generated workflow files, `.gitignore`, dependency locks, and related outputs.
On a converged repository, it should avoid rewriting current tracked state
when the effective inputs did not change.
Its run summaries can also include per-phase timing details so slow steps are
visible before you inspect the full logs.

### asset
Writes a Desktop copy of a shipped profile asset or managed doc template.
Use it when you want one rendered operator copy without writing the
repository-managed target path.

The practical rules are:
- exact target-path matches beat basename matches
- active profiles are considered first in active-profile order
- the optional second argument must be a filename only, not a path
- `--overwrite` is required when the Desktop target already exists

If the winning profile still exposes more than one basename match, rerun the
command with the exact target path.
Use `devcovenant asset --help` for the command-scoped syntax.

### custom
Copies or removes a repo-owned shadow copy of one builtin policy or profile.
Use `--policy <policy-id> --do` or `--profile <profile-name> --do` to copy
the builtin tree into `devcovenant/custom/...` and materialize mirrored tests
under `tests/devcovenant/custom/...`.
Use `--undo` to remove the repo-owned copy and its mirrored tests again.
The command runs refresh automatically after the copy or removal.
Like the other lifecycle commands, `custom` follows the repository's
resolved managed environment instead of assuming a local `.venv`.

### run
Runs the declared workflow steps for the repository.
Use it when a gate command tells you that workflow evidence is stale and
must be refreshed before a new `gate --open` or before `gate --close`
can close.

### clean
Removes disposable build, cache, runtime-registry, or log files according to
the cleanup rules.
Run it only after the active gate session is closed.

### upgrade
Reconciles the installed DevCovenant package from source and then runs
`refresh`.
Use it when DevCovenant is already present and you want the newer runtime.
Like `install` and `deploy`, upgrade should preserve repo-owned custom
policies and copied custom profiles instead of treating them as package
leakage.

### undeploy
Removes managed outputs while keeping the installed core and config.
Use it when you want to deactivate the managed outputs without uninstalling
DevCovenant entirely.
When tracked managed-doc routing is healthy, undeploy should use that declared
routing first and reserve repository-wide recovery scans for broken-config
paths.

### uninstall
Removes the DevCovenant footprint from the repository.
Use it only when you are truly removing DevCovenant from the repo.
Like the other heavy lifecycle commands, it can record per-phase timing
details in the summary artifacts.

## Command Startup Behavior
DevCovenant does not treat every command path the same.
Lightweight command paths should stay lightweight:
- `devcovenant --help`
- `devcovenant <command> --help`
- `devcovenant gate --status`

Those paths should not pay for the full workflow runtime, managed-environment
re-exec checks, or tracked run-log setup when they only need parser output or
read-only local session inspection.
Heavy runtime setup belongs to commands that actually execute policy, gate, or
workflow behavior.

The same principle applies after the repository is converged.
Normal startup paths compare tracked fingerprints first and should skip
rebuilding current policy-registry or dependency-surface state when the
effective inputs did not change.

## Command Surface Ownership
The source-owned command modules under `devcovenant/*.py` stay intentionally
small.
`cli.py` owns argument dispatch, while `check.py`, `clean.py`, `install.py`,
`deploy.py`, `refresh.py`, `run.py`, `gate.py`, `policy.py`, `asset.py`,
`undeploy.py`, `uninstall.py`, and `upgrade.py` hand work to the flat
`devcovenant/core/*.py` runtime surface.

That split keeps the public command entrypoints readable while the flat core
modules own the real orchestration logic.
Operator-facing command behavior should stay documented here in lockstep with
the owning flat-core runtime module instead of drifting into hidden wrapper
logic.

When you run DevCovenant from a source checkout, DevCovenant-owned trees also
have a stricter hygiene contract than ordinary user code:
- `devcovenant/**`
- `tests/devcovenant/**`

Those owned trees must not write repository-local `__pycache__`, `*.pyc`,
`*.pyo`, or similar compiled Python artifacts during normal DevCovenant
source and test runs.
If one of those artifacts appears under the owned trees, treat it as a
DevCovenant bug rather than normal Python residue.

## Package Build Surface
The published package intentionally ships the runtime-facing docs and profile
assets that DevCovenant needs at install time:
- the packaged `devcovenant/README.md` and `devcovenant/VERSION`
- the packaged `devcovenant/runtime-requirements.lock` used when the builtin
  `github` profile bootstraps DevCovenant
- the packaged `devcovenant/licenses/LICENSE`
- the packaged `devcovenant/licenses/**` license files
- the built-in policy descriptors under `devcovenant/builtin/policies`
- the built-in profile descriptors, translators, and asset templates under
  `devcovenant/builtin/profiles`
- the packaged `devcovenant/core/README.md`, because structure validation
  treats the flat-core README as part of the shipped core surface
- the packaged docs under `devcovenant/docs`
- the tracked `README.md` files for `devcovenant/logs` and
  `devcovenant/registry`

The published package must not ship live repository state or local debris such
as:
- `devcovenant/config.yaml`
- `devcovenant/registry/registry.yaml`
- `devcovenant/registry/runtime/**`
- timestamped runtime log folders
- local test trees, build trees, or `*.egg-info` outputs

`MANIFEST.in` and `pyproject.toml` should keep that boundary explicit.
Package metadata should only claim Python versions that the release workflow
actually proves.
Package links should point at release-stable docs instead of a moving branch
head.
Command banners should identify the active packaged build by reading
`devcovenant/VERSION`, so installed operators and source-checkout launches
report the same version string for the same package build.

## First-Time Setup Runbook
Use this as the practical first integration flow:

1. Run `devcovenant install`.
2. Open `devcovenant/config.yaml` and review:
   - `project-governance`
   - `developer_mode`
   - `profiles.active`
   - `paths`
   - `workflow`
   - `integrity`
   - `doc_assets`
   - `policy_state`
   - `engine.*`
3. Set `install.config_reviewed: true`.
4. Run `devcovenant deploy`.
5. Prepare the environment declared by the active profile stack.

   If the active stack seeds a local `.venv`, `deploy` materializes the
   workspace dependency artifacts and one manual equivalent is creating
   `.venv` and installing `requirements.lock`.
   `gate --open` can also run the declared bootstrap commands when the target
   environment is still missing.
   On Windows, use `.venv\\Scripts\\python.exe -m pip install -r \
   requirements.lock`.
   That seeded `.venv` flow is only one starting point.
   If the repository uses a system interpreter, bench-managed environment,
   container-managed environment, or another layout, declare that environment
   first through the profile stack or metadata overlays, then prepare it.
   DevCovenant must either run from that declared managed context already or
   be able to resolve the declared interpreter path or environment root.
6. Prove the reviewed setup with the full gate cycle:

   ```bash
   devcovenant gate --open
   devcovenant gate --verify
   devcovenant run
   devcovenant gate --close
   ```

For a normal repository, do that first cycle before adding custom policies or
profiles under `devcovenant/custom/`.
Start from a clean working base, then add custom extensions on top.
That usually means:
1. a custom profile for project-owned behavior such as `root_workspace`
2. an optional GitHub-specific custom profile only when the repository wants
   extra GitHub CI behavior beyond the builtin base

## Developer Mode
`developer_mode: false` means a normal repository using DevCovenant as a tool.

`developer_mode: true` means the repository is being used to develop
DevCovenant itself.
That enables development-only paths that ordinary user repositories should not
keep.

## Managed Environment Notes
The shipped install baseline keeps `managed-environment` disabled until the
repository opts into a declared environment model such as `python_venv`.
When a repository wants the builtin `.venv` contract, add `python_venv` to
`profiles.active`, set `policy_state.managed-environment: true`, and rerun
`deploy` or `refresh`.
When the managed-environment policy is enabled, DevCovenant chooses one target
environment for each stage and execution mode.
If the current interpreter already matches that setup, DevCovenant reuses it.
If not, it selects the configured interpreter or environment root and then runs
any declared bootstrap commands.
If the selected interpreter path exists but is not executable, DevCovenant
stops with a clear error.
The shipped `python_venv` profile declares both bootstrap-stage and
open-stage bootstrap commands, so bootstrap-mode commands can enter the
repo-local `.venv` after the repository opts in.
If the repository uses a bench-managed, container-managed, system, or other
custom environment, declare that environment through the profile stack or
metadata overlays instead of expecting DevCovenant to guess an unknown layout
or hidden launcher hop.
Tracked dependency and registry fingerprints should stay repo-relative and
checkout-stable.
Machine-local evidence belongs in `devcovenant/registry/runtime/`, not in the
tracked registry or other committed artifacts.

## Quick Reference
```bash
pipx install devcovenant
pipx upgrade devcovenant
devcovenant install
devcovenant deploy
devcovenant asset SPEC.md
devcovenant asset SPEC.md OTHERNAME.md
devcovenant refresh
devcovenant policy dependency-management refresh-all
devcovenant clean --all
devcovenant upgrade
devcovenant undeploy
devcovenant uninstall
```
