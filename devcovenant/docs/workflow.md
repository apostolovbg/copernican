# Workflow
**Last Updated:** 2026-04-27

**Project Version:** 1.0.1b5

## Overview
Use this page for the required command order, the meaning of each gate stage,
and where to look when something fails.

The normal DevCovenant work slice is:

```bash
devcovenant gate --open
# edit files and clear complaints while working
devcovenant gate --verify
devcovenant run
devcovenant gate --close
```

That sequence is the workflow.
It is how DevCovenant proves that hooks, checks, runs, and closing state all
happened in the right order.
If you want a terse static reminder of the lifecycle, run
`devcovenant quickstart`.
If you want the lifecycle shown as a disposable evaluation proof, run
`devcovenant demo`.

## What Each Command Is For
### check
Read-only audit.
Use it when you want to inspect the repository without opening or closing a
gate session.

### gate --status
Session inspection.
Use it when you want to know whether a gate session is open and which run logs
matter for the work slice.
It should stay a lightweight read-only path and should not behave like a full
gate or workflow run.

### gate --open
Opens the tracked work session.
It records the starting state that later checks compare against after
honoring active-profile ignore dirs and the configured engine ignores.
If `gate --open` fails, fix the reported problem before editing.

### gate --verify
Required pre-run check.
It catches pre-commit or DevCovenant changes before `run` records workflow
results.
If it reports changes, apply or clear them and rerun `gate --verify` until
it is clean.

### run
Runs the declared workflow steps for the repository.
This is the middle of the workflow, not the whole workflow by itself.

### gate --close
Runs the closing pre-commit pass and closes the session.
If required workflow evidence is stale or failing, `gate --close` will tell
you to refresh it before the session can close.

### custom
Copies or removes a repo-owned shadow copy of one builtin policy or profile.
Use `--policy <policy-id>` or `--profile <profile-name>` to select the builtin
source, then `--do` to copy that builtin tree into `devcovenant/custom/...`
and materialize mirrored tests under `tests/devcovenant/custom/...`, or
`--undo` to remove the repo-owned copy and mirrored tests.
The command runs refresh automatically after the copy or removal.
It follows the same managed-environment resolution as the other lifecycle
commands, so it works from the repository's declared interpreter layout
instead of assuming a local `.venv`.

## Required Order
The public workflow has four stages:
1. `gate --open`
2. `gate --verify`
3. `run`
4. `gate --close`

The reserved anchors are:
- `open`
- `verify`
- `close`

Declared workflow runs live between `verify` and `close`.
Profiles contribute those runs through `workflow_runs`.
DevCovenant validates and orders them with:
- `after`
- `before`
- `order`

Unknown references fail.
Cycles fail.
Run order is real behavior, not decoration.

## Run Logs
Every DevCovenant command writes a run folder under `devcovenant/logs/`.
When a command prints `Run logs: ...`, inspect files in this order:
1. `summary.txt`
2. `tail.txt` when present
3. `stdout.log`
4. `stderr.log`

The CLI can stream child output live, but the run logs are the stable record.
For lifecycle commands such as `install`, `deploy`, `refresh`, `custom`,
`undeploy`, `uninstall`, and `upgrade`, `summary.txt` and `summary.json` can
also include phase timing details so you can see where the command actually
spent time before opening the full logs.

## Local Workflow State
DevCovenant keeps two local workflow files:
- `devcovenant/registry/runtime/gate_status.json`
- `devcovenant/registry/runtime/workflow_session.json`

`gate_status.json` records gate-stage state and the pre-commit evidence tied to
those stages.
`workflow_session.json` records the declared runs for the open session, whether
they passed, and whether the results are still fresh.

The tracked counterpart to that local state is the workflow section named
`workflow_contract` in `devcovenant/registry/registry.yaml`.
That section records the declared runs and the order DevCovenant must
enforce.

## Freshness
Workflow runs are not only pass or fail.
They also have freshness rules.
By default, changelog-only edits do not force a rerun of otherwise fresh
workflow evidence.
Profiles can declare broader or stricter freshness rules when a run needs them.

## CI Mapping
When the builtin `github` profile is active, the generated source-tree CI
workflow lives at `.github/workflows/ci.yml`.
Its visible workflow name is `CI`.
Without an active CI-owner profile, DevCovenant does not generate that file
unless `config.ci_and_test.overrides` takes full local ownership.

The ownership split is:
1. the built-in `github` profile owns the generic GitHub Actions base workflow
2. active profiles may contribute `ci_and_test` fragments that extend it for a
   stack or shared custom profile
3. local `config.ci_and_test.*` keys are for small local overlays or, more
   rarely, a full local replacement

The github-owned base should stay generic.
The shipped user stack keeps `github` active by default, but the profile is
still optional: remove it when the repository does not want generated GitHub
Actions CI.
It bootstraps DevCovenant from the shipped
`devcovenant/runtime-requirements.lock` so it does not assume the
project dependency files belong to DevCovenant. The packaged
license files under `devcovenant/licenses/` ship with DevCovenant as part of
the package.
Workspace dependency setup belongs in the owning profile or config overlays,
not in the builtin GitHub base.
When that shipped lock pins `pip`, the generic workflow should install from
the lock instead of upgrading `pip` live first.
If a repository needs extra project dependency setup, extra CI steps, or extra
install validation, that extension should come from a profile-owned CI
fragment instead of from the builtin base.
Dependency vulnerability auditing should stay in
`dependency-management.surfaces`, not in a separate CI-only shell command.
That keeps local gates, local `check`, and generated CI on the same lock
health contract.
The same rule applies to Python security scanning: when the active
profile stack contributes Bandit, `security-scanner` owns that backend
through normal policy metadata instead of through a standalone workflow step.
That split usually looks like this:
1. a custom profile owns local behavior such as `root_workspace`,
   managed environment details, and repository workflow runs
2. an optional GitHub-specific custom profile owns reusable GitHub-only CI
   fragments when the repository wants more than the builtin base
When a repository shadows the builtin `github` profile with a same-name custom
profile, that custom profile fully replaces the builtin workflow template for
generated `ci.yml` ownership.
The same profile-merge order also feeds generated `.pre-commit-config.yaml`,
so changes to same-name shadow profiles should be documented as workflow
contract changes.
If a repository uses a hash-locked Python requirements file and also installs a
local wheel or sdist in CI, split that into:
1. install the locked requirements
2. install the local artifact with `--no-deps`
If a repository documents a particular public install path, test that same
public path in a project-owned CI extension.
If a repository adds a separate release workflow, that workflow should publish
validated CI artifacts instead of rebuilding a fresh distribution later.
When CI bootstraps a proof repository and needs `deploy`, flip the actual
`install.config_reviewed` field line in `devcovenant/config.yaml` rather than
replacing the first plain-text match in the commented config template.
If the proof also activates an extra profile such as `python_venv`, reuse the
existing `profiles.active` list-item indentation instead of reconstructing the
YAML block shape from the `active:` key line.

If you intentionally rebuild or re-baseline the changelog during an active
work session, run `devcovenant policy changelog-coverage reset-baseline` after
`devcovenant gate --open`. That command relaxes only the preserved-old-entry
rule for the active session. Normal changelog entry shape, date, summary, and
file-coverage checks still apply.

The generated `.github/workflows/ci.yml` file should stay aligned with the
same public lifecycle the CLI exposes locally:
- `devcovenant gate --open`
- `devcovenant gate --verify`
- `devcovenant run`
- `devcovenant gate --close`

That CI file is part of the workflow contract, not a second workflow model.

## Managed Environment In Workflow Execution
The shipped install baseline keeps `managed-environment` off for a normal
user repository.
When a repository explicitly opts into `python_venv` and enables
`policy_state.managed-environment`, DevCovenant chooses one target
environment for each stage and execution mode.
If the current interpreter already matches that setup, DevCovenant reuses it.
If not, it selects the configured interpreter or environment root and then runs
any declared bootstrap commands.
If a profile intentionally wants `bootstrap`-mode calls to keep running on the
current interpreter until the target environment exists, it must opt in with
`allow_current_interpreter_fallback: true`; otherwise the fallback stays
disabled so an incomplete profile does not get masked.

That keeps `gate --open` non-destructive once a configured environment already
exists.
It also keeps the workflow portable across normal `.venv` repositories,
bench-like environments, and other declared environment layouts.
With a stack that seeds a local `.venv`, `deploy`/`refresh` materializes the
workspace dependency artifacts, and profiles such as `python_venv` can
declare bootstrap-stage plus open-stage bootstrap commands when the target
environment is still missing.
If a repository uses a different environment shape, it should declare that
shape explicitly instead of expecting DevCovenant to guess it.
That includes system interpreters, bench-managed environments, and
container-managed environments.
DevCovenant either runs from that managed context already or resolves the
declared interpreter path or environment root for re-exec.
It does not infer hidden wrapper hops on its own.
If the managed environment keeps executables in additional locations, declare
`command_search_paths` so `required_commands` resolve against the managed
environment paths instead of the outer shell PATH.
On a converged repository, repeated `check`, `gate`, and `run` startup paths do
not intentionally rebuild current tracked registry content or current
dependency surfaces.
They compare tracked fingerprints first and only rebuild those artifacts when
policy, config, or dependency inputs actually changed.
Dependency vulnerability audits still need published advisory data, but the
runtime keeps those lookups in UTC-stamped machine-local cache entries rather
than writing tracked repo files or hiding the audit in CI-only glue.
Help paths and command-status paths should stay lighter still:
they should not initialize the full workflow runtime when they only need parser
output or local session inspection.
Workflow diagnostics should also stay repo-safe: when a gate or run reports a
path, it should prefer repo-relative rendering over leaking absolute checkout
roots.
The same environment-neutral behavior applies to `custom`; it is a lifecycle
command, not a special case that should assume a repo-local virtual
environment.

For Python-owned tools such as the pre-commit gate hook, execution runs
`python -m pre_commit` through the selected interpreter instead of depending on
a console-script shim.

## Recovery Rules
Use these recovery rules consistently:
- if `gate --open` fails, fix the reported complaint before editing
- if `gate --verify` fails, rerun `gate --verify` until clean before `run`
- if `run` fails, inspect run logs first, fix the cause, then rerun `run`
- if `gate --close` fails, inspect logs, refresh required workflow evidence,
  and rerun `gate --close`
- if you are unsure where a slice stands, use `devcovenant gate --status`

## Practical Rule
Profiles may extend the middle of the workflow by declaring runs, but they do
not get to redefine the four-stage sequence itself.
