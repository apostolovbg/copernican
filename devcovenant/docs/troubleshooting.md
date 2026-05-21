# Troubleshooting
**Last Updated:** 2026-04-27
**Project Version:** 1.0.1b5

## Overview
Use this guide when DevCovenant is blocking work and you need the fastest clean
way back to a usable state.

The normal debug order is:
1. read the command's `summary.txt`
2. inspect `tail.txt` if present
3. inspect `stdout.log` and `stderr.log`
4. rerun the right command, not a random louder one

## Fast Triage
Ask these questions first:
1. was this a read-only audit or an active gate session?
2. which command failed?
3. did a gate stage change files?
4. is the problem in config, generated files, runtime state, or tests?

## Gate Failures
If `gate --open` fails, clear the reported problem before editing.
A failed `gate --open` is not a usable starting point.
If `gate --open` reports hook-changed paths or says managed files were
refreshed, the checkout was not settled yet. Apply or clear those changes,
then rerun `gate --open`.

If `gate --verify` fails, clear the issue and rerun `gate --verify` until it is
clean before `run`.

If `gate --close` fails, inspect the latest run logs, rerun `run` if required,
and then rerun `gate --close`.

## Changelog Coverage Problems
The two common causes are:
- the latest entry does not reflect the current slice
- the summary lines do not use accepted action verbs

When that happens, fix the top entry instead of adding noise below it.

## Registry Drift
If the tracked registry looks stale or inconsistent, run `devcovenant refresh`
or the normal gate workflow.
If the problem persists, inspect the owning profile, config, or descriptor
instead of hand-editing the registry.

## Config And Setup Problems
If behavior looks wrong, inspect these in order:
1. `devcovenant/config.yaml`
2. active profile descriptors and overlays
3. the tracked registry
4. `AGENTS.md` generated workflow, project-governance, and policy output

Most strange runtime behavior comes from config, profile ownership, or stale
tracked state rather than from a mysterious hidden engine rule.

## Managed Environment Problems
If the managed interpreter exists but is not executable, DevCovenant stops with
an explicit error.
Fix the interpreter path or permissions, then rerun the appropriate command.
If a gate hook fails before pre-commit starts, check whether the selected
interpreter still has the `pre_commit` package installed.
Gate execution runs `python -m pre_commit`, so the module still needs to exist
in the selected environment.

## Installed CLI Problems
If you installed DevCovenant with `pipx` and the `devcovenant` command is
missing, check the machine install first instead of debugging repository
config:
1. run `pipx list`
2. run `pipx ensurepath`
3. open a new shell and rerun `devcovenant --version`

If you are in a source checkout, do not confuse that machine install with the
repository-managed environment.
Use `python3 -m devcovenant ...` when the checkout does not expose the console
script directly.

## Translator And Profile Problems
If a language-specific policy path looks wrong, verify that the active language
profile owns the relevant translator and that no overlapping profile is trying
to claim the same extension ambiguously.

## Packaging Problems
When packaging or install validation fails, check:
- build logs
- package build configuration
- package metadata
- artifact validation and generation steps

If a repository also keeps a separate release workflow, inspect that workflow's
owning docs and config next instead of papering over the symptom in CI.
