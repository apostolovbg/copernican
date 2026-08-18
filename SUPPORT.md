# Support
**Doc ID:** SUPPORT
**Doc Type:** repo-support
**Project Version:** 12.0.26
**Last Updated:** 2026-08-18
**DevCovenant Version:** 1.0.1b6

## Table of Contents
- [Overview](#overview)
- [Self-Service Help](#self-service-help)
- [Filing A Good Report](#filing-a-good-report)
- [What Maintainers Need](#what-maintainers-need)

## Overview
Copernican support is documentation-first. Most operator questions should be
answerable from the README, the manual docs, the GUI help panel, or the
validation and dataset guides. That keeps routine usage self-service and
reduces the amount of guesswork needed when a run does not behave as expected.

Validation runs write their local summary marker to `~/VALIDATION.md`, so
the package guide in `copernican/validation/README.md` and the GUI validation
help stay aligned with the runtime path layout.

Troubleshooting should start with the worker-owned canonical log in the run
folder. The GUI Run Monitor displays the same worker events through a separate
in-memory transport and preserves their severity, but it does not write a
second log file. Sampling progress distinguishes completed iterations from
cumulative walker evaluations, so partial stages can be diagnosed from the
same counters shown in the CLI and GUI.
If the GUI opens nothing, confirm that the managed `.venv` is active
and that Copernican was launched with `python -m copernican --gui`.
Run and validation workers use that interpreter and the package import root.
If a worker reports `No module named copernican`, include the GUI launch
command and the first worker log lines in the support report.
The first worker summary must match the control model, test model, datasets,
seed, and sampler settings shown at confirmation. Include the saved run
manifest if those values differ.

If a model path fails to load, check the exact path, suffix, and
validation error that the CLI or GUI reports.

If a comparison is rejected, inspect both entries under
`selection.comparison` in the manifest. The control and test models must
declare matching observables, units, multipole grids, and spectrum roles.
The same pair can be selected in the GUI or overridden with
`--control-model` and `--test-model` in the CLI.
The GUI keeps the selected filenames attached to each role, so models with
the same display name remain distinct comparisons; the default control stays
`model_lcdm.yml` until a different control is chosen.

CMB-capable control and test models default to the Copernican declared
declared-graph CCMBS solver. The manifest records the independent
`selection.cmb_solver` choice and its capabilities; inspect `cmb_solver` and
`provenance.cmb_solver`, alongside each model's `declared_cmb_*` summaries,
when
diagnosing CMB provenance.

If launch behavior or GUI defaults look wrong, compare the run logs with
`copernican/workflow.py` and
`copernican/lib/global_settings/defaults.yml` first.

If the docs do not answer your question, the next best step is to file an
issue with enough context to let the maintainers reproduce the problem. Clear
reports are faster to fix than vague ones, and they make the changelog and the
docs easier to keep honest.

## Self-Service Help
Start with the front-door documentation:

- `README.md` for the repository-level workflow
- `ABOUT.md` for the project shape and documentation model
- `docs/cli_guide.md` for command-line usage
- `docs/gui_guide.md` for interactive usage
- `docs/run_manifest.md` for manifest and run-record details
- `copernican/validation/README.md` for the manifest runner and summary path

The shipped defaults live in `copernican/lib/global_settings/defaults.yml`.
The mutable `copernican_settings.yml` file lives at
`~/.config/copernican/copernican_settings.yml` on Unix or the platform
equivalent on Windows.

The GUI also exposes its own help page so users can review the same guidance
inside the application. When you are troubleshooting a run, the logs and the
saved manifest are often the fastest way to find the missing step.

## Filing A Good Report
A useful support report should answer four questions:

1. What were you trying to do?
2. What happened instead?
3. What environment were you using?
4. What did the logs or manifests show?

Include the command, the GUI action, or the file path that matters most.
If the issue involves data loading, mention the dataset and model names.
If the issue involves a run failure, include the relevant output directory and
the exact error text.

## What Maintainers Need
Maintainers can usually move faster when a report includes:

- the repository commit or branch
- the Python version and platform
- the exact command or GUI action
- the manifest or run directory if one exists
- the smallest log snippet that reproduces the failure

That information makes it possible to separate a user error from a real bug
without having to reconstruct the whole session from scratch.
