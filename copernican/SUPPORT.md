# Support
**Doc ID:** SUPPORT
**Doc Type:** repo-support
**Project Version:** 12.0.26
**Last Updated:** 2026-06-02
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
